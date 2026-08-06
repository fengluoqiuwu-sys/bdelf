"""LateCE 变体 B：ELF 双分支 + 晚段轨迹 CE。

训练：per-example 混合 ELF 式 t=1 decode CE（``decoder_prob``）与 denoise
MSE/x-pred（t~U(0,1) 可改）；denoise 行在晚段时间窗内额外叠加
``late_ce_weight`` 加权的轨迹 token CE。推理末步 unembed（decode mode）。
见 ``temp/idea/late-ce/``。
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.late_ce.config import FL_LateCEConfig
from models.late_ce.layers import (
    BottleneckTextProj,
    ELFBlock,
    FinalLayer,
    TextRotaryEmbedding,
    TimestepEmbedder,
    _normal_002_,
    make_linear,
)
from models.late_ce.t5_encoder import (
    T5Encoder,
    encode_text,
    ensure_t5_encoder_cached,
    load_t5_encoder,
)
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    sample_from_logits,
    split_model_cfg,
)
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg
from models.tokens import FL_TokenLayout


class _LateCEBackbone(nn.Module):
    """DiT backbone：ELF 双分支（denoise MSE + decode CE）+ 晚窗轨迹 CE。"""

    full_sequence_training = True
    dual_branch_logging = True
    # 复用 train.py mixed 日志路径：记 mse / late_ce（无 decode_ce）
    mixed_branch_training = True

    def __init__(
        self,
        token_layout: FL_TokenLayout,
        max_seq_len: int = 1024,
        encoder_model_name: str = "t5-small",
        text_encoder_dim: int = 512,
        bottleneck_dim: int = 128,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 0,
        num_model_mode_tokens: int = 4,
        self_cond_prob: float = 0.5,
        self_cond_cfg_min: float = 0.5,
        self_cond_cfg_max: float = 5.0,
        latent_mean: float = 0.0,
        latent_std: float = 0.2,
        denoiser_p_mean: float = -1.5,
        denoiser_p_std: float = 0.8,
        denoiser_noise_scale: float = 2.0,
        decoder_prob: float = 0.2,
        decoder_p_mean: float = 0.8,
        decoder_p_std: float = 0.8,
        decoder_noise_scale: float = 5.0,
        t_eps: float = 0.05,
        time_schedule: str = "uniform",
        late_ce_mode: str = "hard",
        late_ce_delta: float = 0.2,
        late_ce_alpha: float = 10.0,
        late_ce_weight: float = 1.0,
        late_ce_region: str = "late",
    ) -> None:
        super().__init__()
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive")

        self.token_layout = token_layout
        self.vocab_size = token_layout.vocab_size
        self.max_seq_len = max_seq_len
        self.encoder_model_name = encoder_model_name
        self.text_encoder_dim = text_encoder_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.self_cond_prob = self_cond_prob
        self.self_cond_cfg_min = self_cond_cfg_min
        self.self_cond_cfg_max = self_cond_cfg_max
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.denoiser_p_mean = denoiser_p_mean
        self.denoiser_p_std = denoiser_p_std
        self.denoiser_noise_scale = denoiser_noise_scale
        self.decoder_prob = decoder_prob
        self.decoder_p_mean = decoder_p_mean
        self.decoder_p_std = decoder_p_std
        self.decoder_noise_scale = decoder_noise_scale
        self.t_eps = t_eps
        self.time_schedule = time_schedule
        self.late_ce_mode = str(late_ce_mode).lower()
        self.late_ce_delta = float(late_ce_delta)
        self.late_ce_alpha = float(late_ce_alpha)
        self.late_ce_weight = float(late_ce_weight)
        self.late_ce_region = str(late_ce_region).lower()
        if self.late_ce_mode not in ("hard", "soft"):
            raise ValueError(
                f"late_ce_mode must be 'hard' or 'soft', got {late_ce_mode!r}"
            )
        if self.late_ce_region not in ("late", "early"):
            raise ValueError(
                f"late_ce_region must be 'late' or 'early', got {late_ce_region!r}"
            )
        self.last_loss_branch = ""
        self.last_l2_loss = float("nan")
        self.last_ce_loss = float("nan")
        self.last_late_ce_loss = float("nan")

        # Lazy frozen T5; held in a list so PyTorch does not auto-register it
        # as a submodule (avoids DDP / checkpoint surprises).
        self._encoder_holder: list[T5Encoder] = []

        self.self_cond_proj = make_linear(
            2 * text_encoder_dim, text_encoder_dim, bias=True,
        )
        self.text_proj = BottleneckTextProj(
            text_encoder_dim, n_embd, bottleneck_dim,
        )

        self.t_embedder = TimestepEmbedder(n_embd)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, n_embd))
        _normal_002_(self.t_emb_tokens)

        # Optional training-time SC-CFG scale tokens (absent on pre-CFG checkpoints).
        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(n_embd)
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.empty(1, num_self_cond_cfg_tokens, n_embd)
            )
            _normal_002_(self.self_cond_cfg_tokens)
        else:
            self.self_cond_cfg_embedder = None
            self.self_cond_cfg_tokens = None

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(
                torch.empty(1, num_model_mode_tokens, n_embd)
            )
            _normal_002_(self.mode_tokens)
        else:
            self.mode_tokens = None

        prefix_total = (
            num_model_mode_tokens + num_time_tokens + max(num_self_cond_cfg_tokens, 0)
        )
        head_dim = n_embd // n_head
        self.rope = TextRotaryEmbedding(
            head_dim, max_seq_len, num_prefix_tokens=prefix_total,
        )

        # Mid-depth dropout (official ELF): layers in [depth/4, 3*depth/4).
        q1, q3 = n_layer // 4, n_layer // 4 * 3
        blocks = []
        for i in range(n_layer):
            in_drop = q3 > i >= q1
            blocks.append(
                ELFBlock(
                    n_embd,
                    n_head,
                    mlp_ratio=mlp_ratio,
                    attn_drop=dropout if in_drop else 0.0,
                    proj_drop=dropout if in_drop else 0.0,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = FinalLayer(n_embd, text_encoder_dim)

        # Factored unembedding: hidden → encoder_dim → vocab
        self.proj_kernel = nn.Parameter(torch.empty(n_embd, text_encoder_dim))
        self.proj_bias = nn.Parameter(torch.empty(text_encoder_dim))
        self.unembed_kernel = nn.Parameter(
            torch.empty(text_encoder_dim, token_layout.vocab_size)
        )
        self.unembed_bias = nn.Parameter(torch.empty(token_layout.vocab_size))
        nn.init.xavier_uniform_(self.proj_kernel)
        nn.init.zeros_(self.proj_bias)
        nn.init.xavier_uniform_(self.unembed_kernel)
        nn.init.zeros_(self.unembed_bias)

    # ------------------------------------------------------------------
    # Encoder / checkpoint helpers
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _ensure_encoder(self) -> T5Encoder:
        # Must not run under Dynamo: load_t5_encoder → HF from_pretrained.
        device = next(self.parameters()).device
        if not self._encoder_holder:
            _, enc = load_t5_encoder(self.encoder_model_name)
            self._encoder_holder.append(enc.to(device))
        else:
            enc = self._encoder_holder[0]
            enc_device = next(enc.parameters()).device
            if enc_device != device:
                self._encoder_holder[0] = enc.to(device)
        return self._encoder_holder[0]

    @torch.compiler.disable
    def encode_tokens(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder = self._ensure_encoder()
        return encode_text(
            idx,
            encoder,
            attention_mask=attention_mask,
            latent_mean=self.latent_mean,
            latent_std=self.latent_std,
        ).to(dtype=next(self.parameters()).dtype)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _token_loss_mask(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, S) float mask: 1 = valid token (exclude pad), matching official."""
        return (idx != self.token_layout.pad_token_id).to(
            dtype=next(self.parameters()).dtype
        )

    # ------------------------------------------------------------------
    # Network forward
    # ------------------------------------------------------------------

    def build_context(
        self,
        t: torch.Tensor,
        self_cond_cfg_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """In-context conditioning prefix: time + optional SC-CFG scale tokens.

        Mirrors official ``ELF.build_context``; both use TimestepEmbedder +
        learned prefix tokens added to the embedding.
        """
        bsz = t.shape[0]
        time_emb = self.t_embedder(t)
        prefix_tokens = [
            self.t_emb_tokens.expand(bsz, -1, -1) + time_emb.unsqueeze(1)
        ]
        if (
            self_cond_cfg_scale is not None
            and self.num_self_cond_cfg_tokens > 0
        ):
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            prefix_tokens.append(
                self.self_cond_cfg_tokens.expand(bsz, -1, -1)
                + sc_emb.unsqueeze(1)
            )
        return torch.cat(prefix_tokens, dim=1)

    def net_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        decoder_step_active: bool | torch.Tensor | None = None,
        deterministic: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        self_cond_cfg_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``x`` is (B, S, C) or (B, S, 2C) with self-conditioning.

        ``self_cond_cfg_scale`` is an optional (B,) SC-CFG scale prepended as
        in-context prefix tokens.
        """
        bsz = x.shape[0]
        # 对齐官方 ELF：投影段关 AMP、跟随权重精度（训练权重 fp32 → 等价于
        # 原 .float()；generate.py 以 bf16 加载时不再与权重 dtype 冲突）。
        param_dtype = self.text_proj.proj1.weight.dtype
        with torch.amp.autocast("cuda", enabled=False):
            x_f = x.to(dtype=param_dtype)
            if x_f.shape[-1] == 2 * self.text_encoder_dim:
                x_f = self.self_cond_proj(x_f)
            x_h = self.text_proj(x_f)
            sc_cfg_scale_emb = (
                self_cond_cfg_scale.to(dtype=param_dtype)
                if self_cond_cfg_scale is not None
                else None
            )
            prefix = self.build_context(
                t.to(dtype=param_dtype),
                self_cond_cfg_scale=sc_cfg_scale_emb,
            ).to(dtype=x_h.dtype)

        if self.mode_tokens is not None:
            mode = self.mode_tokens.expand(bsz, -1, -1).to(dtype=x_h.dtype)
            if decoder_step_active is None:
                gate = 0.0
            elif isinstance(decoder_step_active, torch.Tensor) and decoder_step_active.dim() > 0:
                gate = decoder_step_active.to(dtype=mode.dtype).view(-1, 1, 1)
            else:
                gate = float(decoder_step_active)
            mode = mode * gate
            x_h = torch.cat([mode, x_h], dim=1)
            mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones(
                    bsz, self.num_model_mode_tokens,
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)
        else:
            mode_offset = 0

        prefix_len = prefix.shape[1]
        x_h = torch.cat([prefix, x_h], dim=1)
        if attention_mask is not None:
            prefix_mask = torch.ones(
                bsz, prefix_len,
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        for block in self.blocks:
            x_h = block(
                x_h,
                self.rope,
                attention_mask=attention_mask,
                deterministic=deterministic,
            )

        x_h = x_h[:, prefix_len + mode_offset :]

        with torch.amp.autocast("cuda", enabled=False):
            decoder_logits = None
            # Official: whenever decoder_step_active is provided (incl. a (B,)
            # tensor that may be all zeros), always run the unembed head so
            # mixed-branch DDP never sees unused parameters.
            if decoder_step_active is not None:
                xf = x_h.to(dtype=self.proj_kernel.dtype)
                hidden = F.gelu(
                    xf @ self.proj_kernel + self.proj_bias,
                    approximate="tanh",
                )
                decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias
            x_pred = self.final_layer(x_h.to(dtype=param_dtype))
        return x_pred, decoder_logits

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _sample_cfg_scale(
        self, batch_size: int, device: torch.device,
    ) -> torch.Tensor:
        """Log-uniform CFG scale in [1+min, 1+max] - 1 (official sample_cfg_scale)."""
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
        a = torch.as_tensor(1.0 + self.self_cond_cfg_min, device=device, dtype=u.dtype)
        b = torch.as_tensor(1.0 + self.self_cond_cfg_max, device=device, dtype=u.dtype)
        return (a * torch.exp(u * torch.log(b / a)) - 1.0).to(
            dtype=next(self.parameters()).dtype,
        )

    def _sample_train_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """训练时间采样。默认 ``uniform``：t~U(0,1)，使 hard 窗 CE 占比≈δ。"""
        schedule = (self.time_schedule or "uniform").lower()
        if schedule == "logit_normal":
            z = (
                torch.randn(batch_size, device=device) * self.denoiser_p_std
                + self.denoiser_p_mean
            )
            return torch.sigmoid(z)
        if schedule in ("uniform", "linear"):
            return torch.rand(batch_size, device=device)
        raise ValueError(
            f"unknown time_schedule {self.time_schedule!r}; "
            "expected 'uniform' or 'logit_normal'"
        )

    def _x_to_v(
        self, x_pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        t_exp = t.reshape(-1, 1, 1)
        return (x_pred - z) / torch.clamp(1.0 - t_exp, min=self.t_eps)

    def _masked_mean(
        self, per_token: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean of ``per_token`` (B, S) over positions where ``mask`` (B, S) > 0."""
        mask_f = mask.to(dtype=per_token.dtype)
        return (per_token * mask_f).sum() / torch.clamp(mask_f.sum(), min=1.0)

    def _late_ce_gate(self, t: torch.Tensor) -> torch.Tensor:
        """按 t 返回 LateCE 门控权重 (B,)；hard∈{0,1}，soft∈(0,1)。"""
        delta = max(float(self.late_ce_delta), 0.0)
        if self.late_ce_region == "early":
            if self.late_ce_mode == "hard":
                return (t <= delta).to(dtype=t.dtype)
            return torch.sigmoid(self.late_ce_alpha * (delta - t))
        threshold = 1.0 - delta
        if self.late_ce_mode == "hard":
            return (t >= threshold).to(dtype=t.dtype)
        return torch.sigmoid(self.late_ce_alpha * (t - threshold))

    def _denoise_loss(
        self,
        x0: torch.Tensor,
        *,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """纯 denoise MSE（forced eval / 调试用；对齐 ELF）。"""
        bsz = x0.shape[0]
        device = x0.device
        t = self._sample_train_t(bsz, device).to(dtype=x0.dtype)
        noise = torch.randn_like(x0) * self.denoiser_noise_scale
        t_exp = t.reshape(-1, 1, 1)
        z = t_exp * x0 + (1.0 - t_exp) * noise
        v_target = (x0 - z) / torch.clamp(1.0 - t_exp, min=self.t_eps)

        if self.self_cond_prob > 0:
            use_sc = (
                (torch.rand((bsz,), device=device, dtype=x0.dtype) < self.self_cond_prob)
                .reshape(-1, 1, 1)
                .to(dtype=x0.dtype)
            )
            with torch.no_grad():
                z_sc0 = torch.cat([z, torch.zeros_like(z)], dim=-1)
                x_init, _ = self.net_forward(
                    z_sc0, t, decoder_step_active=False, deterministic=True,
                )
            sc_half = x_init.detach() * use_sc
            model_in = torch.cat([z, sc_half], dim=-1)
        else:
            model_in = z

        x_pred, _ = self.net_forward(
            model_in, t, decoder_step_active=False, deterministic=False,
        )
        v_pred = self._x_to_v(x_pred, z, t)
        l2_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)
        return self._masked_mean(l2_per_token, loss_mask)

    def _decode_loss(
        self,
        x0: torch.Tensor,
        tokens: torch.Tensor,
        *,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """t=1 decode CE（eval 用；对齐 ELF，与其 eval decode ppl 可比）。"""
        bsz, seq_len, _ = x0.shape
        device = x0.device
        z_vals = (
            torch.randn(bsz * seq_len, device=device, dtype=x0.dtype)
            * self.decoder_p_std
            + self.decoder_p_mean
        )
        lam = torch.sigmoid(z_vals).reshape(bsz, seq_len, 1)
        noise = torch.randn_like(x0) * self.decoder_noise_scale
        z_tilde = lam * x0 + (1.0 - lam) * noise
        t = torch.ones(bsz, device=device, dtype=x0.dtype)

        if self.self_cond_prob > 0:
            model_in = torch.cat([z_tilde, torch.zeros_like(z_tilde)], dim=-1)
        else:
            model_in = z_tilde

        _, logits = self.net_forward(
            model_in, t, decoder_step_active=True, deterministic=False,
        )
        assert logits is not None
        log_probs = F.log_softmax(logits.float(), dim=-1)
        ce_per_token = -log_probs.gather(
            -1, tokens.unsqueeze(-1),
        ).squeeze(-1)
        return self._masked_mean(ce_per_token, loss_mask)

    def _train_loss(
        self,
        x0: torch.Tensor,
        tokens: torch.Tensor,
        *,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """变体 B：ELF per-example denoise/decode 混合 + denoise 行晚窗轨迹 CE。"""
        bsz, seq_len, _ = x0.shape
        device = x0.device
        dtype = x0.dtype

        t = self._sample_train_t(bsz, device).to(dtype=dtype)
        noise = torch.randn_like(x0) * self.denoiser_noise_scale
        t_exp = t.reshape(-1, 1, 1)
        denoiser_z = t_exp * x0 + (1.0 - t_exp) * noise
        v_target = (x0 - denoiser_z) / torch.clamp(1.0 - t_exp, min=self.t_eps)

        # ELF decode 分支：sigmoid-λ corruption + t=1 + mode ON（per-example）。
        decoder_step_active = torch.bernoulli(
            torch.full((bsz,), self.decoder_prob, dtype=torch.float32, device=device),
        ).to(dtype=dtype)
        decoder_mask_b11 = decoder_step_active.view(-1, 1, 1)
        decoder_mask_b1 = decoder_step_active.view(-1, 1)

        decoder_z_vals = (
            torch.randn((bsz * seq_len,), dtype=dtype, device=device)
            * self.decoder_p_std
            + self.decoder_p_mean
        )
        decoder_lam = torch.sigmoid(decoder_z_vals).reshape(bsz, seq_len, 1)
        decoder_noise = torch.randn_like(x0) * self.decoder_noise_scale
        decoder_z = decoder_lam * x0 + (1.0 - decoder_lam) * decoder_noise

        decoder_t = torch.ones_like(t)
        t_mixed = decoder_step_active * decoder_t + (1.0 - decoder_step_active) * t
        z_mixed = decoder_mask_b11 * decoder_z + (1.0 - decoder_mask_b11) * denoiser_z

        self_cond_cfg: torch.Tensor | None = None
        if self.num_self_cond_cfg_tokens > 0:
            self_cond_cfg = self._sample_cfg_scale(bsz, device)
            use_sc = (
                (torch.rand((bsz,), device=device, dtype=dtype) < self.self_cond_prob)
                .reshape(-1, 1, 1)
                .to(dtype=dtype)
            )
            with torch.no_grad():
                z_sc0 = torch.cat([denoiser_z, torch.zeros_like(denoiser_z)], dim=-1)
                x_init, _ = self.net_forward(
                    z_sc0, t, decoder_step_active=False, deterministic=True,
                    self_cond_cfg_scale=self_cond_cfg,
                )
            v_uncond = self._x_to_v(x_init, denoiser_z, t)
            x_uncond = x_init.detach()
            with torch.no_grad():
                z_sc1 = torch.cat([denoiser_z, x_uncond], dim=-1)
                x_cond, _ = self.net_forward(
                    z_sc1, t, decoder_step_active=False, deterministic=True,
                    self_cond_cfg_scale=self_cond_cfg,
                )
            v_cond = self._x_to_v(x_cond, denoiser_z, t)
            sc_w = self_cond_cfg.reshape(-1, 1, 1)
            sc_guidance = (1.0 - 1.0 / sc_w) * (v_cond - v_uncond)
            sc_guidance = torch.where(
                use_sc.bool(), sc_guidance, torch.zeros_like(sc_guidance),
            )
            v_target = (v_target + sc_guidance).detach()
            # decode 行自条件半边清零（对齐 ELF）。
            sc_half = x_uncond * use_sc * (1.0 - decoder_mask_b11)
            model_in = torch.cat([z_mixed, sc_half], dim=-1)
        elif self.self_cond_prob > 0:
            use_sc = (
                (torch.rand((bsz,), device=device, dtype=dtype) < self.self_cond_prob)
                .reshape(-1, 1, 1)
                .to(dtype=dtype)
            )
            with torch.no_grad():
                z_sc0 = torch.cat([denoiser_z, torch.zeros_like(denoiser_z)], dim=-1)
                x_init, _ = self.net_forward(
                    z_sc0, t, decoder_step_active=False, deterministic=True,
                )
            sc_half = x_init.detach() * use_sc * (1.0 - decoder_mask_b11)
            model_in = torch.cat([z_mixed, sc_half], dim=-1)
        else:
            model_in = z_mixed

        # decode 行 mode ON；unembed 恒计算（denoise 行 logits 供晚窗 CE）。
        x_pred, logits = self.net_forward(
            model_in,
            t_mixed,
            decoder_step_active=decoder_step_active,
            deterministic=False,
            self_cond_cfg_scale=self_cond_cfg,
        )
        v_pred = self._x_to_v(x_pred, denoiser_z, t)
        l2_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)

        assert logits is not None
        log_probs = F.log_softmax(logits.float(), dim=-1)
        ce_per_token = -log_probs.gather(
            -1, tokens.unsqueeze(-1),
        ).squeeze(-1)

        loss_mask_f = loss_mask.to(dtype=ce_per_token.dtype)
        ce_mask = loss_mask_f * decoder_mask_b1
        l2_mask = loss_mask_f * (1.0 - decoder_mask_b1)
        # LateCE：仅 denoise 行、原始 t 落在晚窗时叠加轨迹 CE。
        late_gate = self._late_ce_gate(t)
        late_mask = l2_mask * late_gate.view(-1, 1)
        total_sum = (
            (ce_per_token * ce_mask).sum()
            + (l2_per_token * l2_mask).sum()
            + self.late_ce_weight * (ce_per_token * late_mask).sum()
        )
        loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0)

        # 分支指标：保持 detach 张量（不 .item()），避免 torch.compile 图断裂。
        ce_denom = ce_mask.sum()
        l2_denom = l2_mask.sum()
        late_denom = late_mask.sum()
        nan_scalar = torch.full(
            (), float("nan"),
            device=ce_per_token.device, dtype=ce_per_token.dtype,
        )
        self.last_ce_loss = torch.where(
            ce_denom > 0,
            (ce_per_token * ce_mask).sum() / ce_denom.clamp(min=1.0),
            nan_scalar,
        ).detach()
        self.last_l2_loss = torch.where(
            l2_denom > 0,
            (l2_per_token * l2_mask).sum() / l2_denom.clamp(min=1.0),
            nan_scalar,
        ).detach()
        self.last_late_ce_loss = torch.where(
            late_denom > 0,
            (ce_per_token * late_mask).sum() / late_denom.clamp(min=1.0),
            nan_scalar,
        ).detach()
        return loss

    def _touch_unused_heads(self, loss: torch.Tensor) -> torch.Tensor:
        """单分支 forced eval / 调试时保住两头参数在图内（对齐 ELF）。"""
        touch = (
            self.final_layer.linear.weight.sum()
            + self.final_layer.linear.bias.sum()
            + self.final_layer.norm_final.weight.sum()
            + self.proj_kernel.sum()
            + self.proj_bias.sum()
            + self.unembed_kernel.sum()
            + self.unembed_bias.sum()
        )
        if self.mode_tokens is not None:
            touch = touch + self.mode_tokens.sum()
        if self.self_cond_cfg_tokens is not None:
            touch = (
                touch
                + self.self_cond_cfg_embedder.mlp_0.weight.sum()
                + self.self_cond_cfg_embedder.mlp_2.weight.sum()
                + self.self_cond_cfg_tokens.sum()
            )
        if self.self_cond_prob > 0:
            touch = (
                touch
                + self.self_cond_proj.weight.sum()
                + self.self_cond_proj.bias.sum()
            )
        return loss + 0.0 * touch

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        branch: Literal["denoise", "decode"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets
        if idx.size(1) > self.max_seq_len:
            raise ValueError(
                f"sequence length {idx.size(1)} exceeds max_seq_len {self.max_seq_len}"
            )
        loss_mask = self._token_loss_mask(idx)
        x0 = self.encode_tokens(idx, attention_mask=loss_mask.long())

        if branch == "decode":
            loss = self._decode_loss(x0, idx, loss_mask=loss_mask)
            self.last_loss_branch = "decode"
            self.last_ce_loss = loss.detach()
            self.last_l2_loss = float("nan")
            self.last_late_ce_loss = float("nan")
            loss = self._touch_unused_heads(loss)
        elif branch == "denoise":
            loss = self._denoise_loss(x0, loss_mask=loss_mask)
            self.last_loss_branch = "denoise"
            self.last_l2_loss = loss.detach()
            self.last_ce_loss = float("nan")
            self.last_late_ce_loss = float("nan")
            loss = self._touch_unused_heads(loss)
        else:
            # 默认 / 训练：变体 B per-example 混合（两头都在图内）。
            loss = self._train_loss(x0, idx, loss_mask=loss_mask)
            self.last_loss_branch = "mixed"
        return torch.empty(0), loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _get_sampling_steps(
        self, num_steps: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        schedule = getattr(self, "_infer_time_schedule", self.time_schedule)
        if schedule == "logit_normal":
            z = (
                torch.randn(num_steps - 1, device=device, dtype=dtype)
                * self.denoiser_p_std
                + self.denoiser_p_mean
            )
            interior = torch.sigmoid(z).sort().values
            return torch.cat(
                [
                    torch.zeros(1, device=device, dtype=dtype),
                    interior,
                    torch.ones(1, device=device, dtype=dtype),
                ]
            )
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)

    def _forward_self_cond(
        self,
        z: torch.Tensor,
        t_batch: torch.Tensor,
        sc_prev: torch.Tensor | None,
        self_cond_cfg_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One SC-CFG-aware denoiser forward; returns (v, x_pred)."""
        if self.self_cond_prob > 0:
            sc = torch.zeros_like(z) if sc_prev is None else sc_prev
            model_in = torch.cat([z, sc], dim=-1)
        else:
            model_in = z
        x_pred, _ = self.net_forward(
            model_in, t_batch, decoder_step_active=False, deterministic=True,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
        v = self._x_to_v(x_pred, z, t_batch)
        return v, x_pred

    def _forward_sample(
        self,
        z: torch.Tensor,
        t: float,
        x_pred_prev: torch.Tensor | None,
        self_cond_cfg_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoiser forward with SC-CFG (matches official ``_forward_sample_self_cond``).

        With training-time scale tokens (``num_self_cond_cfg_tokens > 0``): one
        forward conditioned on ``w`` and the previous self-cond estimate.
        Without scale tokens: optional inference-time extrapolation
        ``v/x = uncond + w * (cond - uncond)``.
        """
        bsz = z.shape[0]
        t_batch = torch.full((bsz,), t, dtype=z.dtype, device=z.device)

        # Official: training-time SC-CFG → single forward; network already
        # learned guided v for the provided w.
        if self.num_self_cond_cfg_tokens > 0:
            return self._forward_self_cond(
                z, t_batch, x_pred_prev, self_cond_cfg_scale,
            )

        w = (
            float(self_cond_cfg_scale[0].item())
            if self_cond_cfg_scale is not None
            else 1.0
        )
        if self.self_cond_prob <= 0:
            return self._forward_self_cond(z, t_batch, x_pred_prev, None)

        need_uncond = w != 1.0 or x_pred_prev is None
        v_uncond = x_uncond = None
        if need_uncond:
            v_uncond, x_uncond = self._forward_self_cond(z, t_batch, None, None)
            if w == 0.0 or x_pred_prev is None:
                return v_uncond, x_uncond

        v_cond, x_cond = self._forward_self_cond(z, t_batch, x_pred_prev, None)
        if w == 1.0:
            return v_cond, x_cond
        assert v_uncond is not None and x_uncond is not None
        v_out = v_uncond + w * (v_cond - v_uncond)
        x_out = x_uncond + w * (x_cond - x_uncond)
        return v_out, x_out

    def _ode_step(
        self,
        z: torch.Tensor,
        t: float,
        t_next: float,
        x_pred_prev: torch.Tensor | None,
        self_cond_cfg_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v, x_pred = self._forward_sample(
            z, t, x_pred_prev, self_cond_cfg_scale=self_cond_cfg_scale,
        )
        return z + (t_next - t) * v, x_pred

    def _sde_step(
        self,
        z: torch.Tensor,
        t: float,
        t_next: float,
        x_pred_prev: torch.Tensor | None,
        gamma: float,
        self_cond_cfg_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = float(t_next - t)
        alpha = max(0.0, min(1.0, 1.0 - gamma * h))
        t_back = alpha * float(t)
        eps = torch.randn_like(z) * self.denoiser_noise_scale
        z_back = alpha * z + (1.0 - alpha) * eps
        v, x_pred = self._forward_sample(
            z_back, t_back, x_pred_prev, self_cond_cfg_scale=self_cond_cfg_scale,
        )
        return z_back + (t_next - t_back) * v, x_pred

    @staticmethod
    def _mask_after_eos(
        tokens: torch.Tensor, *, eos_token_id: int, pad_token_id: int,
    ) -> torch.Tensor:
        """Replace the first EOS and everything after it with pad (official ELF)."""
        eos_mask = tokens == eos_token_id
        keep_mask = eos_mask.to(torch.int32).cumsum(dim=1) == 0
        return torch.where(keep_mask, tokens, torch.full_like(tokens, pad_token_id))

    @torch.no_grad()
    def _decode_tokens(
        self,
        z: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        self_cond_cfg_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = z.shape[0]
        t = torch.ones(bsz, device=z.device, dtype=z.dtype)
        if self.self_cond_prob > 0:
            model_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
        else:
            model_in = z
        _, logits = self.net_forward(
            model_in, t, decoder_step_active=True, deterministic=True,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
        assert logits is not None
        if not torch.isfinite(logits).all():
            raise RuntimeError("ELF decode produced non-finite logits")
        if temperature <= 0:
            return logits.argmax(dim=-1)
        return sample_from_logits(logits, temperature=temperature, top_k=top_k)

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        seqlen: int | None = None,
        num_steps: int | None = None,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        bos_token_id: int | None = None,
        prefix_tokens: torch.Tensor | None = None,
        sampling_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        del bos_token_id, prefix_tokens  # unconditional ELF
        cfg = sampling_cfg or {}
        if seqlen is None:
            seqlen = self.max_seq_len
        if seqlen > self.max_seq_len:
            raise ValueError(
                f"seqlen {seqlen} exceeds max_seq_len {self.max_seq_len}"
            )

        method = str(cfg.get("sampling_method", "sde")).lower()
        num_sampling_steps = int(
            num_steps if num_steps is not None else cfg.get("num_sampling_steps", 32)
        )
        sde_gamma = float(cfg.get("sde_gamma", 1.5))
        temperature = float(cfg.get("temperature", temperature))
        top_k = cfg.get("top_k", top_k)
        if top_k is not None:
            top_k = int(top_k)
        infer_schedule = cfg.get("time_schedule")
        if infer_schedule is not None:
            self._infer_time_schedule = infer_schedule
        sc_cfg_w = float(cfg.get("self_cond_cfg_scale", 1.0))

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        t_steps = self._get_sampling_steps(num_sampling_steps, device, dtype)
        z = (
            torch.randn(
                num_samples, seqlen, self.text_encoder_dim, device=device, dtype=dtype,
            )
            * self.denoiser_noise_scale
        )
        # Always feed w when scale tokens exist (incl. w=1); else only if w!=1.
        if self.num_self_cond_cfg_tokens > 0:
            self_cond_cfg_scale = torch.full(
                (num_samples,), sc_cfg_w, dtype=dtype, device=device,
            )
        elif sc_cfg_w != 1.0:
            self_cond_cfg_scale = torch.full(
                (num_samples,), sc_cfg_w, dtype=dtype, device=device,
            )
        else:
            self_cond_cfg_scale = None
        x_pred: torch.Tensor | None = None
        nfe = 0

        # Intermediate steps: ODE or SDE; final interval always ODE.
        for i in range(t_steps.numel() - 2):
            t = float(t_steps[i].item())
            t_next = float(t_steps[i + 1].item())
            if method == "sde":
                z, x_pred = self._sde_step(
                    z, t, t_next, x_pred, sde_gamma,
                    self_cond_cfg_scale=self_cond_cfg_scale,
                )
            elif method == "ode":
                z, x_pred = self._ode_step(
                    z, t, t_next, x_pred,
                    self_cond_cfg_scale=self_cond_cfg_scale,
                )
            else:
                raise ValueError(f"unknown sampling_method: {method}")
            nfe += 1

        t = float(t_steps[-2].item())
        t_next = float(t_steps[-1].item())
        z, x_pred = self._ode_step(
            z, t, t_next, x_pred,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
        nfe += 1
        if not torch.isfinite(z).all():
            raise RuntimeError("ELF sampling produced non-finite latents")

        tokens = self._decode_tokens(
            z,
            temperature=temperature,
            top_k=top_k,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
        nfe += 1
        tokens = self._mask_after_eos(
            tokens,
            eos_token_id=self.token_layout.eos_token_id,
            pad_token_id=self.token_layout.pad_token_id,
        )
        return tokens, nfe


class FL_LateCEModel(FL_PreTrainedModel):
    config_class = FL_LateCEConfig

    def __init__(self, config: FL_LateCEConfig) -> None:
        super().__init__(config)
        self.backbone = _LateCEBackbone(**config.backbone_kwargs())

    def count_parameters(self) -> int:
        """Trainable params only (exclude frozen T5 encoder)."""
        return self.backbone.trainable_parameter_count()


def build_model_from_config(config: FL_LateCEConfig) -> FL_LateCEModel:
    ensure_token_layout(config)
    # Populate cache before training starts (auto-download if missing).
    ensure_t5_encoder_cached(config.encoder_model_name)
    return FL_LateCEModel(config)


def build_model(cfg: dict) -> FL_LateCEModel:
    data, sampling = split_model_cfg(cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    config = FL_LateCEConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
