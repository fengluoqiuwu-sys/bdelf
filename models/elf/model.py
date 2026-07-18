"""Embedded Language Flows (ELF) — continuous diffusion LM (no CFG).

Follows arXiv:2605.10938 with self-conditioning but without classifier-free
guidance. Training uses a frozen T5 encoder for clean embeddings; inference
only needs the denoiser + unembedding head.

References:
  - ELF: https://arxiv.org/abs/2605.10938
  - Official PyTorch: https://github.com/lillian039/ELF/tree/pytorch_elf
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.elf.config import FL_ELFConfig
from models.elf.layers import (
    BottleneckTextProj,
    ELFBlock,
    FinalLayer,
    TextRotaryEmbedding,
    TimestepEmbedder,
    _normal_002_,
    make_linear,
)
from models.elf.t5_encoder import (
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


class _ELFBackbone(nn.Module):
    """ELF DiT backbone with dual-branch (denoise MSE / decode CE) training."""

    full_sequence_training = True
    dual_branch_logging = True

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
        num_model_mode_tokens: int = 4,
        self_cond_prob: float = 0.5,
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
        time_schedule: str = "logit_normal",
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
        self.num_model_mode_tokens = num_model_mode_tokens
        self.self_cond_prob = self_cond_prob
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
        self.last_loss_branch = ""

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

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(
                torch.empty(1, num_model_mode_tokens, n_embd)
            )
            _normal_002_(self.mode_tokens)
        else:
            self.mode_tokens = None

        prefix_total = num_model_mode_tokens + num_time_tokens
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

    def _ensure_encoder(self) -> T5Encoder:
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

    def encode_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        encoder = self._ensure_encoder()
        return encode_text(
            idx,
            encoder,
            latent_mean=self.latent_mean,
            latent_std=self.latent_std,
        ).to(dtype=next(self.parameters()).dtype)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Network forward
    # ------------------------------------------------------------------

    def _build_time_prefix(self, t: torch.Tensor) -> torch.Tensor:
        bsz = t.shape[0]
        time_emb = self.t_embedder(t)
        return self.t_emb_tokens.expand(bsz, -1, -1) + time_emb.unsqueeze(1)

    def net_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        decoder_step_active: bool | torch.Tensor | None = None,
        deterministic: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``x`` is (B, S, C) or (B, S, 2C) with self-conditioning."""
        bsz = x.shape[0]
        # Match official ELF: keep embedding projections in fp32 under AMP.
        with torch.amp.autocast("cuda", enabled=False):
            x_f = x.float()
            if x_f.shape[-1] == 2 * self.text_encoder_dim:
                x_f = self.self_cond_proj(x_f)
            x_h = self.text_proj(x_f)
            prefix = self._build_time_prefix(t.float()).to(dtype=x_h.dtype)

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
            if decoder_step_active is not None:
                need_logits = True
                if isinstance(decoder_step_active, torch.Tensor):
                    need_logits = bool(decoder_step_active.any().item())
                elif decoder_step_active is False:
                    need_logits = False
                if need_logits:
                    xf = x_h.float()
                    hidden = F.gelu(
                        xf @ self.proj_kernel + self.proj_bias,
                        approximate="tanh",
                    )
                    decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias
            x_pred = self.final_layer(x_h.float())
        return x_pred, decoder_logits

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _sample_train_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.time_schedule == "logit_normal":
            z = (
                torch.randn(batch_size, device=device) * self.denoiser_p_std
                + self.denoiser_p_mean
            )
            t = torch.sigmoid(z)
        else:
            t = torch.rand(batch_size, device=device)
        # Avoid singular (1-t) in v-target; matches BDELF / Flow Matching practice.
        return t.clamp(max=1.0 - self.t_eps)

    def _x_to_v(
        self, x_pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        t_exp = t.reshape(-1, 1, 1)
        return (x_pred - z) / torch.clamp(1.0 - t_exp, min=self.t_eps)

    def _denoise_loss(self, x0: torch.Tensor) -> torch.Tensor:
        bsz = x0.shape[0]
        device = x0.device
        t = self._sample_train_t(bsz, device).to(dtype=x0.dtype)
        noise = torch.randn_like(x0) * self.denoiser_noise_scale
        t_exp = t.reshape(-1, 1, 1)
        z = t_exp * x0 + (1.0 - t_exp) * noise
        v_target = (x0 - z) / torch.clamp(1.0 - t_exp, min=self.t_eps)

        use_sc = (
            self.self_cond_prob > 0
            and torch.rand((), device=device) < self.self_cond_prob
        )
        if use_sc:
            with torch.no_grad():
                z_sc0 = torch.cat([z, torch.zeros_like(z)], dim=-1)
                x_init, _ = self.net_forward(
                    z_sc0, t, decoder_step_active=False, deterministic=True,
                )
            model_in = torch.cat([z, x_init.detach()], dim=-1)
        elif self.self_cond_prob > 0:
            model_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
        else:
            model_in = z

        x_pred, _ = self.net_forward(
            model_in, t, decoder_step_active=False, deterministic=False,
        )
        v_pred = self._x_to_v(x_pred, z, t)
        return ((v_pred - v_target) ** 2).mean()

    def _decode_loss(self, x0: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
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
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens.reshape(-1),
            ignore_index=self.token_layout.ignore_index,
        )

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
        x0 = self.encode_tokens(idx)
        if branch == "decode":
            loss = self._decode_loss(x0, idx)
            self.last_loss_branch = "decode"
        elif branch == "denoise":
            loss = self._denoise_loss(x0)
            self.last_loss_branch = "denoise"
        else:
            if torch.rand((), device=idx.device) < self.decoder_prob:
                loss = self._decode_loss(x0, idx)
                self.last_loss_branch = "decode"
            else:
                loss = self._denoise_loss(x0)
                self.last_loss_branch = "denoise"

        # Keep both heads + mode tokens in the graph every step so DDP can run
        # with find_unused_parameters=False (denoise skips unembed; decode skips
        # final_layer).
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
        if self.self_cond_prob > 0:
            touch = touch + self.self_cond_proj.weight.sum() + self.self_cond_proj.bias.sum()
        loss = loss + 0.0 * touch
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

    def _forward_sample(
        self,
        z: torch.Tensor,
        t: float,
        x_pred_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = z.shape[0]
        t_batch = torch.full((bsz,), t, dtype=z.dtype, device=z.device)
        if self.self_cond_prob > 0:
            sc = (
                torch.zeros_like(z)
                if x_pred_prev is None
                else x_pred_prev
            )
            model_in = torch.cat([z, sc], dim=-1)
        else:
            model_in = z
        x_pred, _ = self.net_forward(
            model_in, t_batch, decoder_step_active=False, deterministic=True,
        )
        v = self._x_to_v(x_pred, z, t_batch)
        return v, x_pred

    def _ode_step(
        self,
        z: torch.Tensor,
        t: float,
        t_next: float,
        x_pred_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v, x_pred = self._forward_sample(z, t, x_pred_prev)
        return z + (t_next - t) * v, x_pred

    def _sde_step(
        self,
        z: torch.Tensor,
        t: float,
        t_next: float,
        x_pred_prev: torch.Tensor | None,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = float(t_next - t)
        alpha = max(0.0, min(1.0, 1.0 - gamma * h))
        t_back = alpha * float(t)
        eps = torch.randn_like(z) * self.denoiser_noise_scale
        z_back = alpha * z + (1.0 - alpha) * eps
        v, x_pred = self._forward_sample(z_back, t_back, x_pred_prev)
        return z_back + (t_next - t_back) * v, x_pred

    @torch.no_grad()
    def _decode_tokens(
        self,
        z: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        bsz = z.shape[0]
        t = torch.ones(bsz, device=z.device, dtype=z.dtype)
        if self.self_cond_prob > 0:
            model_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
        else:
            model_in = z
        _, logits = self.net_forward(
            model_in, t, decoder_step_active=True, deterministic=True,
        )
        assert logits is not None
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

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        t_steps = self._get_sampling_steps(num_sampling_steps, device, dtype)
        z = (
            torch.randn(
                num_samples, seqlen, self.text_encoder_dim, device=device, dtype=dtype,
            )
            * self.denoiser_noise_scale
        )
        x_pred: torch.Tensor | None = None
        nfe = 0

        # Intermediate steps: ODE or SDE; final interval always ODE.
        for i in range(t_steps.numel() - 2):
            t = float(t_steps[i].item())
            t_next = float(t_steps[i + 1].item())
            if method == "sde":
                z, x_pred = self._sde_step(z, t, t_next, x_pred, sde_gamma)
            elif method == "ode":
                z, x_pred = self._ode_step(z, t, t_next, x_pred)
            else:
                raise ValueError(f"unknown sampling_method: {method}")
            nfe += 1

        t = float(t_steps[-2].item())
        t_next = float(t_steps[-1].item())
        z, x_pred = self._ode_step(z, t, t_next, x_pred)
        nfe += 1

        tokens = self._decode_tokens(z, temperature=temperature, top_k=top_k)
        nfe += 1
        return tokens, nfe


class FL_ELFModel(FL_PreTrainedModel):
    config_class = FL_ELFConfig

    def __init__(self, config: FL_ELFConfig) -> None:
        super().__init__(config)
        self.backbone = _ELFBackbone(**config.backbone_kwargs())

    def count_parameters(self) -> int:
        """Trainable params only (exclude frozen T5 encoder)."""
        return self.backbone.trainable_parameter_count()


def build_model_from_config(config: FL_ELFConfig) -> FL_ELFModel:
    ensure_token_layout(config)
    # Populate cache before training starts (auto-download if missing).
    ensure_t5_encoder_cached(config.encoder_model_name)
    return FL_ELFModel(config)


def build_model(cfg: dict) -> FL_ELFModel:
    data, sampling = split_model_cfg(cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    config = FL_ELFConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
