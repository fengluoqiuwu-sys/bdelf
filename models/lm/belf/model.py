"""BELF：块条件 rectified flow + 2L AdaLN-Zero G。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.latent.encdec.layers import TransformerBlock
from models.lm.belf.config import FL_BelfConfig
from models.lm.belf.generate import (
    MODE_DECODE,
    MODE_DENOISE,
    MODE_NONE,
    block_generate as run_block_generate,
)
from models.lm.belf_relf_core import (
    AdaLNZeroStack,
    LatentBundle,
    as_sdpa_mask,
    blend_v_tgt,
    check_time_step,
    interpolate,
    keep_params_in_graph,
    ladder_levels,
    maybe_drop_left,
    pack_2l,
    pack_2l_mask,
    sample_w_sc,
    v_star,
    validate_loaded_block,
    x_to_v,
)
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    split_model_cfg,
)
from models.rope import pair_positions
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg


def _whiten_vec(mod: nn.Module, names: tuple[str, ...], dim: int) -> torch.Tensor | None:
    """从入口模块取长度为 ``dim`` 的仿射向量；没有则 ``None``。"""
    for name in names:
        val = getattr(mod, name, None)
        if not isinstance(val, torch.Tensor):
            continue
        flat = val.detach().reshape(-1).float()
        if flat.numel() == dim:
            return flat.contiguous()
        if flat.numel() == 1:
            return flat.expand(dim).contiguous().clone()
    return None


class _CausalExit(nn.Module):
    """出口：等宽因果 decoder + Linear(D→V)，或仅 Linear。"""

    def __init__(
        self,
        *,
        kind: str,
        n_embd: int,
        n_head: int,
        vocab_size: int,
        n_layer_dec: int,
        dropout: float,
        attn_backend: str,
    ) -> None:
        super().__init__()
        kind = str(kind).strip().lower()
        if kind not in ("decoder", "linear"):
            raise ValueError(f"exit 须为 decoder|linear，收到 {kind!r}")
        self.kind = kind
        d_kv = n_embd // n_head
        d_ff = 4 * n_embd
        self.blocks = nn.ModuleList()
        if kind == "decoder":
            for _ in range(int(n_layer_dec)):
                self.blocks.append(
                    TransformerBlock(
                        n_embd, n_head, d_kv, d_ff, dropout,
                        use_flash=True, attn_backend=attn_backend, block_size=1,
                    )
                )
            self.ln = nn.LayerNorm(n_embd)
        else:
            self.ln = nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "decoder":
            for block in self.blocks:
                x = block(x, attn_mode="causal")
            x = self.ln(x)
        return self.lm_head(x)


class _BelfBackbone(nn.Module):
    """块条件流：训练 2L 并行，推理 ``block_generate``。"""

    full_sequence_training = True
    supports_prefix = True
    # 一次 forward 按槽拆 MSE/CE；不要 ELF decoder_prob 抽支。
    dual_branch_logging = False
    mixed_branch_training = False

    def __init__(self, config: FL_BelfConfig, bundle: LatentBundle) -> None:
        super().__init__()
        self.config = config
        self.token_layout = config.token_layout()
        self.max_seq_len = int(config.max_seq_len)
        self.n_embd = int(config.n_embd)
        self.n_head = int(config.n_head)
        self.block_size = int(config.block_size)
        if self.block_size < 1:
            raise ValueError(f"block_size W 须为正整数，收到 {config.block_size}")
        self.time_step = int(config.time_step)
        check_time_step(self.time_step)
        validate_loaded_block(
            family="belf",
            loaded_block=bundle.block_size,
            W=self.block_size,
        )
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd={self.n_embd} 须能被 n_head={self.n_head} 整除"
            )
        proj = str(config.proj_type).strip().lower()
        if proj != "linear":
            raise ValueError(f"proj_type 仅支持 linear，收到 {config.proj_type!r}")

        self.bundle = bundle
        self.latent_dim = int(bundle.latent_dim)
        self.sc_cfg = bool(config.sc_cfg)
        self.exit_kind = str(config.exit).strip().lower()
        self.ce_detach_g = bool(config.ce_detach_g)
        self.cond_mode = str(config.cond_mode)
        self.clean_block_prob = float(config.clean_block_prob)
        self.lambda_mse = float(config.lambda_mse)
        self.lambda_ce = float(config.lambda_ce)
        self.lambda_s1 = float(config.lambda_s1)
        self.p_mean = float(config.p_mean)
        self.p_std = float(config.p_std)
        self.t_eps = float(config.t_eps)
        self.vel_eps = float(config.vel_eps)
        self.sc_p_mean = float(config.sc_p_mean)
        self.sc_p_std = float(config.sc_p_std)
        self.w_sc_min = float(config.w_sc_min)
        self.w_sc_max = float(config.w_sc_max)
        self.sc_guided_prob = float(config.sc_guided_prob)
        self.ctx_drop_prob = float(config.ctx_drop_prob)
        self.denoiser_noise_scale = float(config.denoiser_noise_scale)
        self.attn_backend = str(config.attn_backend)

        x_dim = self.latent_dim
        mean = torch.zeros(x_dim)
        std = torch.ones(x_dim)
        src = getattr(bundle.latent, "backbone", bundle.latent)
        found_m = _whiten_vec(src, ("whitening_mean", "latent_mean"), x_dim)
        found_s = _whiten_vec(src, ("whitening_std", "latent_std"), x_dim)
        if found_m is not None:
            mean = found_m
        if found_s is not None:
            std = found_s
        self.register_buffer("whiten_mean", mean, persistent=True)
        self.register_buffer("whiten_std", std, persistent=True)

        self.in_proj = nn.Linear(x_dim, self.n_embd)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)

        head_dim = self.n_embd // self.n_head
        rope_dim = config.rope_dim
        if rope_dim is None:
            rope_dim = head_dim if head_dim % 2 == 0 else head_dim - 1

        self.g = AdaLNZeroStack(
            self.n_embd,
            int(config.n_layer),
            self.n_head,
            out_dim=self.n_embd,
            dropout=float(config.dropout),
            mlp_ratio=float(config.mlp_ratio),
            rope_dim=rope_dim,
            qk_norm=bool(config.qk_norm),
            attn_backend=self.attn_backend,
            num_time_tokens=0,
            use_scale=self.sc_cfg,
        )
        self.exit_head = _CausalExit(
            kind=self.exit_kind,
            n_embd=self.n_embd,
            n_head=self.n_head,
            vocab_size=int(self.token_layout.vocab_size),
            n_layer_dec=int(config.n_layer_dec),
            dropout=float(config.dropout),
            attn_backend=self.attn_backend,
        )

        levels = ladder_levels(
            self.time_step, self.p_mean, self.p_std, self.t_eps,
        )
        self.register_buffer("levels", levels, persistent=True)
        self._mask_cache: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}

        self.last_l2_loss = float("nan")
        self.last_ce_loss = float("nan")
        self.last_s1_loss = float("nan")

    def to_d(self, z: torch.Tensor) -> torch.Tensor:
        """白化后 Linear X→D。"""
        std = self.whiten_std.clamp(min=1e-8).to(dtype=z.dtype, device=z.device)
        mean = self.whiten_mean.to(dtype=z.dtype, device=z.device)
        return self.in_proj((z - mean) / std)

    def exit_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.exit_head(x)

    def on_tokens_seen(self, n: int, optimizer: Any = None) -> bool:
        return self.bundle.on_tokens_seen(n, optimizer)

    def _cached_2l_mask(
        self, left_len: int, right_len: int, device: torch.device,
    ) -> torch.Tensor:
        key = (left_len, right_len, self.block_size, device)
        cached = self._mask_cache.get(key)
        if cached is None:
            cached = as_sdpa_mask(
                pack_2l_mask(
                    left_len, right_len, self.block_size, self.block_size, device,
                )
            )
            if cached is None:
                raise RuntimeError("2L mask 不能为空")
            if len(self._mask_cache) > 16:
                self._mask_cache.pop(next(iter(self._mask_cache)))
            self._mask_cache[key] = cached
        return cached

    def forward_g(
        self,
        h_left: torch.Tensor,
        h_right: torch.Tensor,
        t_right: torch.Tensor | float,
        m_right_mode: int,
        w_sc: torch.Tensor | None,
        *,
        known_right: int = 0,
        drop_left: bool = False,
    ) -> torch.Tensor:
        """2L G，返回右段 x-pred。"""
        bsz, right_len, _ = h_right.shape
        left_len = int(h_left.size(1))
        device = h_right.device
        dtype = h_right.dtype
        left = h_left
        if drop_left:
            left = maybe_drop_left(left, self.ctx_drop_prob)
        packed = pack_2l(left, h_right)
        if isinstance(t_right, torch.Tensor):
            t_r = t_right.to(device=device, dtype=dtype)
            if t_r.ndim == 1:
                t_r = t_r[:, None].expand(bsz, right_len)
        else:
            t_r = h_right.new_full((bsz, right_len), float(t_right))
        m_r = torch.full(
            (bsz, right_len), int(m_right_mode), device=device, dtype=torch.long,
        )
        if known_right > 0:
            t_r = t_r.clone()
            t_r[:, :known_right] = 1.0
            m_r[:, :known_right] = MODE_NONE
        t_l = packed.new_ones(bsz, left_len)
        m_l = torch.zeros(bsz, left_len, device=device, dtype=torch.long)
        t_all = torch.cat([t_l, t_r], dim=1) if left_len > 0 else t_r
        m_all = torch.cat([m_l, m_r], dim=1) if left_len > 0 else m_r
        w_all = w_sc
        mask = self._cached_2l_mask(left_len, right_len, device)
        if left_len == right_len and left_len > 0:
            positions = pair_positions(left_len, device)
        else:
            pos_l = torch.arange(left_len, device=device, dtype=torch.long)
            pos_r = torch.arange(
                left_len, left_len + right_len, device=device, dtype=torch.long,
            )
            positions = torch.cat([pos_l, pos_r]) if left_len > 0 else pos_r
        x_hat = self.g(
            packed, t_all, w_all, m_all, attn_mask=mask, positions=positions,
        )
        return x_hat[:, left_len:] if left_len > 0 else x_hat

    def _masked_mean(
        self, per_token: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.to(dtype=per_token.dtype)
        denom = mask_f.sum().clamp(min=1.0)
        return (per_token * mask_f).sum() / denom

    def train_metrics(self) -> dict[str, float]:
        return {
            "mse": self.last_l2_loss,
            "ce": self.last_ce_loss,
            "denoise_mse": self.last_l2_loss,
            "decode_ce": self.last_ce_loss,
            "s1": self.last_s1_loss,
        }

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets, kwargs
        tokens = idx
        if tokens.ndim != 2:
            raise ValueError(f"idx 须为 (B, L)，收到 {tuple(tokens.shape)}")
        bsz, seq_len = tokens.shape
        w = self.block_size
        if seq_len % w != 0:
            raise ValueError(f"序列长度 {seq_len} 须能被 block_size {w} 整除")
        if seq_len > self.max_seq_len:
            raise ValueError(f"序列长度 {seq_len} 超过 max_seq_len={self.max_seq_len}")
        device = tokens.device
        levels = self.levels.to(device=device)

        z, mu, logvar = self.bundle.encode(tokens, sample=self.training)
        if z.shape[:2] != tokens.shape:
            raise ValueError(
                f"latent 形状 {tuple(z.shape[:2])} 须与 token {tuple(tokens.shape)} 一致"
            )
        x0 = self.to_d(z)
        dtype = x0.dtype
        pad_id = int(self.token_layout.pad_token_id)
        not_pad = tokens != pad_id

        hops = torch.randint(0, self.time_step, (bsz,), device=device)
        t_hop = levels[hops].to(dtype=dtype)
        is_decode = hops == (self.time_step - 1)

        rem = torch.zeros(bsz, device=device, dtype=torch.long)
        if self.cond_mode == "clean" and self.clean_block_prob > 0:
            use_mix = torch.rand(bsz, device=device) < self.clean_block_prob
            rem = torch.where(
                use_mix,
                torch.randint(1, w, (bsz,), device=device),
                rem,
            )
        local = torch.arange(seq_len, device=device) % w
        known = local[None, :] < rem[:, None]
        unknown = (~known) & not_pad
        denoise_mask = unknown & (~is_decode[:, None])
        decode_mask = unknown & is_decode[:, None]

        noise = torch.randn_like(x0) * self.denoiser_noise_scale
        t_right = torch.where(
            known,
            x0.new_ones(bsz, seq_len),
            t_hop[:, None].expand(bsz, seq_len),
        )
        z_t = interpolate(x0, t_right, noise)
        z_t = torch.where(known.unsqueeze(-1), x0, z_t)

        m_den = torch.full(
            (bsz, seq_len), MODE_DENOISE, device=device, dtype=torch.long,
        )
        m_dec = torch.full(
            (bsz, seq_len), MODE_DECODE, device=device, dtype=torch.long,
        )
        m_right = torch.where(is_decode[:, None], m_dec, m_den)
        m_right = torch.where(known, torch.zeros_like(m_right), m_right)

        h_left = maybe_drop_left(
            x0.detach(),
            self.ctx_drop_prob if self.training else 0.0,
        )
        w_sc = None
        guided: torch.Tensor | bool = False
        if self.sc_cfg:
            w_sc = sample_w_sc(
                bsz, self.sc_p_mean, self.sc_p_std,
                self.w_sc_min, self.w_sc_max, device,
            ).to(dtype=dtype)
            guided = torch.rand(bsz, device=device) < self.sc_guided_prob

        v_z = v_star(x0, z_t, t_right, self.vel_eps)
        # teacher 先于 student，避免与 student 图叠峰；v_c 复用 student.detach()。
        # 训练期始终跑 teacher，用 guided mask 混合（B 稍大时与「有 guided 才跑」同结果，无 .item()）。
        if self.sc_cfg and self.training:
            w_zero = torch.zeros_like(w_sc)
            with torch.no_grad():
                x_u = self._forward_g_m(h_left, z_t, t_right, m_right, w_zero)
                v_u = x_to_v(x_u, z_t, t_right, self.vel_eps)
            x_hat = self._forward_g_m(h_left, z_t, t_right, m_right, w_sc)
            v_c = x_to_v(x_hat.detach(), z_t, t_right, self.vel_eps)
            v_tgt = blend_v_tgt(v_z, v_u, v_c, w_sc, guided)
        else:
            x_hat = self._forward_g_m(h_left, z_t, t_right, m_right, w_sc)
            v_tgt = v_z

        v_hat = x_to_v(x_hat, z_t, t_right, self.vel_eps)
        l2_tok = (v_hat - v_tgt).pow(2).mean(dim=-1)
        zero = x0.new_zeros(())
        mse = self._masked_mean(l2_tok, denoise_mask)
        self.last_l2_loss = mse.detach()

        # 出口按样本因果，只跑 decode hop 行；与整批再 mask 的 CE 相同。
        dec_rows = is_decode.nonzero(as_tuple=True)[0]
        if dec_rows.numel() == 0:
            ce = zero + keep_params_in_graph(self.exit_head, zero)
            self.last_ce_loss = zero.new_full((), float("nan"))
        else:
            x_ce = x_hat.index_select(0, dec_rows)
            if self.ce_detach_g:
                x_ce = x_ce.detach()
            logits = self.exit_logits(x_ce)
            ce_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tokens.index_select(0, dec_rows).reshape(-1),
                ignore_index=self.token_layout.ignore_index,
                reduction="none",
            ).view(dec_rows.size(0), seq_len)
            ce = self._masked_mean(ce_tok, decode_mask.index_select(0, dec_rows))
            self.last_ce_loss = ce.detach()
            del logits

        s1 = self.bundle.s1_loss(tokens, z=z, mu=mu, logvar=logvar)
        if self.bundle.is_trainable:
            self.last_s1_loss = s1.detach()
        else:
            self.last_s1_loss = nan
            s1 = zero.to(dtype=x0.dtype)

        loss = self.lambda_mse * mse + self.lambda_ce * ce + self.lambda_s1 * s1
        empty = tokens.new_zeros(
            bsz, 0, int(self.token_layout.vocab_size),
        )
        return empty, loss

    def _forward_g_m(
        self,
        h_left: torch.Tensor,
        h_right: torch.Tensor,
        t_right: torch.Tensor,
        m_right: torch.Tensor,
        w_sc: torch.Tensor | None,
    ) -> torch.Tensor:
        """与 ``forward_g`` 相同，但右段 ``m`` 逐列给定。"""
        bsz, right_len, _ = h_right.shape
        left_len = int(h_left.size(1))
        device = h_right.device
        packed = pack_2l(h_left, h_right)
        t_l = packed.new_ones(bsz, left_len)
        m_l = torch.zeros(bsz, left_len, device=device, dtype=torch.long)
        t_all = torch.cat([t_l, t_right], dim=1)
        m_all = torch.cat([m_l, m_right], dim=1)
        mask = self._cached_2l_mask(left_len, right_len, device)
        positions = pair_positions(left_len, device) if left_len == right_len else (
            torch.cat([
                torch.arange(left_len, device=device, dtype=torch.long),
                torch.arange(left_len, left_len + right_len, device=device, dtype=torch.long),
            ])
        )
        x_hat = self.g(
            packed, t_all, w_sc, m_all, attn_mask=mask, positions=positions,
        )
        return x_hat[:, left_len:]

    @torch.compiler.disable
    @torch.no_grad()
    def block_generate(
        self,
        num_samples: int = 1,
        seqlen: int | None = None,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        bos_token_id: int | None = None,
        prefix_tokens: torch.Tensor | None = None,
        sampling_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        return run_block_generate(
            self,
            num_samples,
            seqlen,
            temperature=temperature,
            top_k=top_k,
            bos_token_id=bos_token_id,
            prefix_tokens=prefix_tokens,
            sampling_cfg=sampling_cfg,
        )

    @torch.compiler.disable
    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        seqlen: int | None = None,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        bos_token_id: int | None = None,
        prefix_tokens: torch.Tensor | None = None,
        sampling_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        return self.block_generate(
            num_samples,
            seqlen,
            temperature=temperature,
            top_k=top_k,
            bos_token_id=bos_token_id,
            prefix_tokens=prefix_tokens,
            sampling_cfg=sampling_cfg,
        )


class FL_BelfModel(FL_PreTrainedModel):
    config_class = FL_BelfConfig

    def __init__(self, config: FL_BelfConfig, bundle: LatentBundle) -> None:
        super().__init__(config)
        self.backbone = _BelfBackbone(config, bundle)
        self.post_init()
        self._restore_adaln_zero_init()

    def _restore_adaln_zero_init(self) -> None:
        """post_init 会打乱 AdaLN-Zero / FinalLayer 零初始化；按 Cola 意图补回。"""
        stack = self.backbone.g
        for block in stack.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(stack.final.adaLN_modulation[-1].weight)
        nn.init.zeros_(stack.final.adaLN_modulation[-1].bias)
        nn.init.zeros_(stack.final.linear.weight)
        nn.init.zeros_(stack.final.linear.bias)
        nn.init.zeros_(stack.mode_embed.weight)

    def on_tokens_seen(self, n: int, optimizer: Any = None) -> bool:
        return self.backbone.on_tokens_seen(n, optimizer)


def build_model_from_config(
    config: FL_BelfConfig,
    *,
    variant: str | None = None,
    load_latent_weights: bool = True,
) -> FL_BelfModel:
    del variant
    ensure_token_layout(config)
    if not load_latent_weights:
        raise ValueError("BELF 入口须经 LatentBundle 加载 artifacts，禁止跳过权重")
    bundle = LatentBundle(
        latent_model=config.latent_model,
        tag=config.tag,
        tune=config.latent_tune,
        latent_thaw_tokens=config.latent_thaw_tokens,
        lambda_vae=config.lambda_vae,
        lambda_ref=config.lambda_ref,
    )
    print(f"[belf] loaded latent {bundle.latent_model}/{bundle.tag}")
    return FL_BelfModel(config, bundle)


def build_model(model_cfg: dict) -> FL_BelfModel:
    data, sampling = split_model_cfg(model_cfg)
    data.pop("train_variant", None)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_BelfConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
