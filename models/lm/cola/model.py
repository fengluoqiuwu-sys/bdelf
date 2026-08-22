"""Cola DLM Stage-2：块因果 DiT latent prior + 已加载 Text VAE。

训练 / 推理性质对齐官方 Cola-DLM 源码（``cola_dlm/``），规模除外：
- Flow Matching：``z_t = (1-t) z_0 + t z_1``，``t∈(0,1)``，AdaLN 输入 ``t * ode_T``
- 2L pack：``[clean | noisy]`` + 官方 2L mask，一次前向监督全部块
- Stage-2 VAE 项：重建 + ``β E[log q_φ]`` + mask（不再对 N(0,I) 做 KL）
- 推理：``linspace(T,0)`` Euler、CFG 无条件 = 空前缀
"""

from __future__ import annotations

import copy
import math
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lm.cola.config import FL_ColaConfig
from models.lm.cola.infer import sample_block_ode
from models.lm.cola.layers import (
    DiTBlock,
    FinalLayer,
    TimestepEmbedder,
    block_causal_mask_mod,
    build_block_causal_mask,
    build_cola_2l_mask,
    cola_2l_mask_mod,
)
from models.lm.cola.vae_loader import load_vae_backbone
from models.latent.cola_vae.layers import (
    FLEX_ATTN_AVAILABLE,
    bool_mask_to_sdpa_additive,
    create_block_mask,
)
from models.latent.cola_vae.model import _ColaVAEBackbone
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    sample_from_logits,
    split_model_cfg,
)
from models.rope import pair_positions
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg


class _ColaDiT(nn.Module):
    """块因果 DiT，连续 latent 上的速度场。"""

    def __init__(
        self,
        *,
        latent_dim: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        max_seq_len: int,
        diffusion_block_size: int,
        dropout: float = 0.0,
        rope_dim: int | None = None,
        qk_norm: bool = True,
        expand_ratio: int = 4,
        attn_backend: str = "flex",
    ) -> None:
        super().__init__()
        if attn_backend == "flex" and not FLEX_ATTN_AVAILABLE:
            raise RuntimeError(
                "attn_backend=flex 需要 PyTorch FlexAttention；请升级或改用 sdpa"
            )
        if attn_backend not in ("flex", "sdpa"):
            raise ValueError(f"unknown attn_backend: {attn_backend}")
        self.latent_dim = latent_dim
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        self.diffusion_block_size = diffusion_block_size
        self.attn_backend = attn_backend
        self.input_proj = nn.Linear(latent_dim, n_embd)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        self.t_embedder = TimestepEmbedder(n_embd)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    n_embd, n_head, dropout=dropout,
                    rope_dim=rope_dim, qk_norm=qk_norm, expand_ratio=expand_ratio,
                    attn_backend=attn_backend,
                )
                for _ in range(n_layer)
            ]
        )
        self.final = FinalLayer(n_embd, latent_dim)
        self._mask_cache: dict[tuple[str, str, int, torch.device], object] = {}

    def _cached_mask(self, kind: str, seq_len: int, device: torch.device):
        """``kind=2l`` 时 ``seq_len`` 为单侧 L；``kind=block`` 时为全长。"""
        key = (self.attn_backend, kind, seq_len, device)
        cached = self._mask_cache.get(key)
        if cached is None:
            if kind == "2l":
                full = 2 * seq_len
                if self.attn_backend == "flex":
                    mask_mod = partial(
                        cola_2l_mask_mod,
                        block_size=self.diffusion_block_size,
                        n=seq_len,
                    )
                    cached = create_block_mask(
                        mask_mod, B=None, H=None, Q_LEN=full, KV_LEN=full, device=device,
                    )
                else:
                    cached = bool_mask_to_sdpa_additive(
                        build_cola_2l_mask(seq_len, self.diffusion_block_size, device),
                    )
            else:
                if self.attn_backend == "flex":
                    mask_mod = partial(
                        block_causal_mask_mod,
                        block_size=self.diffusion_block_size,
                    )
                    cached = create_block_mask(
                        mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
                    )
                else:
                    cached = bool_mask_to_sdpa_additive(
                        build_block_causal_mask(seq_len, self.diffusion_block_size, device),
                    )
            if len(self._mask_cache) > 16:
                self._mask_cache.pop(next(iter(self._mask_cache)))
            self._mask_cache[key] = cached
        return cached

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        mask_kind: str = "block",
    ) -> torch.Tensor:
        """预测 ``v(z_t, t)``，形状与 ``z_t`` 相同。

        ``t``：``(B,)`` 广播，或 ``(B, L)`` 逐位置（2L 训练时干净半段为 0）。
        """
        del attn_mask
        seq_len = z_t.size(1)
        x = self.input_proj(z_t)
        cond = self.t_embedder(t)
        mask_seq = seq_len // 2 if mask_kind == "2l" else seq_len
        mask = self._cached_mask(mask_kind, mask_seq, z_t.device)
        flex_mask = mask if self.attn_backend == "flex" else None
        sdpa_mask = None if self.attn_backend == "flex" else mask
        if positions is None:
            positions = torch.arange(seq_len, device=z_t.device, dtype=torch.long)
        for block in self.blocks:
            x = block(
                x, cond,
                flex_block_mask=flex_mask, sdpa_attn_mask=sdpa_mask, positions=positions,
            )
        return self.final(x, cond)


class _ColaBackbone(nn.Module):
    """Stage-2：联合 VAE + DiT prior。"""

    full_sequence_training = True

    def __init__(self, config: FL_ColaConfig, vae: _ColaVAEBackbone) -> None:
        super().__init__()
        self.config = config
        self.token_layout = config.token_layout()
        self.max_seq_len = config.max_seq_len
        self.latent_dim = config.latent_dim
        self.diffusion_block_size = config.diffusion_block_size
        self.denoiser_p_mean = config.denoiser_p_mean
        self.denoiser_p_std = config.denoiser_p_std
        self.time_schedule = config.time_schedule
        self.schedule_loc = config.schedule_loc
        self.t_eps = config.t_eps
        self.ode_T = float(config.ode_T)
        self.lambda_vae = config.lambda_vae
        self.lambda_fm = config.lambda_fm
        self.lambda_ref = config.lambda_ref
        self.freeze_vae = config.freeze_vae
        self.scaling_factor = float(getattr(vae, "scaling_factor", 1.0))
        self.shifting_factor = float(getattr(vae, "shifting_factor", 0.0))

        self.vae = vae
        self.vae.beta_kl = config.beta_kl
        self.vae.lambda_mask = config.lambda_mask
        self.vae.mask_ratio = config.mask_ratio

        if config.latent_dim != vae.latent_dim:
            raise ValueError(
                f"cola latent_dim={config.latent_dim} != vae.latent_dim={vae.latent_dim}"
            )
        vae_bs = getattr(vae, "block_size", config.diffusion_block_size)
        if int(vae_bs) != int(config.diffusion_block_size):
            raise ValueError(
                f"cola diffusion_block_size={config.diffusion_block_size} "
                f"!= vae.block_size={vae_bs}"
            )

        head_dim = config.n_embd // config.n_head
        rope_dim = config.rope_dim
        if rope_dim is None:
            rope_dim = (3 * head_dim) // 4
            if rope_dim % 2:
                rope_dim -= 1

        self.dit = _ColaDiT(
            latent_dim=config.latent_dim,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            max_seq_len=config.max_seq_len,
            diffusion_block_size=config.diffusion_block_size,
            dropout=config.dropout,
            rope_dim=rope_dim,
            qk_norm=config.qk_norm,
            expand_ratio=config.expand_ratio,
            attn_backend=config.attn_backend,
        )

        self._ref_encoder = copy.deepcopy(vae)
        for p in self._ref_encoder.parameters():
            p.requires_grad_(False)
        self._ref_encoder.eval()

        if self.freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad_(False)

        self.last_l2_loss = float("nan")
        self.last_ce_loss = float("nan")
        self.last_kl_loss = float("nan")

    def _to_dit_latent(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.shifting_factor) * self.scaling_factor

    def _from_dit_latent(self, z: torch.Tensor) -> torch.Tensor:
        scale = self.scaling_factor if abs(self.scaling_factor) > 1e-8 else 1.0
        return z / scale + self.shifting_factor

    def _sample_train_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # 论文 / 官方：logit-normal，loc≈1；t=0 干净、t=1 噪声。
        if self.time_schedule == "logit_normal":
            mean = self.denoiser_p_mean + self.schedule_loc
            z = torch.randn(batch_size, device=device) * self.denoiser_p_std + mean
            t = torch.sigmoid(z)
        else:
            t = torch.rand(batch_size, device=device)
        return t.clamp(min=self.t_eps, max=1.0 - self.t_eps)

    def _fm_loss(self, z0: torch.Tensor) -> torch.Tensor:
        """官方 2L Flow Matching：全部块并行，目标速度 ``z_1 - z_0``。"""
        bsz, seq_len, _ = z0.shape
        t = self._sample_train_t(bsz, z0.device)
        t_exp = t[:, None, None]
        z1 = torch.randn_like(z0)
        z_t = (1.0 - t_exp) * z0 + t_exp * z1
        v_tgt = z1 - z0
        # [clean | noisy]；干净半段 stop-grad，时间嵌入为 0。
        inp = torch.cat([z0.detach(), z_t], dim=1)
        t_tokens = torch.zeros(bsz, 2 * seq_len, device=z0.device, dtype=z0.dtype)
        t_tokens[:, seq_len:] = (t * self.ode_T)[:, None]
        positions = pair_positions(seq_len, z0.device)
        v_hat = self.dit(inp, t_tokens, positions=positions, mask_kind="2l")
        v_noisy = v_hat[:, seq_len:]
        return (v_noisy - v_tgt).pow(2).mean()

    def _ref_kl(self, tokens: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, mu_ref, logvar_ref = self._ref_encoder.encode(tokens, sample=False)
        var = logvar.exp()
        var_ref = logvar_ref.exp()
        kl = 0.5 * (
            (logvar_ref - logvar)
            + (var + (mu - mu_ref).pow(2)) / var_ref.clamp(min=1e-6)
            - 1.0
        )
        return kl.mean()

    def _stage2_vae_loss(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage-2：CE + β E[log q_φ] + mask。"""
        ce = F.cross_entropy(
            logits.reshape(-1, self.vae.vocab_size),
            tokens.reshape(-1),
            ignore_index=self.token_layout.ignore_index,
        )
        # 对角高斯 E[log q] = -0.5 (1 + log(2π) + logvar)；最小化 +β E[log q] ≡ 最大化熵。
        log_q = -0.5 * (1.0 + math.log(2.0 * math.pi) + logvar)
        entropy_term = log_q.mean()
        mask_loss = torch.zeros((), device=tokens.device, dtype=ce.dtype)
        if (
            self.training
            and not self.freeze_vae
            and self.vae.lambda_mask > 0
            and self.vae.mask_ratio > 0
        ):
            mask_loss = self.vae.bert_mask_loss(tokens)
        vae_loss = ce + self.vae.beta_kl * entropy_term + self.vae.lambda_mask * mask_loss
        return vae_loss, ce, entropy_term

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets, kwargs
        tokens = idx
        if self.freeze_vae:
            with torch.no_grad():
                z0, mu, logvar = self.vae.encode(tokens, sample=self.training)
                logits = self.vae.decode_logits(z0)
        else:
            z0, mu, logvar = self.vae.encode(tokens, sample=self.training)
            logits = self.vae.decode_logits(z0)

        vae_loss, ce, entropy_term = self._stage2_vae_loss(tokens, logits, logvar)
        # 保持 detach 张量，避免 torch.compile 因 .item() 图断裂。
        self.last_ce_loss = ce.detach()
        self.last_kl_loss = entropy_term.detach()

        if not self.training:
            self.last_l2_loss = torch.full(
                (), float("nan"), device=tokens.device, dtype=ce.dtype,
            )
            return logits, ce + self.vae.beta_kl * entropy_term

        z0_fm = z0 if not self.freeze_vae else z0.detach()
        fm_loss = self._fm_loss(self._to_dit_latent(z0_fm))

        ref_loss = torch.zeros((), device=tokens.device, dtype=fm_loss.dtype)
        if self.lambda_ref > 0 and not self.freeze_vae:
            ref_loss = self._ref_kl(tokens, mu, logvar)

        loss = (
            self.lambda_vae * vae_loss
            + self.lambda_fm * fm_loss
            + self.lambda_ref * ref_loss
        )
        self.last_l2_loss = fm_loss.detach()
        return logits, loss

    def _dit_velocity_on_window(
        self, z_full: torch.Tensor, t_batch: torch.Tensor, block_len: int,
    ) -> torch.Tensor:
        """生成时 DiT 前向固定为 ``max_seq_len``，避免 16、32、… 换形状。

        真实 token 靠左、零 pad 在右。块因果下当前块看不到未来 pad，
        因此只保留一套 Flex mask。返回真实序列末 ``block_len`` 的速度。
        """
        real_len = int(z_full.size(1))
        if real_len < block_len:
            raise ValueError(
                f"DiT 窗口长度 {real_len} 短于 block_len={block_len}"
            )
        pad_to = int(self.max_seq_len)
        if real_len > pad_to:
            raise ValueError(
                f"generate 窗口 {real_len} 超过 max_seq_len={pad_to}"
            )
        if real_len < pad_to:
            bsz, _, dim = z_full.shape
            pad_n = pad_to - real_len
            z_full = torch.cat(
                [z_full, z_full.new_zeros(bsz, pad_n, dim)], dim=1,
            )
            if t_batch.ndim == 2:
                t_batch = torch.cat(
                    [t_batch, t_batch.new_zeros(bsz, pad_n)], dim=1,
                )
        v = self.dit(z_full, t_batch)
        return v[:, real_len - block_len : real_len]

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
        cfg = sampling_cfg or {}
        seqlen = int(seqlen or self.max_seq_len)
        db = self.diffusion_block_size
        if seqlen % db != 0:
            raise ValueError(f"seqlen {seqlen} must be divisible by block_size {db}")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        num_steps = int(cfg.get("num_ode_steps", 16))
        cfg_scale = float(cfg.get("cfg_scale", 7.0))
        ode_T = float(cfg.get("ode_T", self.ode_T))
        temperature = float(cfg.get("temperature", temperature))
        top_k = cfg.get("top_k", top_k)

        prefix_len = 0
        if prefix_tokens is not None:
            prefix = prefix_tokens.to(device=device, dtype=torch.long)
            if prefix.size(0) != num_samples:
                raise ValueError("prefix_tokens batch must match num_samples")
            prefix_len = int(prefix.size(1))
            if prefix_len >= seqlen:
                raise ValueError("prefix length must be < seqlen")
            n_full_blocks = prefix_len // db
            if n_full_blocks > 0:
                prefix_full = prefix[:, : n_full_blocks * db]
                z_prefix, _, _ = self.vae.encode(prefix_full, sample=False)
                clean = self._to_dit_latent(z_prefix)
            else:
                clean = torch.zeros(
                    num_samples, 0, self.latent_dim, device=device, dtype=dtype,
                )
            start_block = n_full_blocks
        else:
            _ = bos_token_id
            clean = torch.zeros(
                num_samples, 0, self.latent_dim, device=device, dtype=dtype,
            )
            start_block = 0

        n_blocks = seqlen // db
        nfe = 0
        # DiT 在 _dit_velocity_on_window 内 pad 到 max_seq_len，整段 generate 共用一块 mask。
        for _b in range(start_block, n_blocks):
            z_block = torch.randn(
                num_samples, db, self.latent_dim, device=device, dtype=dtype,
            )

            def predict_v(z_full: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
                nonlocal nfe
                nfe += 1
                return self._dit_velocity_on_window(z_full, t_batch, db)

            z_block = sample_block_ode(
                predict_v=predict_v,
                z_init=z_block,
                clean_prefix=clean if clean.size(1) > 0 else None,
                num_steps=num_steps,
                cfg_scale=cfg_scale if clean.size(1) > 0 else 1.0,
                ode_T=ode_T,
            )
            clean = torch.cat([clean, z_block], dim=1)

        z_all = self._from_dit_latent(clean[:, :seqlen])
        logits = self.vae.decode_logits(z_all)
        # 官方任务评测默认 temperature=0 → argmax；≤0 走 argmax。
        if temperature <= 0:
            tokens = logits.argmax(dim=-1)
        else:
            tokens = sample_from_logits(logits, temperature=temperature, top_k=top_k)
        if prefix_len > 0:
            tokens = tokens.clone()
            tokens[:, :prefix_len] = prefix_tokens.to(tokens.device)
        return tokens, nfe


class FL_ColaModel(FL_PreTrainedModel):
    config_class = FL_ColaConfig

    def __init__(self, config: FL_ColaConfig, vae: _ColaVAEBackbone) -> None:
        super().__init__(config)
        self.backbone = _ColaBackbone(config, vae)
        self.post_init()
        # post_init 会打乱 AdaLN-Zero / 末层零初始化；按官方意图补回。
        self._restore_dit_zero_init()

    def _restore_dit_zero_init(self) -> None:
        for block in self.backbone.dit.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.backbone.dit.final.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.backbone.dit.final.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.backbone.dit.final.linear.weight)
        nn.init.zeros_(self.backbone.dit.final.linear.bias)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        from models.lm.cola.state_dict import remap_cola_mlp_keys

        return super().load_state_dict(remap_cola_mlp_keys(state_dict), strict=strict)


def _build_vae_for_config(
    config: FL_ColaConfig,
    *,
    variant: str | None = None,
    load_weights: bool = True,
) -> _ColaVAEBackbone:
    if load_weights:
        vae, path = load_vae_backbone(
            vae_model=config.vae_model,
            vae_size=config.vae_size,
            variant=variant,
            vae_run=config.vae_run,
        )
        print(f"[cola] loaded VAE from {path}")
        return vae
    from models.latent.cola_vae import FL_ColaVAEConfig, build_model_from_config as build_vae
    from models.model import config_from_yaml, resolve_model_config_path

    cfg_path = resolve_model_config_path(config.vae_model, config.vae_size)
    vae_cfg = config_from_yaml(FL_ColaVAEConfig, cfg_path)
    ensure_token_layout(vae_cfg)
    return build_vae(vae_cfg).backbone


def build_model_from_config(
    config: FL_ColaConfig,
    *,
    variant: str | None = None,
    load_vae_weights: bool = True,
) -> FL_ColaModel:
    ensure_token_layout(config)
    vae = _build_vae_for_config(
        config, variant=variant, load_weights=load_vae_weights,
    )
    return FL_ColaModel(config, vae)


def build_model(model_cfg: dict) -> FL_ColaModel:
    data, sampling = split_model_cfg(model_cfg)
    variant = data.pop("train_variant", None)
    load_vae = bool(data.pop("load_vae_weights", True))
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_ColaConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(
        config, variant=variant, load_vae_weights=load_vae,
    )
