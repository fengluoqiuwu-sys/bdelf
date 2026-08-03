"""Cola DLM Stage-2: block-causal DiT latent prior + loaded Text VAE."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cola.config import FL_ColaConfig
from models.cola.infer import sample_block_ode
from models.cola.layers import DiTBlock, FinalLayer, TimestepEmbedder, build_block_causal_mask
from models.cola.vae_loader import load_vae_backbone, resolve_vae_checkpoint
from models.cola_vae.model import _ColaVAEBackbone
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    sample_from_logits,
    split_model_cfg,
)
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg


class _ColaDiT(nn.Module):
    """Block-causal DiT over continuous latents."""

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
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        self.diffusion_block_size = diffusion_block_size
        self.input_proj = nn.Linear(latent_dim, n_embd)
        self.t_embedder = TimestepEmbedder(n_embd)
        self.blocks = nn.ModuleList(
            [DiTBlock(n_embd, n_head, dropout=dropout) for _ in range(n_layer)]
        )
        self.final = FinalLayer(n_embd, latent_dim)
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        cached = self._mask_cache.get(key)
        if cached is None:
            cached = build_block_causal_mask(
                seq_len, self.diffusion_block_size, device,
            )
            if len(self._mask_cache) > 16:
                self._mask_cache.pop(next(iter(self._mask_cache)))
            self._mask_cache[key] = cached
        return cached

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field ``v(z_t, t)`` with shape matching ``z_t``."""
        bsz, seq_len, _ = z_t.shape
        x = self.input_proj(z_t)
        cond = self.t_embedder(t)
        mask = self._mask(seq_len, z_t.device)
        positions = torch.arange(seq_len, device=z_t.device, dtype=torch.long)
        for block in self.blocks:
            x = block(x, cond, mask, positions)
        return self.final(x, cond)


class _ColaBackbone(nn.Module):
    """Joint VAE + DiT prior for Cola Stage-2."""

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
        self.lambda_vae = config.lambda_vae
        self.lambda_fm = config.lambda_fm
        self.lambda_ref = config.lambda_ref
        self.freeze_vae = config.freeze_vae

        self.vae = vae
        # Keep VAE hyperparams aligned with Stage-2 yaml overrides.
        self.vae.beta_kl = config.beta_kl
        self.vae.lambda_mask = config.lambda_mask
        self.vae.mask_ratio = config.mask_ratio

        if config.latent_dim != vae.latent_dim:
            raise ValueError(
                f"cola latent_dim={config.latent_dim} != vae.latent_dim={vae.latent_dim}"
            )

        self.dit = _ColaDiT(
            latent_dim=config.latent_dim,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            max_seq_len=config.max_seq_len,
            diffusion_block_size=config.diffusion_block_size,
            dropout=config.dropout,
        )

        # Reference encoder for drift regularizer (stop-grad copy of VAE encoder).
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

    def _sample_train_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.time_schedule == "logit_normal":
            # schedule_loc shifts the logit-normal mean (paper loc≈1).
            mean = self.denoiser_p_mean + self.schedule_loc
            z = torch.randn(batch_size, device=device) * self.denoiser_p_std + mean
            t = torch.sigmoid(z)
        else:
            t = torch.rand(batch_size, device=device)
        return t.clamp(min=self.t_eps, max=1.0 - self.t_eps)

    def _fm_loss(self, z0: torch.Tensor) -> torch.Tensor:
        bsz = z0.size(0)
        t = self._sample_train_t(bsz, z0.device)
        t_exp = t[:, None, None]
        noise = torch.randn_like(z0)
        z_t = t_exp * z0 + (1.0 - t_exp) * noise
        v_tgt = (z0 - z_t) / torch.clamp(1.0 - t_exp, min=self.t_eps)
        db = self.diffusion_block_size
        seq_len = z0.size(1)
        n_blocks = max(1, seq_len // db)
        # Paper-style: one current block is noisy; past blocks are stop-grad clean.
        b_cur = int(torch.randint(0, n_blocks, (1,), device=z0.device).item())
        s, e = b_cur * db, (b_cur + 1) * db
        inp = z0.detach().clone()
        inp[:, s:e] = z_t[:, s:e]
        v_hat = self.dit(inp, t)
        # Only supervise the noisy current block (matches block-wise ODE inference).
        return (v_hat[:, s:e] - v_tgt[:, s:e]).pow(2).mean()

    def _ref_kl(self, tokens: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Same token view as the trainable encoder (clean reconstruction path).
            _, mu_ref, logvar_ref = self._ref_encoder.encode(tokens, sample=False)
        # KL(q_phi || q_ref) between diagonal Gaussians.
        var = logvar.exp()
        var_ref = logvar_ref.exp()
        kl = 0.5 * (
            (logvar_ref - logvar)
            + (var + (mu - mu_ref).pow(2)) / var_ref.clamp(min=1e-6)
            - 1.0
        )
        return kl.mean()

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets, kwargs
        tokens = idx
        # Joint VAE path: clean reconstruction + KL (+ optional BERT-mask).
        if self.freeze_vae:
            with torch.no_grad():
                z0, mu, logvar = self.vae.encode(tokens, sample=self.training)
                logits = self.vae.decode_logits(z0)
        else:
            z0, mu, logvar = self.vae.encode(tokens, sample=self.training)
            logits = self.vae.decode_logits(z0)

        ce = F.cross_entropy(
            logits.reshape(-1, self.vae.vocab_size),
            tokens.reshape(-1),
            ignore_index=self.token_layout.ignore_index,
        )
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
        mask_loss = torch.zeros((), device=tokens.device, dtype=ce.dtype)
        if (
            self.training
            and not self.freeze_vae
            and self.vae.lambda_mask > 0
            and self.vae.mask_ratio > 0
        ):
            mask = torch.rand(tokens.shape, device=tokens.device) < self.vae.mask_ratio
            mask[:, 0] = False
            if mask.any():
                enc_masked = tokens.clone()
                enc_masked[mask] = self.token_layout.pad_token_id
                z_m, _, _ = self.vae.encode(enc_masked, sample=True)
                logits_m = self.vae.decode_logits(z_m)
                mask_loss = F.cross_entropy(
                    logits_m[mask],
                    tokens[mask],
                    ignore_index=self.token_layout.ignore_index,
                )
        vae_loss = ce + self.vae.beta_kl * kl + self.vae.lambda_mask * mask_loss
        self.last_ce_loss = float(ce.detach().item())
        self.last_kl_loss = float(kl.detach().item())

        # Eval: report deterministic VAE reconstruction loss only (no random FM).
        if not self.training:
            self.last_l2_loss = float("nan")
            return logits, ce + self.vae.beta_kl * kl

        # Detach z0 for FM when VAE is frozen; otherwise keep joint graph.
        z0_fm = z0 if not self.freeze_vae else z0.detach()
        fm_loss = self._fm_loss(z0_fm)

        ref_loss = torch.zeros((), device=tokens.device, dtype=fm_loss.dtype)
        if self.lambda_ref > 0 and not self.freeze_vae:
            ref_loss = self._ref_kl(tokens, mu, logvar)

        loss = (
            self.lambda_vae * vae_loss
            + self.lambda_fm * fm_loss
            + self.lambda_ref * ref_loss
        )
        self.last_l2_loss = float(fm_loss.detach().item())
        return logits, loss

    def _dit_velocity_on_window(
        self, z_full: torch.Tensor, t_batch: torch.Tensor, block_len: int,
    ) -> torch.Tensor:
        v = self.dit(z_full, t_batch)
        return v[:, -block_len:]

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
        time_schedule = str(cfg.get("time_schedule", self.time_schedule))
        temperature = float(cfg.get("temperature", temperature))
        top_k = cfg.get("top_k", top_k)

        # Encode prefix clean condition from *fully covered* blocks only.
        # Partial trailing block is regenerated by ODE, then tokens[:prefix_len]
        # are restored so non-aligned prompts never leave pad-decoded garbage.
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
                clean = z_prefix
            else:
                clean = torch.zeros(
                    num_samples, 0, self.latent_dim, device=device, dtype=dtype,
                )
            start_block = n_full_blocks
        else:
            _ = bos_token_id  # unconditional: sample block-0 from noise
            clean = torch.zeros(
                num_samples, 0, self.latent_dim, device=device, dtype=dtype,
            )
            start_block = 0

        n_blocks = seqlen // db
        nfe = 0
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
                t_eps=self.t_eps,
                time_schedule=time_schedule,
                p_mean=self.denoiser_p_mean + self.schedule_loc,
                p_std=self.denoiser_p_std,
            )
            clean = torch.cat([clean, z_block], dim=1)

        z_all = clean[:, :seqlen]
        logits = self.vae.decode_logits(z_all)
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

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        from models.cola.state_dict import remap_cola_mlp_keys

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
    # Random init VAE (same architecture) when no checkpoint — for unit tests.
    from models.cola_vae import FL_ColaVAEConfig, build_model_from_config as build_vae
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
    # Optional train-time hints (not part of HF config fields).
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
