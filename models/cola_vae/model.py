"""Causal Text VAE for Cola DLM Stage-1.

Learns a stable text↔latent map with reconstruction, KL, and BERT-style mask loss.
Encoder/decoder are strictly causal (paper Cola DLM).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cola_vae.config import FL_ColaVAEConfig
from models.cola_vae.layers import CausalBlock
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    sample_from_logits,
    split_model_cfg,
)
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg


class _ColaVAEBackbone(nn.Module):
    """Causal Text VAE backbone used for Stage-1 pretraining."""

    full_sequence_training = True

    def __init__(
        self,
        token_layout: FL_TokenLayout,
        max_seq_len: int = 1024,
        n_layer_enc: int = 4,
        n_layer_dec: int = 4,
        n_head: int = 6,
        n_embd: int = 384,
        latent_dim: int = 16,
        dropout: float = 0.1,
        beta_kl: float = 0.1,
        lambda_mask: float = 1.0,
        mask_ratio: float = 0.15,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.token_layout = token_layout
        self.vocab_size = token_layout.vocab_size
        self.max_seq_len = max_seq_len
        self.n_embd = n_embd
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.lambda_mask = lambda_mask
        self.mask_ratio = mask_ratio

        self.wte = nn.Embedding(self.vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [
                CausalBlock(n_embd, n_head, dropout, use_flash=use_flash)
                for _ in range(n_layer_enc)
            ]
        )
        self.enc_ln = nn.LayerNorm(n_embd)
        self.to_mu = nn.Linear(n_embd, latent_dim)
        self.to_logvar = nn.Linear(n_embd, latent_dim)

        self.from_latent = nn.Linear(latent_dim, n_embd)
        self.decoder = nn.ModuleList(
            [
                CausalBlock(n_embd, n_head, dropout, use_flash=use_flash)
                for _ in range(n_layer_dec)
            ]
        )
        self.dec_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.last_ce_loss = float("nan")
        self.last_kl_loss = float("nan")
        self.last_mask_loss = float("nan")

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(
        self,
        tokens: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(z0, mu, logvar)`` with shape ``(B, L, latent_dim)``."""
        x = self.drop(self.wte(tokens))
        for block in self.encoder:
            x = block(x)
        x = self.enc_ln(x)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x).clamp(-20.0, 20.0)
        if sample:
            std = torch.exp(0.5 * logvar)
            z0 = mu + torch.randn_like(std) * std
        else:
            z0 = mu
        return z0, mu, logvar

    def decode_logits(self, z0: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.from_latent(z0))
        for block in self.decoder:
            x = block(x)
        x = self.dec_ln(x)
        return self.lm_head(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets, kwargs
        tokens = idx
        # Clean reconstruction CE + KL; BERT-mask is a separate auxiliary term.
        z0, mu, logvar = self.encode(tokens, sample=self.training)
        logits = self.decode_logits(z0)

        ce = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens.reshape(-1),
            ignore_index=self.token_layout.ignore_index,
        )
        # KL(q(z|x) || N(0,I)) averaged over batch/seq/latent.
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()

        # BERT-style: encode a masked view; CE only on masked positions.
        mask_loss = torch.zeros((), device=tokens.device, dtype=ce.dtype)
        if self.training and self.lambda_mask > 0 and self.mask_ratio > 0:
            mask = torch.rand(tokens.shape, device=tokens.device) < self.mask_ratio
            mask[:, 0] = False
            if mask.any():
                enc_masked = tokens.clone()
                enc_masked[mask] = self.token_layout.pad_token_id
                z_m, _, _ = self.encode(enc_masked, sample=True)
                logits_m = self.decode_logits(z_m)
                mask_loss = F.cross_entropy(
                    logits_m[mask],
                    tokens[mask],
                    ignore_index=self.token_layout.ignore_index,
                )

        self.last_ce_loss = float(ce.detach().item())
        self.last_kl_loss = float(kl.detach().item())
        self.last_mask_loss = float(mask_loss.detach().item())
        # Eval: CE only so eval_ppl ≈ reconstruction perplexity.
        if not self.training:
            return logits, ce
        loss = ce + self.beta_kl * kl + self.lambda_mask * mask_loss
        return logits, loss

    @torch.no_grad()
    def reconstruct(
        self,
        tokens: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        z0, _, _ = self.encode(tokens, sample=False)
        logits = self.decode_logits(z0)
        return sample_from_logits(logits, temperature=temperature, top_k=top_k)

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
        """Stage-1 has no latent prior; reconstruct from BOS/prefix via encode→decode.

        For open-ended generation use the Stage-2 ``cola`` model.
        """
        del sampling_cfg
        seqlen = int(seqlen or self.max_seq_len)
        device = next(self.parameters()).device
        bos = self.token_layout.bos_token_id if bos_token_id is None else bos_token_id
        if prefix_tokens is not None:
            idx = prefix_tokens.to(device=device, dtype=torch.long)
            if idx.size(0) != num_samples:
                raise ValueError("prefix_tokens batch must match num_samples")
            # Pad/truncate to seqlen then reconstruct.
            if idx.size(1) < seqlen:
                pad = torch.full(
                    (num_samples, seqlen - idx.size(1)),
                    self.token_layout.pad_token_id,
                    device=device,
                    dtype=torch.long,
                )
                idx = torch.cat([idx, pad], dim=1)
            else:
                idx = idx[:, :seqlen]
        else:
            idx = torch.full(
                (num_samples, seqlen), bos, dtype=torch.long, device=device,
            )
        out = self.reconstruct(idx, temperature=temperature, top_k=top_k)
        return out, 1


class FL_ColaVAEModel(FL_PreTrainedModel):
    config_class = FL_ColaVAEConfig

    def __init__(self, config: FL_ColaVAEConfig) -> None:
        super().__init__(config)
        self.backbone = _ColaVAEBackbone(**config.backbone_kwargs())
        self.post_init()

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        from models.cola_vae.state_dict import remap_vae_mlp_keys

        return super().load_state_dict(remap_vae_mlp_keys(state_dict), strict=strict)


def build_model_from_config(config: FL_ColaVAEConfig) -> FL_ColaVAEModel:
    ensure_token_layout(config)
    return FL_ColaVAEModel(config)


def build_model(model_cfg: dict) -> FL_ColaVAEModel:
    data, sampling = split_model_cfg(model_cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_ColaVAEConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
