"""Cola DLM Stage-1 Text VAE。

对齐官方 ``ColaTextVAEModel``（规模除外）：块因果、post-norm SwiGLU、
QK-norm、rope_theta=500000、独立 lm_head、额外 mask token、patch Conv1d。
论文 Stage-1 损失：重建 + β KL(q||N(0,I)) + λ_mask BERT-mask。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.latent.cola_vae.config import FL_ColaVAEConfig
from models.latent.cola_vae.layers import TextVAEBlock
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    sample_from_logits,
    split_model_cfg,
)
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg


def _trunc_normal_(tensor: torch.Tensor, std: float, cutoff: float) -> None:
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-cutoff * std, b=cutoff * std)


class _ColaVAEBackbone(nn.Module):
    """官方风格 Text VAE；块因果 encoder/decoder。"""

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
        dropout: float = 0.0,
        beta_kl: float = 0.1,
        lambda_mask: float = 1.0,
        mask_ratio: float = 0.15,
        use_flash: bool = True,
        attn_backend: str = "flex",
        block_size: int = 16,
        rope_theta: float = 500000.0,
        qk_norm: bool = True,
        post_norm: bool = True,
        patch_size: int = 1,
        ffn_mult: int = 4,
        scaling_factor: float = 1.0,
        shifting_factor: float = 0.0,
        init_std: float = 0.02,
        init_cutoff_factor: float = 3.0,
    ) -> None:
        super().__init__()
        if max_seq_len % (patch_size * block_size) != 0:
            raise ValueError(
                f"max_seq_len={max_seq_len} 须能被 patch_size*block_size="
                f"{patch_size * block_size} 整除"
            )
        self.token_layout = token_layout
        self.vocab_size = token_layout.vocab_size
        self.mask_token_id = self.vocab_size
        self.max_seq_len = max_seq_len
        self.n_embd = n_embd
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.lambda_mask = lambda_mask
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.patch_size = patch_size
        self.scaling_factor = scaling_factor
        self.shifting_factor = shifting_factor

        ffn_dim = int(ffn_mult) * n_embd
        block_kwargs = dict(
            n_embd=n_embd,
            n_head=n_head,
            dropout=dropout,
            ffn_dim=ffn_dim,
            block_size=block_size,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
            post_norm=post_norm,
            use_flash=use_flash,
            attn_backend=attn_backend,
        )

        # 官方 wte 多 1 行，专供 BERT mask，不进 lm_head。
        self.wte = nn.Embedding(self.vocab_size + 1, n_embd)
        self.patch_embedder = nn.Conv1d(
            n_embd, n_embd, kernel_size=patch_size, stride=patch_size,
        )
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [TextVAEBlock(**block_kwargs) for _ in range(n_layer_enc)]
        )
        self.to_posterior = nn.Linear(n_embd, latent_dim * 2, bias=True)
        self.enc_latent_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)

        self.from_latent = nn.Linear(latent_dim, n_embd, bias=True)
        self.decoder = nn.ModuleList(
            [TextVAEBlock(**block_kwargs) for _ in range(n_layer_dec)]
        )
        self.unpatch_layer = nn.Linear(n_embd, patch_size * n_embd, bias=True)
        self.dec_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=True)

        self.last_ce_loss = float("nan")
        self.last_kl_loss = float("nan")
        self.last_mask_loss = float("nan")
        self.last_token_acc = float("nan")
        self.last_mask_acc = float("nan")

        self.apply(lambda m: self._init_weights(m, init_std, init_cutoff_factor))

    @staticmethod
    def _init_weights(module: nn.Module, std: float, cutoff: float) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            _trunc_normal_(module.weight, std, cutoff)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            _trunc_normal_(module.weight, std, cutoff)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.wte(tokens)
        x = x.transpose(1, 2)
        x = self.patch_embedder(x)
        x = x.transpose(1, 2)
        return self.drop(x)

    def encode(
        self,
        tokens: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 ``(z0, mu, logvar)``，形状 ``(B, L/patch, latent_dim)``。"""
        x = self._embed_tokens(tokens)
        for block in self.encoder:
            x = block(x)
        stats = self.to_posterior(x)
        mu, logvar = stats.chunk(2, dim=-1)
        mu = self.enc_latent_norm(mu)
        logvar = logvar.clamp(-30.0, 20.0)
        if sample:
            std = torch.exp(0.5 * logvar)
            z0 = mu + torch.randn_like(std) * std
        else:
            z0 = mu
        return z0, mu, logvar

    def decode_logits(
        self,
        z0: torch.Tensor,
        *,
        last_n: int | None = None,
        **_kwargs: object,
    ) -> torch.Tensor:
        x = self.drop(self.from_latent(z0))
        for block in self.decoder:
            x = block(x)
        x = self.unpatch_layer(x)
        if self.patch_size != 1:
            bsz, n_lat, _ = x.shape
            x = x.view(bsz, n_lat, self.patch_size, self.n_embd).reshape(
                bsz, n_lat * self.patch_size, self.n_embd,
            )
        x = self.dec_ln(x)
        if last_n is not None:
            n = int(last_n)
            if n < 0:
                raise ValueError(f"last_n 须非负，收到 {n}")
            if n < int(x.size(1)):
                x = x[:, -n:]
        return self.lm_head(x)

    def bert_mask_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """BERT-style：编码被 mask 的视图，只在 mask 位置算 CE。

        固定形状（不用 ``logits[mask]`` / ``mask.any()``），以便 torch.compile。
        """
        mask = torch.rand(tokens.shape, device=tokens.device) < self.mask_ratio
        mask[:, 0] = False
        enc_masked = torch.where(
            mask, torch.full_like(tokens, self.mask_token_id), tokens,
        )
        z_m, _, _ = self.encode(enc_masked, sample=True)
        logits_m = self.decode_logits(z_m)
        ignore = self.token_layout.ignore_index
        targets = tokens.masked_fill(~mask, ignore)
        logits_flat = logits_m.reshape(-1, self.vocab_size)
        mask_loss = F.cross_entropy(
            logits_flat,
            targets.reshape(-1),
            ignore_index=ignore,
        )
        with torch.no_grad():
            if mask.any():
                pred_m = logits_m.argmax(dim=-1)
                self.last_mask_acc = (
                    (pred_m[mask] == tokens[mask]).float().mean().detach()
                )
            else:
                self.last_mask_acc = torch.tensor(
                    float("nan"), device=tokens.device,
                )
        return mask_loss

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets, kwargs
        tokens = idx
        z0, mu, logvar = self.encode(tokens, sample=self.training)
        logits = self.decode_logits(z0)

        ce = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens.reshape(-1),
            ignore_index=self.token_layout.ignore_index,
        )
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()

        mask_loss = torch.zeros((), device=tokens.device, dtype=ce.dtype)
        if self.training and self.lambda_mask > 0 and self.mask_ratio > 0:
            mask_loss = self.bert_mask_loss(tokens)

        self.last_ce_loss = ce.detach()
        self.last_kl_loss = kl.detach()
        self.last_mask_loss = mask_loss.detach()
        with torch.no_grad():
            ignore = self.token_layout.ignore_index
            valid = tokens != ignore
            if valid.any():
                pred = logits.argmax(dim=-1)
                self.last_token_acc = (
                    (pred[valid] == tokens[valid]).float().mean().detach()
                )
            else:
                self.last_token_acc = torch.tensor(
                    float("nan"), device=tokens.device,
                )
            if not (self.training and self.lambda_mask > 0 and self.mask_ratio > 0):
                self.last_mask_acc = torch.tensor(
                    float("nan"), device=tokens.device,
                )
        if not self.training:
            return logits, ce
        loss = ce + self.beta_kl * kl + self.lambda_mask * mask_loss
        return logits, loss

    def train_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for attr, key in (
            ("last_ce_loss", "recon_ce"),
            ("last_kl_loss", "kl"),
            ("last_mask_loss", "mask"),
            ("last_token_acc", "token_acc"),
            ("last_mask_acc", "mask_acc"),
        ):
            val = getattr(self, attr, None)
            if val is None:
                continue
            if hasattr(val, "item"):
                val = val.item()
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if fval == fval:
                out[key] = fval
        out["beta_kl"] = float(self.beta_kl)
        out["lambda_mask"] = float(self.lambda_mask)
        return out

    def online_eval_components(self) -> list:
        return []

    @torch.compiler.disable
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
        del sampling_cfg
        seqlen = int(seqlen or self.max_seq_len)
        device = next(self.parameters()).device
        bos = self.token_layout.bos_token_id if bos_token_id is None else bos_token_id
        if prefix_tokens is not None:
            idx = prefix_tokens.to(device=device, dtype=torch.long)
            if idx.size(0) != num_samples:
                raise ValueError("prefix_tokens batch must match num_samples")
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
        from models.latent.cola_vae.state_dict import remap_vae_mlp_keys

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
