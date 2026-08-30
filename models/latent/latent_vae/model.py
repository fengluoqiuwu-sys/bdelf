"""T5-small 维数 Text VAE：因果 / 可选块因果 encoder + 并行 decoder + Cola 损失。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.latent.encdec.encoder import LatentEncoder
from models.latent.encdec.layers import TransformerBlock
from models.latent.encdec.readout import PosteriorBReadout, posterior_regularizer
from models.latent.latent_vae.config import FL_LatentVAEConfig
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    sample_from_logits,
    split_model_cfg,
)
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg


class _LatentVAEBackbone(nn.Module):
    full_sequence_training = True
    supports_prefix = True

    def __init__(
        self,
        token_layout: FL_TokenLayout,
        max_seq_len: int = 4096,
        n_layer_enc: int = 6,
        n_layer_dec: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        latent_dim: int = 64,
        dropout: float = 0.0,
        beta_kl: float = 0.1,
        kl_entropy: bool = False,
        lambda_mask: float = 1.0,
        mask_ratio: float = 0.15,
        use_flash: bool = True,
        attn_backend: str = "sdpa",
        block_size: int = 1,
    ) -> None:
        super().__init__()
        if block_size > 1 and max_seq_len % block_size != 0:
            raise ValueError(
                f"max_seq_len={max_seq_len} 须能被 block_size={block_size} 整除"
            )
        self.token_layout = token_layout
        self.vocab_size = token_layout.vocab_size
        self.mask_token_id = self.vocab_size
        self.max_seq_len = max_seq_len
        self.n_embd = n_embd
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.kl_entropy = bool(kl_entropy)
        self.lambda_mask = lambda_mask
        self.mask_ratio = mask_ratio
        self.block_size = block_size

        self.encoder = LatentEncoder(
            token_layout,
            n_embd=n_embd,
            n_head=n_head,
            d_kv=d_kv,
            d_ff=d_ff,
            n_layer=n_layer_enc,
            dropout=dropout,
            use_flash=use_flash,
            attn_backend=attn_backend,
            bidirectional=False,
            block_size=block_size,
            extra_vocab=1,
        )
        self.readout = PosteriorBReadout(n_embd, latent_dim)
        self.from_latent = nn.Linear(latent_dim, n_embd, bias=True)
        self.decoder = nn.ModuleList([
            TransformerBlock(
                n_embd, n_head, d_kv, d_ff, dropout,
                use_flash=use_flash,
                attn_backend=attn_backend,
                block_size=block_size,
            )
            for _ in range(n_layer_dec)
        ])
        self.dec_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=True)

        self.last_ce_loss = float("nan")
        self.last_kl_loss = float("nan")
        self.last_mask_loss = float("nan")
        self.last_token_acc = float("nan")
        self.last_mask_acc = float("nan")

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
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens == self.token_layout.pad_token_id

    def _attn_pad_mask(self, tokens: torch.Tensor) -> torch.Tensor | None:
        """因果不加 pad mask（以便 Flash）；块因果才传 bool pad。避免 ``.any()`` 同步。"""
        if self.encoder.attn_mode() == "causal":
            return None
        return self._pad_mask(tokens)

    def _loss_targets(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.masked_fill(
            tokens == self.token_layout.pad_token_id,
            self.token_layout.ignore_index,
        )

    def _ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.token_layout.ignore_index,
        )

    def _token_acc(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        pred = logits.argmax(dim=-1)
        n = valid.float().sum()
        acc = ((pred == tokens) & valid).float().sum() / n.clamp_min(1.0)
        return torch.where(
            n > 0, acc, torch.full((), float("nan"), device=tokens.device),
        )

    def encode(
        self,
        tokens: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(tokens, key_padding_mask=self._attn_pad_mask(tokens))
        return self.readout(h, sample=sample)

    def decode_logits(
        self,
        z: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        last_n: int | None = None,
    ) -> torch.Tensor:
        x = self.from_latent(z)
        mode = self.encoder.attn_mode()
        for block in self.decoder:
            x = block(x, attn_mode=mode, key_padding_mask=key_padding_mask)
        x = self.dec_ln(x)
        if last_n is not None:
            n = int(last_n)
            if n < 0:
                raise ValueError(f"last_n 须非负，收到 {n}")
            if n < int(x.size(1)):
                x = x[:, -n:]
        return self.lm_head(x)

    def bert_mask_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        pad = self._pad_mask(tokens)
        mask = torch.rand(tokens.shape, device=tokens.device) < self.mask_ratio
        mask[:, 0] = False
        mask = mask & ~pad
        enc_masked = torch.where(
            mask, torch.full_like(tokens, self.mask_token_id), tokens,
        )
        z_m, _, _ = self.encode(enc_masked, sample=True)
        logits_m = self.decode_logits(
            z_m, key_padding_mask=self._attn_pad_mask(tokens),
        )
        ignore = self.token_layout.ignore_index
        targets = self._loss_targets(tokens).masked_fill(~mask, ignore)
        mask_loss = self._ce(logits_m, targets)
        n_mask = mask.to(dtype=mask_loss.dtype).sum()
        mask_loss = torch.where(
            n_mask > 0, mask_loss,
            torch.zeros((), device=tokens.device, dtype=mask_loss.dtype),
        )
        with torch.no_grad():
            self.last_mask_acc = self._token_acc(logits_m, tokens, mask).detach()
        return mask_loss

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets, kwargs
        tokens = idx
        pad = self._pad_mask(tokens)
        do_aux = bool(
            self.training and self.lambda_mask > 0 and self.mask_ratio > 0
        )
        mask: torch.Tensor | None = None
        enc_in = tokens
        if do_aux:
            # 文档：encoder 看 mask 序列；沿 batch 拼两次独立序列（self-attn 不跨样本）。
            # 损失仍只对清洁半段算 KL、只对 mask 位算 CE。
            mask = torch.rand(tokens.shape, device=tokens.device) < self.mask_ratio
            mask[:, 0] = False
            mask = mask & ~pad
            enc_masked = torch.where(
                mask, torch.full_like(tokens, self.mask_token_id), tokens,
            )
            enc_in = torch.cat([tokens, enc_masked], dim=0)

        pad_enc = self._attn_pad_mask(enc_in)
        h = self.encoder(enc_in, key_padding_mask=pad_enc)
        z_all, mu_all, logvar_all = self.readout(h, sample=self.training)
        logits_all = self.decode_logits(z_all, key_padding_mask=pad_enc)

        bsz = tokens.size(0)
        logits_m: torch.Tensor | None = None
        if do_aux:
            logits, logits_m = logits_all.split(bsz, dim=0)
            mu, _ = mu_all.split(bsz, dim=0)
            logvar, _ = logvar_all.split(bsz, dim=0)
        else:
            logits, mu, logvar = logits_all, mu_all, logvar_all

        loss_targets = self._loss_targets(tokens)
        ce = self._ce(logits, loss_targets)
        kl = posterior_regularizer(
            mu, logvar, mask=~pad, kl_entropy=self.kl_entropy,
        )

        mask_loss = torch.zeros((), device=tokens.device, dtype=ce.dtype)
        if logits_m is not None and mask is not None:
            ignore = self.token_layout.ignore_index
            raw_mask_ce = self._ce(
                logits_m, loss_targets.masked_fill(~mask, ignore),
            )
            # 空 mask 时 CE 为 NaN；文档要求仅在 mask 位计 CE，无位则为 0。
            n_mask = mask.to(dtype=ce.dtype).sum()
            mask_loss = torch.where(
                n_mask > 0, raw_mask_ce, torch.zeros((), device=tokens.device, dtype=ce.dtype),
            )

        self.last_ce_loss = ce.detach()
        self.last_kl_loss = kl.detach()
        self.last_mask_loss = mask_loss.detach()
        with torch.no_grad():
            valid = loss_targets != self.token_layout.ignore_index
            self.last_token_acc = self._token_acc(logits, tokens, valid).detach()
            if logits_m is not None and mask is not None:
                self.last_mask_acc = self._token_acc(
                    logits_m, tokens, mask,
                ).detach()
            else:
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
        out["kl_entropy"] = 1.0 if self.kl_entropy else 0.0
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
        z, _, _ = self.encode(tokens, sample=False)
        logits = self.decode_logits(
            z, key_padding_mask=self._attn_pad_mask(tokens),
        )
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
        del bos_token_id
        cfg = sampling_cfg or {}
        temperature = float(cfg.get("temperature", temperature))
        top_k = cfg.get("top_k", top_k)
        if top_k is not None:
            top_k = int(top_k)
        seqlen = int(seqlen or self.max_seq_len)
        device = next(self.parameters()).device
        if prefix_tokens is not None:
            prefix = prefix_tokens.to(device=device, dtype=torch.long)
            if prefix.size(0) != num_samples:
                raise ValueError("prefix_tokens batch must match num_samples")
            prefix_len = prefix.size(1)
            if prefix_len >= seqlen:
                return prefix[:, :seqlen], 0
            pad = torch.full(
                (num_samples, seqlen - prefix_len),
                self.token_layout.pad_token_id,
                device=device,
                dtype=torch.long,
            )
            enc_tokens = torch.cat([prefix, pad], dim=1)
            z, _, _ = self.encode(enc_tokens, sample=False)
            logits = self.decode_logits(
                z, key_padding_mask=self._attn_pad_mask(enc_tokens),
            )
            rest = sample_from_logits(
                logits[:, prefix_len:, :], temperature=temperature, top_k=top_k,
            )
            return torch.cat([prefix, rest], dim=1), 1
        z = torch.randn(num_samples, seqlen, self.latent_dim, device=device)
        logits = self.decode_logits(z)
        out = sample_from_logits(logits, temperature=temperature, top_k=top_k)
        return out, 1


class FL_LatentVAEModel(FL_PreTrainedModel):
    config_class = FL_LatentVAEConfig

    def __init__(self, config: FL_LatentVAEConfig) -> None:
        super().__init__(config)
        self.backbone = _LatentVAEBackbone(**config.backbone_kwargs())
        self.post_init()


def build_model_from_config(config: FL_LatentVAEConfig) -> FL_LatentVAEModel:
    ensure_token_layout(config)
    return FL_LatentVAEModel(config)


def build_model(model_cfg: dict) -> FL_LatentVAEModel:
    data, sampling = split_model_cfg(model_cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_LatentVAEConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
