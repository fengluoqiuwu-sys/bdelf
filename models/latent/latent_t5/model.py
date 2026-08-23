"""T5-small 维数 latent AR：双向 encoder + cross-attn decoder + span 辅助损失。"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.latent.encdec.encoder import LatentEncoder
from models.latent.encdec.layers import DecoderBlock
from models.latent.encdec.readout import (
    PosteriorBReadout,
    PosteriorEReadout,
    kl_gaussian,
)
from models.latent.latent_t5.config import FL_LatentT5Config
from models.latent.latent_t5.span import apply_span_sentinels, span_corruption_mask
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    sample_from_logits,
    split_model_cfg,
)
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg

ReadoutMode = Literal["e", "b"]


class _LatentT5Backbone(nn.Module):
    full_sequence_training = True
    supports_prefix = True

    def __init__(
        self,
        token_layout: FL_TokenLayout,
        max_seq_len: int = 1024,
        readout: ReadoutMode = "e",
        n_layer_enc: int = 6,
        n_layer_dec: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        latent_dim: int = 64,
        dropout: float = 0.0,
        beta_kl: float = 0.1,
        lambda_span: float = 1.0,
        span_mask_ratio: float = 0.15,
        span_mean_len: int = 3,
        num_sentinels: int = 100,
        bidirectional: bool = True,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.token_layout = token_layout
        self.vocab_size = token_layout.vocab_size
        self.max_seq_len = max_seq_len
        self.readout = readout
        self.n_embd = n_embd
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.lambda_span = lambda_span
        self.span_mask_ratio = span_mask_ratio
        self.span_mean_len = span_mean_len
        self.num_sentinels = num_sentinels
        self.memory_dim = n_embd if readout == "e" else latent_dim

        self.encoder = LatentEncoder(
            token_layout,
            n_embd=n_embd,
            n_head=n_head,
            d_kv=d_kv,
            d_ff=d_ff,
            n_layer=n_layer_enc,
            dropout=dropout,
            use_flash=use_flash,
            attn_backend="sdpa",
            bidirectional=bidirectional,
            block_size=1,
            extra_vocab=num_sentinels,
        )
        if readout == "e":
            self.readout_head = PosteriorEReadout(n_embd, latent_dim)
        else:
            self.readout_head = PosteriorBReadout(n_embd, latent_dim)

        self.decoder = nn.ModuleList([
            DecoderBlock(
                n_embd, n_head, d_kv, d_ff, self.memory_dim, dropout,
                use_flash=use_flash,
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

    def _loss_targets(self, tokens: torch.Tensor) -> torch.Tensor:
        pad = self.token_layout.pad_token_id
        ignore = self.token_layout.ignore_index
        targets = tokens.clone()
        targets[tokens == pad] = ignore
        return targets

    def encode(
        self,
        tokens: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(tokens)
        return self.readout_head(h, sample=sample)

    def _decoder_inputs(self, tokens: torch.Tensor) -> torch.Tensor:
        bos = torch.full(
            (tokens.size(0), 1),
            self.token_layout.bos_token_id,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return torch.cat([bos, tokens[:, :-1]], dim=1)

    def decode_logits(self, dec_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.encoder.embed(dec_tokens)
        for block in self.decoder:
            x = block(x, memory)
        x = self.dec_ln(x)
        return self.lm_head(x)

    def span_aux_loss(self, tokens: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        span_mask = span_corruption_mask(
            tokens.shape,
            mask_ratio=self.span_mask_ratio,
            mean_span_len=self.span_mean_len,
            device=tokens.device,
        )
        corrupted = apply_span_sentinels(
            tokens,
            span_mask,
            vocab_size=self.vocab_size,
            num_sentinels=self.num_sentinels,
        )
        z_c, _, _ = self.encode(corrupted, sample=True)
        logits_c = self.decode_logits(dec_in, z_c)
        ignore = self.token_layout.ignore_index
        targets = self._loss_targets(tokens).masked_fill(~span_mask, ignore)
        span_loss = F.cross_entropy(
            logits_c.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=ignore,
        )
        with torch.no_grad():
            if span_mask.any():
                pred = logits_c.argmax(dim=-1)
                self.last_mask_acc = (
                    (pred[span_mask] == tokens[span_mask]).float().mean().detach()
                )
            else:
                self.last_mask_acc = torch.tensor(
                    float("nan"), device=tokens.device,
                )
        return span_loss

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets, kwargs
        tokens = idx
        z, mu, logvar = self.encode(tokens, sample=self.training)
        dec_in = self._decoder_inputs(tokens)
        logits = self.decode_logits(dec_in, z)

        loss_targets = self._loss_targets(tokens)
        ce = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            loss_targets.reshape(-1),
            ignore_index=self.token_layout.ignore_index,
        )
        kl = kl_gaussian(mu, logvar)

        span_loss = torch.zeros((), device=tokens.device, dtype=ce.dtype)
        if self.training and self.lambda_span > 0 and self.span_mask_ratio > 0:
            span_loss = self.span_aux_loss(tokens, dec_in)

        self.last_ce_loss = ce.detach()
        self.last_kl_loss = kl.detach()
        self.last_mask_loss = span_loss.detach()
        with torch.no_grad():
            ignore = self.token_layout.ignore_index
            valid = loss_targets != ignore
            if valid.any():
                pred = logits.argmax(dim=-1)
                self.last_token_acc = (
                    (pred[valid] == tokens[valid]).float().mean().detach()
                )
            else:
                self.last_token_acc = torch.tensor(
                    float("nan"), device=tokens.device,
                )
            if not (self.training and self.lambda_span > 0 and self.span_mask_ratio > 0):
                self.last_mask_acc = torch.tensor(
                    float("nan"), device=tokens.device,
                )
        if not self.training:
            return logits, ce
        loss = ce + self.beta_kl * kl + self.lambda_span * span_loss
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
        out["lambda_mask"] = float(self.lambda_span)
        return out

    def online_eval_components(self) -> list:
        return []

    def _sample_prior_memory(
        self,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.randn(batch, seq_len, self.memory_dim, device=device)

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
        temperature = float(cfg.get("temperature", temperature))
        top_k = cfg.get("top_k", top_k)
        if top_k is not None:
            top_k = int(top_k)
        seqlen = int(seqlen or self.max_seq_len)
        device = next(self.parameters()).device
        bos = self.token_layout.bos_token_id if bos_token_id is None else bos_token_id

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
            idx = prefix.clone()
            start = prefix_len
        else:
            z = self._sample_prior_memory(num_samples, seqlen, device)
            idx = torch.full((num_samples, 1), bos, dtype=torch.long, device=device)
            start = 1

        nfe = 0
        for _ in range(start, seqlen):
            logits = self.decode_logits(idx, z)
            next_tok = sample_from_logits(
                logits[:, -1, :], temperature=temperature, top_k=top_k,
            ).unsqueeze(-1)
            idx = torch.cat([idx, next_tok], dim=1)
            nfe += 1
        return idx, nfe


class FL_LatentT5Model(FL_PreTrainedModel):
    config_class = FL_LatentT5Config

    def __init__(self, config: FL_LatentT5Config) -> None:
        super().__init__(config)
        self.backbone = _LatentT5Backbone(**config.backbone_kwargs())
        self.post_init()


def build_model_from_config(config: FL_LatentT5Config) -> FL_LatentT5Model:
    ensure_token_layout(config)
    return FL_LatentT5Model(config)


def build_model(model_cfg: dict) -> FL_LatentT5Model:
    data, sampling = split_model_cfg(model_cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_LatentT5Config(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
