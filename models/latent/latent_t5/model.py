"""T5-small 维数 latent：encoder/decoder self-attn 默认同模式；可读出 none 做原 T5。"""

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
    posterior_regularizer,
)
from models.latent.latent_t5.config import FL_LatentT5Config
from models.latent.latent_t5.span import apply_span_sentinels, span_corruption_mask
from models.latent.latent_t5.t5_blocks import (
    T5Attention,
    T5DecoderBlock,
    T5DenseReluDense,
    T5LayerNorm,
    T5StyleEncoder,
)
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    sample_from_logits,
    split_model_cfg,
)
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg

ReadoutMode = Literal["e", "b", "none"]


class _LatentT5Backbone(nn.Module):
    full_sequence_training = True
    supports_prefix = True

    def __init__(
        self,
        token_layout: FL_TokenLayout,
        max_seq_len: int = 4096,
        readout: ReadoutMode = "e",
        n_layer_enc: int = 6,
        n_layer_dec: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        latent_dim: int = 32,
        dropout: float = 0.0,
        beta_kl: float = 0.1,
        kl_entropy: bool = False,
        lambda_span: float = 1.0,
        span_mask_ratio: float = 0.15,
        span_mean_len: int = 3,
        num_sentinels: int = 100,
        bidirectional: bool = True,
        decoder_bidirectional: bool | None = None,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.token_layout = token_layout
        self.vocab_size = token_layout.vocab_size
        self.max_seq_len = max_seq_len
        self.readout = readout
        self.n_embd = n_embd
        self.latent_dim = latent_dim
        # 入口始终逐 token 因果块长 1（无块因果）；BELF/RELF LatentBundle 须能读到。
        self.block_size = 1
        self.beta_kl = beta_kl
        self.kl_entropy = bool(kl_entropy)
        self.lambda_span = lambda_span
        self.span_mask_ratio = span_mask_ratio
        self.span_mean_len = span_mean_len
        self.num_sentinels = num_sentinels
        # null：decoder 与 encoder 同模式。readout=none（原 T5）写死双向。
        if readout == "none":
            if bidirectional is False or decoder_bidirectional is False:
                raise ValueError("readout=none（原 T5）只支持双向，禁止 unidirectional")
            bidirectional = True
            decoder_bidirectional = True
        self.decoder_bidirectional = (
            bidirectional if decoder_bidirectional is None else bool(decoder_bidirectional)
        )
        self.memory_dim = n_embd if readout in ("e", "none") else latent_dim
        # 双向 decoder 从 z 起并行重建，避免 teacher-force 看到未来 token。
        self.from_latent: nn.Linear | None = (
            nn.Linear(self.memory_dim, n_embd, bias=True)
            if self.decoder_bidirectional and self.memory_dim != n_embd
            else None
        )
        # readout=none：HF 原版 t5-small 算子；e/b 仍走 encdec RoPE 栈。
        self._t5_style = readout == "none"
        self._logits_scale = 1.0
        self.n_head = n_head
        self.d_kv = d_kv
        self.d_ff = d_ff

        if self._t5_style:
            self.encoder = T5StyleEncoder(
                self.vocab_size,
                num_sentinels,
                n_embd=n_embd,
                n_head=n_head,
                d_kv=d_kv,
                d_ff=d_ff,
                n_layer=n_layer_enc,
                dropout=dropout,
            )
            self.readout_head = None
            self.decoder = nn.ModuleList([
                T5DecoderBlock(
                    n_embd, n_head, d_kv, d_ff, dropout,
                    has_relative_attention_bias=(i == 0),
                )
                for i in range(n_layer_dec)
            ])
            self.dec_ln = T5LayerNorm(n_embd)
            self.dec_dropout = nn.Dropout(dropout)
            # 与 HF T5 默认 tie：lm_head 无 bias，权重与基础 vocab embed 共享。
            self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=False)
            self.lm_head.weight = self.encoder.wte.weight
            self._logits_scale = n_embd ** -0.5
            self._init_t5_weights()
        else:
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
            self.dec_dropout = nn.Identity()
            self.lm_head = nn.Linear(n_embd, self.vocab_size, bias=True)
            self.apply(self._init_weights)

        self.last_ce_loss = float("nan")
        self.last_kl_loss = float("nan")
        self.last_mask_loss = float("nan")
        self.last_token_acc = float("nan")
        self.last_mask_acc = float("nan")

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

    def _init_t5_weights(self) -> None:
        """HF T5 Mesh 初始化（factor=1）；tied lm_head 只经 embedding 初始化。"""
        factor = 1.0
        d_model = self.n_embd
        d_kv = self.d_kv
        n_heads = self.n_head
        d_ff = self.d_ff
        for name, module in self.named_modules():
            if isinstance(module, T5LayerNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, T5Attention):
                nn.init.normal_(
                    module.q.weight, mean=0.0,
                    std=factor * ((d_model * d_kv) ** -0.5),
                )
                nn.init.normal_(
                    module.k.weight, mean=0.0, std=factor * (d_model ** -0.5),
                )
                nn.init.normal_(
                    module.v.weight, mean=0.0, std=factor * (d_model ** -0.5),
                )
                nn.init.normal_(
                    module.o.weight, mean=0.0,
                    std=factor * ((n_heads * d_kv) ** -0.5),
                )
                if module.has_relative_attention_bias:
                    nn.init.normal_(
                        module.relative_attention_bias.weight,
                        mean=0.0, std=factor * (d_model ** -0.5),
                    )
            elif isinstance(module, T5DenseReluDense):
                nn.init.normal_(
                    module.wi.weight, mean=0.0, std=factor * (d_model ** -0.5),
                )
                nn.init.normal_(
                    module.wo.weight, mean=0.0, std=factor * (d_ff ** -0.5),
                )
            elif isinstance(module, nn.Embedding):
                if name.endswith("relative_attention_bias"):
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=factor * 1.0)

    def _pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens == self.token_layout.pad_token_id

    def _attn_pad_mask(self, tokens: torch.Tensor) -> torch.Tensor | None:
        """bool pad，供双向 self-attn 与 T5 cross-attn 屏蔽 pad key。

        全 False 与 ``None`` 对注意力分数等价，避免 ``.any()`` 同步。
        因果 self-attn 在层内仍忽略此 mask（右 pad + Flash）。
        """
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
        if self.readout_head is None:
            return h, h, torch.zeros((), device=h.device, dtype=h.dtype)
        return self.readout_head(h, sample=sample)

    def _decoder_inputs(self, tokens: torch.Tensor) -> torch.Tensor:
        bos = torch.full(
            (tokens.size(0), 1),
            self.token_layout.bos_token_id,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return torch.cat([bos, tokens[:, :-1]], dim=1)

    def decode_logits(
        self,
        dec_tokens: torch.Tensor | None,
        memory: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        memory_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode: Literal["bidirectional", "causal"] = (
            "bidirectional" if self.decoder_bidirectional else "causal"
        )
        if self.decoder_bidirectional:
            x = memory if self.from_latent is None else self.from_latent(memory)
            key_padding_mask = memory_pad_mask
        else:
            if dec_tokens is None:
                raise ValueError("因果 decoder 需要 dec_tokens")
            x = self.encoder.embed(dec_tokens)
        if self._t5_style:
            x = self.dec_dropout(x)
            position_bias: torch.Tensor | None = None
            enc_dec_bias: torch.Tensor | None = None
            for block in self.decoder:
                x, position_bias, enc_dec_bias = block(
                    x, memory,
                    key_padding_mask=key_padding_mask,
                    memory_pad_mask=memory_pad_mask,
                    position_bias=position_bias,
                    encoder_decoder_position_bias=enc_dec_bias,
                )
            x = self.dec_dropout(self.dec_ln(x))
            return self.lm_head(x * self._logits_scale)
        for block in self.decoder:
            x = block(
                x, memory,
                attn_mode=mode,
                key_padding_mask=key_padding_mask,
                memory_pad_mask=memory_pad_mask,
            )
        x = self.dec_ln(x)
        return self.lm_head(x)

    def span_aux_loss(
        self,
        tokens: torch.Tensor,
        dec_in: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pad = self._pad_mask(tokens)
        span_mask = span_corruption_mask(
            tokens.shape,
            mask_ratio=self.span_mask_ratio,
            mean_span_len=self.span_mean_len,
            device=tokens.device,
        )
        span_mask = span_mask & ~pad
        corrupted = apply_span_sentinels(
            tokens,
            span_mask,
            vocab_size=self.vocab_size,
            num_sentinels=self.num_sentinels,
        )
        z_c, _, _ = self.encode(corrupted, sample=True)
        mem_pad = self._attn_pad_mask(tokens)
        if self.decoder_bidirectional:
            logits_c = self.decode_logits(None, z_c, memory_pad_mask=mem_pad)
        else:
            if dec_in is None:
                raise ValueError("因果 span 辅助需要 dec_in")
            logits_c = self.decode_logits(dec_in, z_c, memory_pad_mask=mem_pad)
        ignore = self.token_layout.ignore_index
        targets = self._loss_targets(tokens).masked_fill(~span_mask, ignore)
        span_loss = self._ce(logits_c, targets)
        n_span = span_mask.to(dtype=span_loss.dtype).sum()
        span_loss = torch.where(
            n_span > 0, span_loss,
            torch.zeros((), device=tokens.device, dtype=span_loss.dtype),
        )
        with torch.no_grad():
            self.last_mask_acc = self._token_acc(
                logits_c, tokens, span_mask,
            ).detach()
        return span_loss

    def _encode_readout(
        self,
        tokens: torch.Tensor,
        *,
        sample: bool,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(tokens, key_padding_mask=key_padding_mask)
        if self.readout_head is None:
            zeros = torch.zeros((), device=h.device, dtype=h.dtype)
            return h, h, zeros
        return self.readout_head(h, sample=sample)

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
            self.training and self.lambda_span > 0 and self.span_mask_ratio > 0
        )
        span_mask: torch.Tensor | None = None
        enc_in = tokens
        if do_aux:
            span_mask = span_corruption_mask(
                tokens.shape,
                mask_ratio=self.span_mask_ratio,
                mean_span_len=self.span_mean_len,
                device=tokens.device,
            )
            span_mask = span_mask & ~pad
            corrupted = apply_span_sentinels(
                tokens,
                span_mask,
                vocab_size=self.vocab_size,
                num_sentinels=self.num_sentinels,
            )
            # 文档：encoder 看腐蚀序列、decoder 仍重建原 x；沿 batch 拼两次独立序列。
            enc_in = torch.cat([tokens, corrupted], dim=0)

        mem_pad = self._attn_pad_mask(enc_in)
        z_all, mu_all, logvar_all = self._encode_readout(
            enc_in, sample=self.training, key_padding_mask=mem_pad,
        )
        if self.decoder_bidirectional:
            dec_tokens = None
        else:
            dec_in = self._decoder_inputs(tokens)
            dec_tokens = dec_in if not do_aux else dec_in.repeat(2, 1)
        logits_all = self.decode_logits(
            dec_tokens, z_all, memory_pad_mask=mem_pad,
        )

        bsz = tokens.size(0)
        logits_c: torch.Tensor | None = None
        if do_aux:
            logits, logits_c = logits_all.split(bsz, dim=0)
            if self.readout_head is None:
                mu, logvar = mu_all, logvar_all
            else:
                mu, _ = mu_all.split(bsz, dim=0)
                logvar, _ = logvar_all.split(bsz, dim=0)
        else:
            logits, mu, logvar = logits_all, mu_all, logvar_all

        loss_targets = self._loss_targets(tokens)
        ce = self._ce(logits, loss_targets)
        if self.readout_head is None:
            kl = torch.zeros((), device=tokens.device, dtype=ce.dtype)
        else:
            kl = posterior_regularizer(
                mu, logvar, mask=~pad, kl_entropy=self.kl_entropy,
            )

        span_loss = torch.zeros((), device=tokens.device, dtype=ce.dtype)
        if logits_c is not None and span_mask is not None:
            ignore = self.token_layout.ignore_index
            raw_span_ce = self._ce(
                logits_c, loss_targets.masked_fill(~span_mask, ignore),
            )
            n_span = span_mask.to(dtype=ce.dtype).sum()
            span_loss = torch.where(
                n_span > 0, raw_span_ce,
                torch.zeros((), device=tokens.device, dtype=ce.dtype),
            )

        self.last_ce_loss = ce.detach()
        self.last_kl_loss = kl.detach()
        self.last_mask_loss = span_loss.detach()
        with torch.no_grad():
            valid = loss_targets != self.token_layout.ignore_index
            self.last_token_acc = self._token_acc(logits, tokens, valid).detach()
            if logits_c is not None and span_mask is not None:
                self.last_mask_acc = self._token_acc(
                    logits_c, tokens, span_mask,
                ).detach()
            else:
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
        out["kl_entropy"] = 1.0 if self.kl_entropy else 0.0
        out["lambda_mask"] = float(self.lambda_span)
        return out

    def online_eval_components(self) -> list:
        return []

    def _uncond_memory(
        self,
        batch: int,
        seq_len: int,
        device: torch.device,
        *,
        bos_token_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """无条件 memory：VAE 从先验采样；无瓶颈则 encode BOS+pad（原 T5 空源）。"""
        if self.readout_head is None:
            bos = (
                self.token_layout.bos_token_id
                if bos_token_id is None
                else bos_token_id
            )
            enc = torch.full(
                (batch, seq_len),
                self.token_layout.pad_token_id,
                device=device,
                dtype=torch.long,
            )
            enc[:, 0] = bos
            z, _, _ = self.encode(enc, sample=False)
            return z, self._attn_pad_mask(enc)
        return torch.randn(batch, seq_len, self.memory_dim, device=device), None

    def _sample_prior_memory(
        self,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        z, _ = self._uncond_memory(batch, seq_len, device)
        return z

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
            memory_pad = self._attn_pad_mask(enc_tokens)
            if self.decoder_bidirectional:
                logits = self.decode_logits(None, z, memory_pad_mask=memory_pad)
                rest = sample_from_logits(
                    logits[:, prefix_len:, :], temperature=temperature, top_k=top_k,
                )
                return torch.cat([prefix, rest], dim=1), 1
        else:
            z, memory_pad = self._uncond_memory(
                num_samples, seqlen, device, bos_token_id=bos,
            )
            if self.decoder_bidirectional:
                logits = self.decode_logits(None, z, memory_pad_mask=memory_pad)
                out = sample_from_logits(logits, temperature=temperature, top_k=top_k)
                return out, 1

        # 因果 AR：训练是 dec_in = BOS ‖ x[:-1] 预测 x。生成从单独 BOS 起步，
        # 采 seqlen 个 token 作为 x（有前缀则 dec = BOS ‖ prefix 后续写）。
        bos_col = torch.full(
            (num_samples, 1), bos, dtype=torch.long, device=device,
        )
        if prefix_tokens is not None:
            dec = torch.cat([bos_col, prefix], dim=1)
            pieces: list[torch.Tensor] = [prefix]
            steps = seqlen - prefix_len
        else:
            dec = bos_col
            pieces = []
            steps = seqlen
        nfe = 0
        for _ in range(steps):
            logits = self.decode_logits(dec, z, memory_pad_mask=memory_pad)
            next_tok = sample_from_logits(
                logits[:, -1, :], temperature=temperature, top_k=top_k,
            ).unsqueeze(-1)
            pieces.append(next_tok)
            dec = torch.cat([dec, next_tok], dim=1)
            nfe += 1
        return torch.cat(pieces, dim=1), nfe


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
