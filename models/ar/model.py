"""Autoregressive GPT-2 style decoder (AR).

Backbone: hand-written causal Transformer (``_GPTBackbone``).
Wrapper: ``FL_ARModel`` inherits HuggingFace ``PreTrainedModel`` for save/load
and standard LM interfaces; training still calls ``backbone.forward(idx, targets)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ar.config import FL_ARConfig
from models.model import FL_PreTrainedModel, ensure_token_layout, sample_from_logits, split_model_cfg
from models.rope import RotaryEmbedding
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg


@dataclass
class ARKVCache:
    """Per-layer cached K/V for autoregressive decode, shape ``(B, H, L, D)``."""

    k: torch.Tensor
    v: torch.Tensor


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        max_seq_len: int,
        dropout: float,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.use_flash = use_flash

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

        if not use_flash:
            mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
            self.register_buffer(
                "bias", mask.view(1, 1, max_seq_len, max_seq_len), persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        q, k = self.rope.apply_qk(q, k, positions)

        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            attn = attn.masked_fill(~self.bias[:, :, :seq_len, :seq_len], float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        return self.resid_dropout(self.c_proj(y))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cache: ARKVCache | None,
        pos_start: int,
    ) -> tuple[torch.Tensor, ARKVCache]:
        """Incremental decode: ``x`` is usually a single token; ``pos_start`` is its absolute position."""
        bsz, seq_len, _ = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        positions = torch.arange(
            pos_start, pos_start + seq_len, device=x.device, dtype=torch.long,
        )
        q, k = self.rope.apply_qk(q, k, positions)

        if cache is not None:
            k = torch.cat([cache.k, k], dim=2)
            v = torch.cat([cache.v, v], dim=2)

        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            attn = F.softmax(attn, dim=-1)
            y = attn @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        out = self.resid_dropout(self.c_proj(y))
        return out, ARKVCache(k=k, v=v)


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        hidden = 4 * n_embd
        self.c_fc = nn.Linear(n_embd, hidden)
        self.c_proj = nn.Linear(hidden, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        max_seq_len: int,
        dropout: float,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, dropout, use_flash)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cache: ARKVCache | None,
        pos_start: int,
    ) -> tuple[torch.Tensor, ARKVCache]:
        attn_out, cache = self.attn.forward_with_cache(self.ln_1(x), cache, pos_start)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, cache


class _GPTBackbone(nn.Module):
    """Hand-written GPT-2 style causal LM used for training."""

    full_sequence_training = False

    def __init__(
        self,
        token_layout,
        max_seq_len: int = 1024,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 672,
        dropout: float = 0.1,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.token_layout = token_layout
        self.max_seq_len = max_seq_len
        self.vocab_size = token_layout.vocab_size

        self.wte = nn.Embedding(token_layout.vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList(
            Block(n_embd, n_head, max_seq_len, dropout, use_flash) for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, token_layout.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len = idx.size()
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        x = self.drop(self.wte(idx))
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.token_layout.ignore_index,
            )
        return logits, loss

    def _forward_with_kv_cache(
        self,
        idx: torch.Tensor,
        kv_caches: list[ARKVCache | None] | None,
        pos_start: int,
    ) -> tuple[torch.Tensor, list[ARKVCache]]:
        x = self.drop(self.wte(idx))
        new_caches: list[ARKVCache] = []
        for layer_idx, block in enumerate(self.h):
            layer_cache = None if kv_caches is None else kv_caches[layer_idx]
            x, layer_cache = block.forward_with_cache(x, layer_cache, pos_start)
            new_caches.append(layer_cache)
        x = self.ln_f(x)
        return self.lm_head(x), new_caches

    @torch.no_grad()
    def _generate_with_kv_cache(
        self,
        num_samples: int,
        seqlen: int,
        *,
        temperature: float,
        top_k: int | None,
        bos: int,
    ) -> tuple[torch.Tensor, int]:
        device = next(self.parameters()).device
        idx = torch.full((num_samples, 1), bos, dtype=torch.long, device=device)
        kv_caches: list[ARKVCache | None] | None = None
        nfe = 0

        for pos in range(seqlen - 1):
            logits, kv_caches = self._forward_with_kv_cache(
                idx[:, -1:],
                kv_caches,
                pos_start=pos,
            )
            nfe += 1
            next_token = sample_from_logits(
                logits[:, -1, :], temperature=temperature, top_k=top_k,
            ).unsqueeze(-1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx, nfe

    @torch.no_grad()
    def _generate_legacy(
        self,
        num_samples: int,
        seqlen: int,
        *,
        temperature: float,
        top_k: int | None,
        bos: int,
    ) -> tuple[torch.Tensor, int]:
        device = next(self.parameters()).device
        idx = torch.full((num_samples, 1), bos, dtype=torch.long, device=device)
        nfe = 0
        for _ in range(seqlen - 1):
            logits, _ = self(idx)
            next_token = sample_from_logits(
                logits[:, -1, :], temperature=temperature, top_k=top_k,
            ).unsqueeze(-1)
            idx = torch.cat((idx, next_token), dim=1)
            nfe += 1
        return idx, nfe

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        seqlen: int | None = None,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        bos_token_id: int | None = None,
        sampling_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Temperature / top-k autoregressive sampling."""
        cfg = sampling_cfg or {}
        use_kv_cache = cfg.get("use_kv_cache", True)
        temperature = float(cfg.get("temperature", temperature))
        top_k = cfg.get("top_k", top_k)
        if top_k is not None:
            top_k = int(top_k)
        seqlen = seqlen or self.max_seq_len
        if seqlen > self.max_seq_len:
            raise ValueError(
                f"seqlen ({seqlen}) exceeds max_seq_len ({self.max_seq_len})"
            )
        bos = bos_token_id if bos_token_id is not None else self.token_layout.bos_token_id
        gen_kwargs = dict(
            temperature=temperature,
            top_k=top_k,
            bos=bos,
        )
        if use_kv_cache:
            return self._generate_with_kv_cache(num_samples, seqlen, **gen_kwargs)
        return self._generate_legacy(num_samples, seqlen, **gen_kwargs)


class FL_ARModel(FL_PreTrainedModel):
    config_class = FL_ARConfig

    def __init__(self, config: FL_ARConfig) -> None:
        super().__init__(config)
        self.backbone = _GPTBackbone(**config.backbone_kwargs())


def build_model_from_config(config: FL_ARConfig) -> FL_ARModel:
    ensure_token_layout(config)
    return FL_ARModel(config)


def build_model(cfg: dict) -> FL_ARModel:
    data, sampling = split_model_cfg(cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_ARConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
