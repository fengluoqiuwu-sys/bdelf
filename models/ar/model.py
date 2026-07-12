"""Autoregressive GPT-2 style decoder (AR).

Backbone: hand-written causal Transformer (``_GPTBackbone``).
Wrapper: ``FL_ARModel`` inherits HuggingFace ``PreTrainedModel`` for save/load
and standard LM interfaces; training still calls ``backbone.forward(idx, targets)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ar.config import FL_ARConfig
from models.model import FL_PreTrainedModel, ensure_token_layout, split_model_cfg
from models.rope import RotaryEmbedding
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg


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
            raise ValueError(f"n_embd ({n_embd}) 必须能被 n_head ({n_head}) 整除")
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
            raise ValueError(f"序列长度 {seq_len} 超过 max_seq_len {self.max_seq_len}")

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

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        """Greedy / top-k autoregressive sampling."""
        del sampling_cfg
        seqlen = seqlen or self.max_seq_len
        if seqlen > self.max_seq_len:
            raise ValueError(
                f"seqlen ({seqlen}) 超过 max_seq_len ({self.max_seq_len})"
            )
        device = next(self.parameters()).device
        bos = bos_token_id if bos_token_id is not None else self.token_layout.bos_token_id
        idx = torch.full((num_samples, 1), bos, dtype=torch.long, device=device)
        nfe = 0
        for _ in range(seqlen - 1):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                cutoff = values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            nfe += 1
        return idx, nfe


class FL_ARModel(FL_PreTrainedModel):
    config_class = FL_ARConfig

    def __init__(self, config: FL_ARConfig) -> None:
        super().__init__(config)
        self.backbone = _GPTBackbone(**config.backbone_kwargs())


def build_model_from_config(config: FL_ARConfig) -> FL_ARModel:
    ensure_token_layout(config)
    return FL_ARModel(config)


def build_model(cfg: dict) -> FL_ARModel:
    data, _ = split_model_cfg(cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_ARConfig(**data)
    apply_token_layout_to_config(config, layout)
    return build_model_from_config(config)
