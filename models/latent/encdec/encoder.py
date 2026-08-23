"""共享 latent encoder。"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.latent.encdec.layers import AttnMode, TransformerBlock
from models.tokens import FL_TokenLayout


class LatentEncoder(nn.Module):
    def __init__(
        self,
        token_layout: FL_TokenLayout,
        *,
        n_embd: int = 512,
        n_head: int = 8,
        d_kv: int = 64,
        d_ff: int = 2048,
        n_layer: int = 6,
        dropout: float = 0.0,
        use_flash: bool = True,
        attn_backend: str = "sdpa",
        bidirectional: bool = False,
        block_size: int = 1,
        extra_vocab: int = 0,
    ) -> None:
        super().__init__()
        self.token_layout = token_layout
        self.n_embd = n_embd
        self.bidirectional = bidirectional
        self.block_size = block_size
        self.attn_backend = attn_backend
        vocab = token_layout.vocab_size + extra_vocab
        self.wte = nn.Embedding(vocab, n_embd)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(
                n_embd, n_head, d_kv, d_ff, dropout,
                use_flash=use_flash,
                attn_backend=attn_backend,
                block_size=block_size,
            )
            for _ in range(n_layer)
        ])

    def attn_mode(self) -> AttnMode:
        if self.bidirectional:
            return "bidirectional"
        if self.block_size > 1:
            return "block_causal"
        return "causal"

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.drop(self.wte(tokens))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        mode = self.attn_mode()
        for layer in self.layers:
            x = layer(x, attn_mode=mode)
        return x
