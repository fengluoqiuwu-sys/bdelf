"""DiT-style layers with block-causal attention for Cola latent prior."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import RotaryEmbedding


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))


def build_block_causal_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask (L, L): True = keep. Block-internal bidirectional; cross-block causal."""
    q = torch.arange(seq_len, device=device)[:, None]
    kv = torch.arange(seq_len, device=device)[None, :]
    bq = q // block_size
    bkv = kv // block_size
    return bkv <= bq


class BlockCausalAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        q, k = self.rope.apply_qk(q, k, positions)

        if attn_mask is not None:
            # attn_mask: (L, L) bool keep → SDPA additive (B, 1, L, L) or broadcast
            additive = torch.zeros(
                seq_len, seq_len, device=x.device, dtype=q.dtype,
            )
            additive = additive.masked_fill(~attn_mask, torch.finfo(q.dtype).min)
            additive = additive.view(1, 1, seq_len, seq_len)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=additive,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        return self.resid_dropout(self.c_proj(y))


class _DiTMLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero time conditioning."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.attn = BlockCausalAttention(n_embd, n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        # Named c_fc/c_proj so Muon picks up DiT MLP weights.
        self.mlp = _DiTMLP(n_embd, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_embd, 6 * n_embd),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        h = self.norm1(x) * (1.0 + scale_msa[:, None]) + shift_msa[:, None]
        x = x + gate_msa[:, None] * self.attn(h, attn_mask, positions)
        h = self.norm2(x) * (1.0 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp[:, None] * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, n_embd: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(n_embd, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_embd, 2 * n_embd),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1.0 + scale[:, None]) + shift[:, None]
        return self.linear(x)
