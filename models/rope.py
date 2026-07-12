"""Rotary Position Embedding (RoPE) shared by AR, BD3LM, and BDELF."""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to ``q``/``k`` with shape ``(B, H, L, D)``."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """GPT-NeoX style RoPE; supports arbitrary position indices per token."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim ({head_dim}) must be even")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.head_dim = head_dim
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(1, 1, L, head_dim)`` cos/sin tables from token positions ``(L,)``."""
        freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(dtype)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(dtype)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._cos_sin(positions, q.dtype)
        return apply_rotary_pos_emb(q, k, cos, sin)


def pair_positions(n: int, device: torch.device, start: int = 0) -> torch.Tensor:
    """Position ids for concatenated ``[xt, x0]`` each of length ``n``."""
    local = torch.arange(n, device=device, dtype=torch.long) + start
    return torch.cat((local, local))


def window_positions(
    window_start: int,
    window_len: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.arange(
        window_start, window_start + window_len, device=device, dtype=torch.long,
    )
