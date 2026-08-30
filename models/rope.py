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

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim ({head_dim}) must be even")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.head_dim = head_dim
        self.max_seq_len = int(max_seq_len)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 顺序位置预计算，避免每层每次前向 outer+sincos。
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer(
            "cos_cached",
            torch.repeat_interleave(freqs.cos(), 2, dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            torch.repeat_interleave(freqs.sin(), 2, dim=-1),
            persistent=False,
        )

    def _cos_sin(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cos/sin from positions ``(L,)`` → ``(1,1,L,D)`` or ``(B,L)`` → ``(B,1,L,D)``."""
        pos = positions.float()
        inv = self.inv_freq.to(device=positions.device)
        if pos.ndim == 1:
            freqs = torch.outer(pos, inv)
            cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(dtype)
            sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(dtype)
            return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        if pos.ndim == 2:
            freqs = pos.unsqueeze(-1) * inv.view(1, 1, -1)
            cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(dtype)
            sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(dtype)
            return cos.unsqueeze(1), sin.unsqueeze(1)
        raise ValueError(
            f"RoPE positions 须为 (L,) 或 (B, L)，收到 {tuple(positions.shape)}"
        )

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._cos_sin(positions, q.dtype)
        return apply_rotary_pos_emb(q, k, cos, sin)

    def apply_sequential(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """顺序位置 ``0..L-1``：切片预计算表，超长则回退现场计算。"""
        seq_len = q.size(-2)
        cached = int(self.cos_cached.size(0))
        if seq_len > cached:
            positions = torch.arange(seq_len, device=q.device, dtype=torch.long)
            return self.apply_qk(q, k, positions)
        cos = self.cos_cached[:seq_len].to(dtype=q.dtype)
        sin = self.sin_cached[:seq_len].to(dtype=q.dtype)
        return apply_rotary_pos_emb(q, k, cos[None, None], sin[None, None])


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
