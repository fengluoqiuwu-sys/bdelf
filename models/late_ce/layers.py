"""Layer primitives for Embedded Language Flows (ELF)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier_uniform_(weight: torch.Tensor) -> None:
    nn.init.xavier_uniform_(weight)


def _zeros_(t: torch.Tensor) -> None:
    nn.init.zeros_(t)


def _normal_002_(t: torch.Tensor) -> None:
    nn.init.normal_(t, mean=0.0, std=0.02)


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    kernel_init=_xavier_uniform_,
    bias_init=_zeros_,
) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    kernel_init(layer.weight)
    if bias and bias_init is not None:
        bias_init(layer.bias)
    return layer


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class TextRotaryEmbedding(nn.Module):
    """1D RoPE; prefix (condition) tokens get identity rotation."""

    def __init__(
        self,
        dim: int,
        seq_len: int = 1024,
        *,
        theta: float = 10000.0,
        num_prefix_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.theta = theta
        self.num_prefix_tokens = num_prefix_tokens
        freqs_cos, freqs_sin = self._compute_freqs()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _compute_freqs(self) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32)[: self.dim // 2]
                / self.dim
            )
        )
        pos = torch.arange(self.seq_len, dtype=torch.float32)
        angles = torch.outer(pos, freqs)
        angles = angles.repeat_interleave(2, dim=-1)
        d = angles.shape[-1]
        parts_cos: list[torch.Tensor] = []
        parts_sin: list[torch.Tensor] = []
        if self.num_prefix_tokens > 0:
            parts_cos.append(
                torch.ones(self.num_prefix_tokens, d, dtype=torch.float32)
            )
            parts_sin.append(
                torch.zeros(self.num_prefix_tokens, d, dtype=torch.float32)
            )
        parts_cos.append(torch.cos(angles))
        parts_sin.append(torch.sin(angles))
        return torch.cat(parts_cos, dim=0), torch.cat(parts_sin, dim=0)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, H, L, D)
        seq = t.size(-2)
        cos = self.freqs_cos[:seq].to(dtype=t.dtype)
        sin = self.freqs_sin[:seq].to(dtype=t.dtype)
        return t * cos + rotate_half(t) * sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(dtype)
        return self.weight.to(dtype) * x


class BottleneckTextProj(nn.Module):
    """Encoder dim → bottleneck → model hidden size."""

    def __init__(
        self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int,
    ) -> None:
        super().__init__()
        self.proj1 = make_linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = make_linear(bottleneck_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = make_linear(
            frequency_embedding_size,
            hidden_size,
            kernel_init=_normal_002_,
        )
        self.mlp_2 = make_linear(
            hidden_size,
            hidden_size,
            kernel_init=_normal_002_,
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000,
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.mlp_0(self.timestep_embedding(t, self.frequency_embedding_size))
        return self.mlp_2(F.silu(t_emb))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.qkv = make_linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = make_linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[nn.Module] = None,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)
        attn_mask = None
        if attention_mask is not None:
            # (B, S) with 1=valid → SDPA bool mask broadcast over heads/queries
            attn_mask = attention_mask[:, None, None, :].bool()
        drop_p = 0.0 if deterministic else self.attn_drop
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_p,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        out = self.proj(out)
        if self.proj_drop > 0.0 and not deterministic:
            out = F.dropout(out, p=self.proj_drop, training=True)
        return out


class SwiGLUFFN(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, *, drop: float = 0.0, bias: bool = True,
    ) -> None:
        super().__init__()
        # Match ELF / LLaMA SwiGLU effective width: 2/3 of 4× expansion.
        hidden_eff = int(hidden_dim * 2 / 3)
        self.drop = drop
        self.w12 = make_linear(dim, 2 * hidden_eff, bias=bias)
        self.w3 = make_linear(hidden_eff, dim, bias=bias)

    def forward(self, x: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        if self.drop > 0.0 and not deterministic:
            hidden = F.dropout(hidden, p=self.drop, training=True)
        return self.w3(hidden)


class FinalLayer(nn.Module):
    """Zero-init flow-matching head back to encoder embedding dim."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = make_linear(
            hidden_size,
            out_channels,
            kernel_init=_zeros_,
            bias_init=_zeros_,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class ELFBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(
            hidden_size,
            num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(
            hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[nn.Module] = None,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope,
            attention_mask=attention_mask,
            deterministic=deterministic,
        )
        x = x + self.mlp(self.norm2(x), deterministic=deterministic)
        return x
