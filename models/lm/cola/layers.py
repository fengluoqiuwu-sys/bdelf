"""DiT 层：对齐官方 Cola-DLM ``modeling_cola_dit.py``（AdaLN-Zero + 块因果）。

与官方一致的性质（规模相关的 width/depth/head 除外）：
- 时间嵌入：sin 后 cos、``half_dim`` 分母、三层 SiLU MLP
- 块内 AdaLN-Zero + 无仿射 LayerNorm；末层仿射 LayerNorm
- 每头 QK LayerNorm；MLP 为 GELU(tanh)、expand=4
- RoPE 只作用前 ``rope_dim`` 通道（官方 96/128，等比例缩小）
- 训练 2L mask：``[clean | noisy]``，可见集 ``V_b = {sg(z_0^(<b)), z_t^(b)}``
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.latent.cola_vae.layers import fused_flex_attention
from models.rope import RotaryEmbedding


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """正弦时间嵌入。对齐 diffusers ``flip_sin_to_cos=False, downscale_freq_shift=0``。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    # 官方：sin 再 cos（不是 cos 再 sin）。
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """官方 ``TimestepEmbedding``：Linear → SiLU → Linear → SiLU → Linear。"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t`` 为 ``(B,)`` 或 ``(B, L)``，返回 ``(B, C)`` 或 ``(B, L, C)``。"""
        orig = t.shape
        emb = self.mlp(timestep_embedding(t.reshape(-1), self.frequency_embedding_size))
        if t.ndim == 2:
            return emb.view(orig[0], orig[1], -1)
        return emb


def build_block_causal_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """布尔 mask ``(L, L)``：True=保留。块内双向、块间因果。"""
    q = torch.arange(seq_len, device=device)[:, None]
    kv = torch.arange(seq_len, device=device)[None, :]
    bq = q // block_size
    bkv = kv // block_size
    return bkv <= bq


def build_cola_2l_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """官方 2L 训练 mask，布局 ``[clean(L) | noisy(L)]``。

    噪声 Q 块 ``b``：看干净 K 的 ``0..b-1``（不含当前干净块）+ 自身噪声块。
    干净 Q：只在干净半段上块因果，不看噪声半段。
    """
    two_l = 2 * seq_len
    idx = torch.arange(two_l, device=device)
    local = torch.where(idx < seq_len, idx, idx - seq_len)
    blk = local // block_size
    is_clean = idx < seq_len
    q_blk = blk[:, None]
    k_blk = blk[None, :]
    q_clean = is_clean[:, None]
    k_clean = is_clean[None, :]
    noisy_q_clean_k = (~q_clean) & k_clean & (q_blk > k_blk)
    noisy_q_noisy_k = (~q_clean) & (~k_clean) & (q_blk == k_blk)
    clean_q_clean_k = q_clean & k_clean & (q_blk >= k_blk)
    return noisy_q_clean_k | noisy_q_noisy_k | clean_q_clean_k


def cola_2l_mask_mod(b, h, q_idx, kv_idx, block_size: int, n: int):
    """FlexAttention 版 2L mask；布局 ``[clean(L) | noisy(L)]``，与 ``build_cola_2l_mask`` 一致。"""
    del b, h
    is_clean_q = q_idx < n
    is_clean_k = kv_idx < n
    local_q = torch.where(is_clean_q, q_idx, q_idx - n)
    local_k = torch.where(is_clean_k, kv_idx, kv_idx - n)
    q_b = local_q // block_size
    k_b = local_k // block_size
    noisy_q_clean_k = (~is_clean_q) & is_clean_k & (q_b > k_b)
    noisy_q_noisy_k = (~is_clean_q) & (~is_clean_k) & (q_b == k_b)
    clean_q_clean_k = is_clean_q & is_clean_k & (q_b >= k_b)
    return noisy_q_clean_k | noisy_q_noisy_k | clean_q_clean_k


def block_causal_mask_mod(b, h, q_idx, kv_idx, block_size: int):
    del b, h
    return (kv_idx // block_size) <= (q_idx // block_size)


class BlockCausalAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        *,
        rope_dim: int | None = None,
        qk_bias: bool = False,
        qk_norm: bool = True,
        norm_eps: float = 1e-5,
        attn_backend: str = "flex",
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        if attn_backend not in ("flex", "sdpa"):
            raise ValueError(f"unknown attn_backend: {attn_backend}")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.attn_backend = attn_backend
        # 官方 rope_dim=96 / head_dim=128 → 等比例 3/4。
        rd = self.head_dim if rope_dim is None else int(rope_dim)
        if rd <= 0 or rd > self.head_dim or rd % 2 != 0:
            raise ValueError(f"rope_dim={rd} 须为偶数且 0 < rope_dim <= head_dim={self.head_dim}")
        self.rope_dim = rd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=qk_bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=qk_bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=norm_eps, elementwise_affine=True)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=norm_eps, elementwise_affine=True)
        self.rope = RotaryEmbedding(self.rope_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        flex_block_mask=None,
        sdpa_attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        if self.rope_dim < self.head_dim:
            q_rot, q_pass = q[..., : self.rope_dim], q[..., self.rope_dim :]
            k_rot, k_pass = k[..., : self.rope_dim], k[..., self.rope_dim :]
            q_rot, k_rot = self.rope.apply_qk(q_rot, k_rot, positions)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        else:
            q, k = self.rope.apply_qk(q, k, positions)

        if self.attn_backend == "flex" and flex_block_mask is not None:
            y = fused_flex_attention(q, k, v, block_mask=flex_block_mask)
        elif sdpa_attn_mask is not None:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_attn_mask,
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
    """官方 DiT MLP：GELU tanh 近似、expand_ratio=4。"""

    def __init__(self, n_embd: int, dropout: float = 0.0, expand_ratio: int = 4) -> None:
        super().__init__()
        hidden = expand_ratio * n_embd
        self.c_fc = nn.Linear(n_embd, hidden)
        self.c_proj = nn.Linear(hidden, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class DiTBlock(nn.Module):
    """AdaLN-Zero 时间条件 Transformer 块（官方 ``ColaDiTBlock``）。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        *,
        rope_dim: int | None = None,
        qk_norm: bool = True,
        norm_eps: float = 1e-5,
        expand_ratio: int = 4,
        attn_backend: str = "flex",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=norm_eps)
        self.attn = BlockCausalAttention(
            n_embd, n_head, dropout=dropout,
            rope_dim=rope_dim, qk_norm=qk_norm, norm_eps=norm_eps,
            attn_backend=attn_backend,
        )
        self.norm2 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=norm_eps)
        self.mlp = _DiTMLP(n_embd, dropout=dropout, expand_ratio=expand_ratio)
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
        *,
        flex_block_mask=None,
        sdpa_attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod = self.adaLN_modulation(cond)
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        h = self.norm1(x) * (1.0 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(
            h, flex_block_mask=flex_block_mask, sdpa_attn_mask=sdpa_attn_mask, positions=positions,
        )
        h = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, n_embd: int, out_dim: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        # 官方末层 LayerNorm 带仿射。
        self.norm = nn.LayerNorm(n_embd, elementwise_affine=True, eps=norm_eps)
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
        mod = self.adaLN_modulation(cond)
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        shift, scale = mod.chunk(2, dim=-1)
        x = self.norm(x) * (1.0 + scale) + shift
        return self.linear(x)
