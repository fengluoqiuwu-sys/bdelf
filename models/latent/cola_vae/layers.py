"""Cola Text VAE 层。

``CausalBlock`` 保持 GPT 式 pre-norm，供 BDELF 复用，**不要改其语义**。
Cola VAE 本体用 ``TextVAEBlock``：对齐官方 ``modeling_cola_vae.py``
（post-norm、SwiGLU、QK-norm、rope_theta=500000、块因果）。
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import RotaryEmbedding

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    FLEX_ATTN_AVAILABLE = True
except ImportError:
    create_block_mask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]
    FLEX_ATTN_AVAILABLE = False

_flex_attention_compiled = None


def fused_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask=None,
) -> torch.Tensor:
    """外层 ``torch.compile(model)`` 时由 Dynamo 融合；eager 则单独 compile Flex。"""
    if torch.compiler.is_dynamo_compiling():
        return flex_attention(q, k, v, block_mask=block_mask)
    global _flex_attention_compiled
    if _flex_attention_compiled is None:
        _flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_attention_compiled(q, k, v, block_mask=block_mask)


def bool_mask_to_sdpa_additive(bool_mask: torch.Tensor) -> torch.Tensor:
    """``(L, L)`` bool → ``(1, 1, L, L)`` 加性 mask，供 SDPA 回退。"""
    additive = torch.zeros(
        1, 1, bool_mask.size(0), bool_mask.size(1),
        dtype=torch.float32, device=bool_mask.device,
    )
    additive.masked_fill_(
        ~bool_mask.view(1, 1, bool_mask.size(0), bool_mask.size(1)),
        float("-inf"),
    )
    return additive


def block_causal_mask_mod(b, h, q_idx, kv_idx, block_size: int):
    """块内双向、块间因果（VAE / DiT 推理）。"""
    del b, h
    return (kv_idx // block_size) <= (q_idx // block_size)


def build_block_causal_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """布尔 mask ``(L, L)``：True=保留。块内双向、块间因果。"""
    q = torch.arange(seq_len, device=device)[:, None]
    kv = torch.arange(seq_len, device=device)[None, :]
    return (kv // block_size) <= (q // block_size)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
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
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            )
            attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        # Names match train/muon.py Muon eligibility (c_fc / c_proj).
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class CausalBlock(nn.Module):
    """GPT 式 pre-norm 因果块；BDELF decoder 依赖此接口，禁止改语义。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, use_flash=use_flash)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _VAERotary(nn.Module):
    """官方 VAE RoPE：``theta=500000``，freq 复制拼接 + interleaved rotate。"""

    def __init__(self, head_dim: int, rope_theta: float = 500000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim ({head_dim}) must be even")
        self.head_dim = head_dim
        self.rope_theta = float(rope_theta)
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], 2, x.shape[-1] // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=q.dtype)[None, None, :, :]
        q = (q * cos) + (self._rotate_interleaved(q) * sin)
        k = (k * cos) + (self._rotate_interleaved(k) * sin)
        return q, k


class SwiGLU(nn.Module):
    """官方：先半段为值、后半段为 gate。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * value


class _VAESwiGLU(nn.Module):
    def __init__(self, n_embd: int, ffn_dim: int, dropout: float, bias: bool = True) -> None:
        super().__init__()
        self.w12 = nn.Linear(n_embd, ffn_dim, bias=bias)
        self.act = SwiGLU()
        self.w3 = nn.Linear(ffn_dim // 2, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(self.act(self.w12(x))))


class TextVAEBlock(nn.Module):
    """官方 ``TextVAEBlock``：默认 post-norm + 块因果 + QK-norm + SwiGLU。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        *,
        ffn_dim: int,
        block_size: int,
        rope_theta: float = 500000.0,
        qk_norm: bool = True,
        post_norm: bool = True,
        qk_bias: bool = False,
        bias: bool = True,
        layer_norm_eps: float = 1e-6,
        use_flash: bool = True,
        attn_backend: str = "flex",
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        if ffn_dim % 2 != 0:
            raise ValueError(f"ffn_dim ({ffn_dim}) must be even for SwiGLU")
        if attn_backend == "flex" and not FLEX_ATTN_AVAILABLE:
            raise RuntimeError(
                "attn_backend=flex 需要 PyTorch FlexAttention；请升级或改用 sdpa"
            )
        if attn_backend not in ("flex", "sdpa"):
            raise ValueError(f"unknown attn_backend: {attn_backend}")
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.block_size = block_size
        self.post_norm = post_norm
        self.use_flash = use_flash
        self.attn_backend = attn_backend
        self.norm_attn = nn.LayerNorm(n_embd, eps=layer_norm_eps)
        self.norm_ffn = nn.LayerNorm(n_embd, eps=layer_norm_eps)
        # ``attn.c_attn`` / ``attn.c_proj`` 供 Muon 识别。
        self.attn = nn.Module()
        self.attn.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=qk_bias)
        self.attn.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.q_norm: nn.LayerNorm | None
        self.k_norm: nn.LayerNorm | None
        if qk_norm:
            self.q_norm = nn.LayerNorm(n_embd, eps=layer_norm_eps)
            self.k_norm = nn.LayerNorm(n_embd, eps=layer_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.rope = _VAERotary(self.head_dim, rope_theta=rope_theta)
        self.mlp = _VAESwiGLU(n_embd, ffn_dim, dropout, bias=bias)
        self._mask_cache: dict[tuple[str, int, torch.device], object] = {}

    @torch.compiler.disable
    def _attn_mask(self, seq_len: int, device: torch.device):
        """Flex BlockMask 或 SDPA 加性 mask；按 seq_len 缓存，避免每层重建 L×L。

        必须 eager：与 ``encdec.SelfAttention._block_mask`` 相同，避免 Dynamo+DDP
        把 ``seq_len`` 送进子图后 Inductor 报 ``int`` 无 ``.meta``。
        """
        key = (self.attn_backend, seq_len, device)
        cached = self._mask_cache.get(key)
        if cached is None:
            if self.attn_backend == "flex":
                mask_mod = partial(block_causal_mask_mod, block_size=self.block_size)
                cached = create_block_mask(
                    mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
                )
            else:
                cached = bool_mask_to_sdpa_additive(
                    build_block_causal_mask(seq_len, self.block_size, device),
                )
            if len(self._mask_cache) > 16:
                self._mask_cache.pop(next(iter(self._mask_cache)))
            self._mask_cache[key] = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        h = x if self.post_norm else self.norm_attn(x)
        qkv = self.attn.c_attn(h)
        q, k, v = qkv.split(self.n_embd, dim=2)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        q, k = self.rope.apply_qk(q, k, positions)
        attn_mask = self._attn_mask(seq_len, x.device)
        if self.attn_backend == "flex":
            y = fused_flex_attention(q, k, v, block_mask=attn_mask)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        attn = self.attn.c_proj(y)
        residual = self.norm_attn(x) if self.post_norm else x
        x = residual + self.resid_dropout(attn)

        residual = x
        h = x if self.post_norm else self.norm_ffn(x)
        h = self.mlp(h)
        if self.post_norm:
            h = self.norm_ffn(h)
        return residual + h
