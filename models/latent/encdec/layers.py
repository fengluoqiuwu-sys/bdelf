"""T5-small 维数 Transformer block：RoPE + 独立 d_kv + 三路注意力 mask。"""

from __future__ import annotations

from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.latent.cola_vae.layers import (
    block_causal_mask_mod,
    bool_mask_to_sdpa_additive,
    build_block_causal_mask,
    fused_flex_attention,
)
from models.rope import RotaryEmbedding

try:
    from torch.nn.attention.flex_attention import create_block_mask

    FLEX_ATTN_AVAILABLE = True
except ImportError:
    create_block_mask = None  # type: ignore[assignment]
    FLEX_ATTN_AVAILABLE = False

AttnMode = Literal["bidirectional", "causal", "block_causal"]


def key_padding_additive(
    pad: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``(B, S)`` True=pad → ``(B, 1, 1, S)`` 加性 mask，pad 键为 ``-inf``。"""
    return torch.zeros(
        pad.size(0), 1, 1, pad.size(1),
        dtype=dtype or torch.float32, device=pad.device,
    ).masked_fill(pad.unsqueeze(1).unsqueeze(1), float("-inf"))


class SelfAttention(nn.Module):
    """Pre-norm self-attention；Q/K/V 投影到 ``n_head * d_kv``。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        dropout: float,
        *,
        use_flash: bool = True,
        attn_backend: str = "sdpa",
        block_size: int = 1,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_kv = d_kv
        self.n_embd = n_embd
        self.inner_dim = n_head * d_kv
        self.dropout = dropout
        self.use_flash = use_flash
        self.attn_backend = attn_backend
        self.block_size = block_size
        qkv_dim = 3 * self.inner_dim
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(n_embd, qkv_dim)
        self.attn.proj = nn.Linear(self.inner_dim, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(d_kv)
        self._mask_cache: dict[tuple, object] = {}

    def _resolve_mode(self, attn_mode: AttnMode) -> AttnMode:
        if attn_mode == "block_causal" and self.block_size <= 1:
            return "causal"
        return attn_mode

    @torch.compiler.disable
    def _block_mask(self, seq_len: int, device: torch.device):
        """块因果 mask（Flex BlockMask 或 SDPA 加性）。

        必须 eager：Dynamo+DDP 若把 ``seq_len`` 当 Python int 送进子图，
        Inductor 会在 ``n.meta['val']`` 上报 ``'int' object has no attribute 'meta'``
        （``block_size>1`` 在课程切到更长 L 时复现）。
        """
        key = (self.attn_backend, seq_len, device, self.block_size)
        cached = self._mask_cache.get(key)
        if cached is None:
            if self.attn_backend == "flex":
                if not FLEX_ATTN_AVAILABLE:
                    raise RuntimeError("attn_backend=flex 需要 PyTorch FlexAttention")
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mode: AttnMode = "causal",
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_mode = self._resolve_mode(attn_mode)
        bsz, seq_len, _ = x.size()
        qkv = self.attn.qkv(x)
        q, k, v = qkv.split(self.inner_dim, dim=2)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.n_head, self.d_kv).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        q, k = self.rope.apply_qk(q, k, positions)

        drop = self.dropout if self.training else 0.0
        # 右 pad + 逐 token 因果：真实 token 本来看不到右侧 pad，勿加 attn_mask，否则 SDPA 走不出 Flash。
        use_pad = key_padding_mask is not None and (
            attn_mode == "bidirectional"
            or (attn_mode == "block_causal" and self.block_size > 1)
        )
        pad_add = (
            key_padding_additive(key_padding_mask, dtype=q.dtype)
            if use_pad
            else None
        )

        if attn_mode == "bidirectional":
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=pad_add, dropout_p=drop,
            )
        elif attn_mode == "causal":
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=pad_add, dropout_p=drop, is_causal=True,
            )
        else:
            attn_mask = self._block_mask(seq_len, x.device)
            if self.attn_backend == "flex" and pad_add is None:
                y = fused_flex_attention(q, k, v, block_mask=attn_mask)
            else:
                if not isinstance(attn_mask, torch.Tensor):
                    attn_mask = bool_mask_to_sdpa_additive(
                        build_block_causal_mask(seq_len, self.block_size, x.device),
                    )
                if pad_add is not None:
                    attn_mask = attn_mask + pad_add
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=drop,
                )

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.inner_dim)
        return self.resid_dropout(self.attn.proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(n_embd, d_ff)
        self.mlp.c_proj = nn.Linear(d_ff, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.mlp.c_proj(F.gelu(self.mlp.c_fc(x))))


class TransformerBlock(nn.Module):
    """Pre-norm：self-attn + FFN。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        d_ff: int,
        dropout: float,
        *,
        use_flash: bool = True,
        attn_backend: str = "sdpa",
        block_size: int = 1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(
            n_embd, n_head, d_kv, dropout,
            use_flash=use_flash,
            attn_backend=attn_backend,
            block_size=block_size,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mode: AttnMode = "causal",
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mode=attn_mode, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CrossAttention(nn.Module):
    """Decoder cross-attn：Q 来自 decoder E；K/V 来自 memory。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        memory_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_kv = d_kv
        self.inner_dim = n_head * d_kv
        self.cross_attn = nn.Module()
        self.cross_attn.q_proj = nn.Linear(n_embd, self.inner_dim)
        self.cross_attn.k_proj = nn.Linear(memory_dim, self.inner_dim)
        self.cross_attn.v_proj = nn.Linear(memory_dim, self.inner_dim)
        self.cross_attn.proj = nn.Linear(self.inner_dim, n_embd)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        mem_len = memory.size(1)
        q = self.cross_attn.q_proj(x)
        k = self.cross_attn.k_proj(memory)
        v = self.cross_attn.v_proj(memory)

        def reshape_heads(t: torch.Tensor, length: int) -> torch.Tensor:
            return t.view(bsz, length, self.n_head, self.d_kv).transpose(1, 2)

        q = reshape_heads(q, seq_len)
        k = reshape_heads(k, mem_len)
        v = reshape_heads(v, mem_len)
        pad_add = (
            key_padding_additive(key_padding_mask, dtype=q.dtype)
            if key_padding_mask is not None
            else None
        )
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=pad_add,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.inner_dim)
        return self.resid_dropout(self.cross_attn.proj(y))


class DecoderBlock(nn.Module):
    """self-attn（模式由调用方传入）+ cross-attn + FFN。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        d_ff: int,
        memory_dim: int,
        dropout: float,
        *,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.self_attn = SelfAttention(
            n_embd, n_head, d_kv, dropout,
            use_flash=use_flash,
            attn_backend="sdpa",
            block_size=1,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.cross_attn = CrossAttention(
            n_embd, n_head, d_kv, memory_dim, dropout,
        )
        self.ln3 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        attn_mode: AttnMode = "causal",
        key_padding_mask: torch.Tensor | None = None,
        memory_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.ln1(x), attn_mode=attn_mode, key_padding_mask=key_padding_mask,
        )
        x = x + self.cross_attn(
            self.ln2(x), memory, key_padding_mask=memory_pad_mask,
        )
        x = x + self.mlp(self.ln3(x))
        return x
