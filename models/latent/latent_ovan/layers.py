"""csh BlockVAE 风格块：无 bias LN、QK-RMSNorm、SwiGLU、块因果 SDPA。"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.latent.cola_vae.layers import (
    block_causal_mask_mod,
    bool_mask_to_sdpa_additive,
    build_block_causal_mask,
    fused_flex_attention,
)
from models.latent.encdec.layers import key_padding_additive
from models.rope import RotaryEmbedding

try:
    from torch.nn.attention.flex_attention import create_block_mask

    FLEX_ATTN_AVAILABLE = True
except ImportError:
    create_block_mask = None  # type: ignore[assignment]
    FLEX_ATTN_AVAILABLE = False


def xavier_linear(layer: nn.Linear) -> None:
    """Xavier uniform + bias 0（对齐 csh / ELF）。"""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class RMSNorm(nn.Module):
    """无中心化 RMSNorm；weight=1。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(dtype)
        return self.weight.to(dtype) * x


class SwiGLU(nn.Module):
    """``d → 2×⌊2/3·mult·d⌋``，SiLU(x1)⊙x2 → d。"""

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        hidden_eff = int(dim * mult * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_eff)
        self.w3 = nn.Linear(hidden_eff, dim)
        xavier_linear(self.w12)
        xavier_linear(self.w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class SelfAttention(nn.Module):
    """Pre-norm 之后的自注意力；Q/K/V 到 ``n_head * d_kv``。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        dropout: float,
        *,
        qk_norm: bool = True,
        use_flash: bool = True,
        attn_backend: str = "sdpa",
        block_size: int = 16,
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
        self.qkv = nn.Linear(n_embd, 3 * self.inner_dim)
        self.proj = nn.Linear(self.inner_dim, n_embd)
        xavier_linear(self.qkv)
        xavier_linear(self.proj)
        self.q_norm: nn.Module = RMSNorm(d_kv) if qk_norm else nn.Identity()
        self.k_norm: nn.Module = RMSNorm(d_kv) if qk_norm else nn.Identity()
        self._mask_cache: dict[tuple, object] = {}

    @torch.compiler.disable
    def _block_mask(self, seq_len: int, device: torch.device):
        """块因果 mask（Flex BlockMask 或 SDPA 加性）。"""
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
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        rope: RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.inner_dim, dim=2)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.n_head, self.d_kv).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            if positions is None:
                positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            q, k = rope.apply_qk(q, k, positions)

        drop = self.dropout if self.training else 0.0
        use_block = self.block_size > 1
        use_pad = key_padding_mask is not None and use_block
        pad_add = (
            key_padding_additive(key_padding_mask, dtype=q.dtype)
            if use_pad
            else None
        )

        if not use_block:
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
        y = self.proj(y)
        if self.training and self.dropout > 0.0:
            y = F.dropout(y, p=self.dropout, training=True)
        return y


class CausalBlock(nn.Module):
    """无 AdaLN 的 pre-norm 块（对齐 csh BlockVAE）。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        dropout: float,
        *,
        mlp_mult: int = 4,
        qk_norm: bool = True,
        use_flash: bool = True,
        attn_backend: str = "sdpa",
        block_size: int = 16,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = SelfAttention(
            n_embd, n_head, d_kv, dropout,
            qk_norm=qk_norm,
            use_flash=use_flash,
            attn_backend=attn_backend,
            block_size=block_size,
        )
        self.ln2 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = SwiGLU(n_embd, mult=mlp_mult)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        rope: RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x),
            key_padding_mask=key_padding_mask,
            positions=positions,
            rope=rope,
        )
        h2 = self.mlp(self.ln2(x))
        if self.training and self.attn.dropout > 0.0:
            h2 = F.dropout(h2, p=self.attn.dropout, training=True)
        return x + h2


class VaeHeads(nn.Module):
    """μ / logσ² 头；无 LayerNorm（对齐 csh）。"""

    def __init__(
        self,
        n_embd: int,
        latent_dim: int,
        *,
        logvar_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.to_mu = nn.Linear(n_embd, latent_dim)
        self.to_logvar = nn.Linear(n_embd, latent_dim)
        if n_embd == latent_dim:
            nn.init.eye_(self.to_mu.weight)
            nn.init.zeros_(self.to_mu.bias)
        else:
            xavier_linear(self.to_mu)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, float(logvar_init))

    def forward(
        self,
        h: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-20.0, 20.0)
        if sample:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z, mu, logvar
