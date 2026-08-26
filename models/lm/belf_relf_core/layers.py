"""AdaLN-Zero 去噪器骨干：逐列 ``(t, w_sc, m)``，无 ELF in-context 时间 token。

对齐 Cola DiT 的 AdaLN-Zero（无仿射 LN、残差 gate、调制与 FinalLayer 零初始化），
骨干为 RoPE、qk-RMSNorm、SwiGLU。条件必须逐位置，不用 Cola 的 ``t * ode_T``。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import RotaryEmbedding

# 每列模式：0=none（左/已知/PAD），1=denoise，2=decode。
_MODE_NONE = 0
_MODE_DENOISE = 1
_MODE_DECODE = 2


def _as_sdpa_mask(attn_mask: torch.Tensor | None) -> torch.Tensor | None:
    """布尔 ``True=可见`` → 加性 SDPA mask；浮点原样。避免 bool 极性踩坑。"""
    if attn_mask is None:
        return None
    if attn_mask.dtype != torch.bool:
        return attn_mask
    vis = attn_mask
    if vis.ndim == 2:
        vis = vis.view(1, 1, vis.size(0), vis.size(1))
    elif vis.ndim == 3:
        vis = vis.unsqueeze(1)
    elif vis.ndim != 4:
        raise ValueError(f"attn_mask bool 维数须为 2/3/4，收到 {tuple(vis.shape)}")
    additive = torch.zeros(vis.shape, dtype=torch.float32, device=vis.device)
    additive.masked_fill_(~vis, float("-inf"))
    return additive


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """正弦时间嵌入。对齐 Cola / diffusers：sin 再 cos。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """对齐 Cola ``TimestepEmbedder``：Linear→SiLU→Linear→SiLU→Linear。

    ``t`` 为 ``(B,)`` 或 ``(B, L)``，返回 ``(B, C)`` 或 ``(B, L, C)``。
    """

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
        orig = t.shape
        raw = timestep_embedding(t.reshape(-1), self.frequency_embedding_size)
        raw = raw.to(dtype=self.mlp[0].weight.dtype)
        emb = self.mlp(raw)
        if t.ndim == 2:
            return emb.view(orig[0], orig[1], -1)
        return emb


class ScaleEmbedder(nn.Module):
    """把标量 ``w_sc`` 嵌到 ``D``；与时间嵌入同构（正弦 + MLP）。"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.inner = TimestepEmbedder(hidden_size, frequency_embedding_size)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.inner(w)


class _RMSNorm(nn.Module):
    """每头 Q/K RMSNorm。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(dtype)
        return self.weight.to(dtype) * x


class _SwiGLU(nn.Module):
    """LLaMA / ELF 风格 SwiGLU：有效中间宽 ``(2/3)*mlp_ratio*D``。"""

    def __init__(
        self,
        n_embd: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_eff = int(n_embd * mlp_ratio * 2.0 / 3.0)
        self.w12 = nn.Linear(n_embd, 2 * hidden_eff)
        self.w3 = nn.Linear(hidden_eff, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.dropout(self.w3(F.silu(x1) * x2))


class _AdaLNAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        *,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        qk_norm: bool = True,
        attn_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) 须能被 n_head ({n_head}) 整除")
        if attn_backend != "sdpa":
            raise ValueError(f"attn_backend 仅支持 sdpa，收到 {attn_backend!r}")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.attn_backend = attn_backend
        rd = self.head_dim if rope_dim is None else int(rope_dim)
        if rd <= 0 or rd > self.head_dim or rd % 2 != 0:
            raise ValueError(
                f"rope_dim={rd} 须为偶数且 0 < rope_dim <= head_dim={self.head_dim}"
            )
        self.rope_dim = rd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = _RMSNorm(self.head_dim)
            self.k_norm = _RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.rope_dim, base=rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
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

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        return self.resid_dropout(self.c_proj(y))


class _DiTBlock(nn.Module):
    """AdaLN-Zero 块：无仿射 LN、残差 gate、调制末层零初始化。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        *,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        qk_norm: bool = True,
        mlp_ratio: float = 4.0,
        attn_backend: str = "sdpa",
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=norm_eps)
        self.attn = _AdaLNAttention(
            n_embd, n_head, dropout=dropout,
            rope_dim=rope_dim, rope_theta=rope_theta,
            qk_norm=qk_norm, attn_backend=attn_backend,
        )
        self.norm2 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=norm_eps)
        self.mlp = _SwiGLU(n_embd, mlp_ratio=mlp_ratio, dropout=dropout)
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
        attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod = self.adaLN_modulation(cond)
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        h = self.norm1(x) * (1.0 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(h, attn_mask=attn_mask, positions=positions)
        h = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)
        return x


class _FinalLayer(nn.Module):
    def __init__(self, n_embd: int, out_dim: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
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


class AdaLNZeroStack(nn.Module):
    """DiT AdaLN-Zero 栈。``num_time_tokens`` 必须为 0。

    逐列模式 ``m``：``0=none`` 只吃 ``t=1``、不吃 ``w``、不加 ``m``；
    ``1=denoise`` 吃 ``t,w,m``；``2=decode`` 吃 ``t`` 与 decode 模式、不吃 ``w``。
    """

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        n_head: int,
        *,
        out_dim: int | None = None,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        qk_norm: bool = True,
        attn_backend: str = "sdpa",
        num_time_tokens: int = 0,
        use_scale: bool = True,
        t_freq_dim: int = 256,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if int(num_time_tokens) != 0:
            raise ValueError(
                f"AdaLNZeroStack 要求 num_time_tokens=0，收到 {num_time_tokens}"
            )
        if attn_backend != "sdpa":
            raise ValueError(f"attn_backend 仅支持 sdpa，收到 {attn_backend!r}")
        self.n_embd = n_embd
        self.attn_backend = attn_backend
        self.use_scale = bool(use_scale)
        out = n_embd if out_dim is None else int(out_dim)
        self.t_embedder = TimestepEmbedder(n_embd, frequency_embedding_size=t_freq_dim)
        self.w_embedder = (
            ScaleEmbedder(n_embd, frequency_embedding_size=t_freq_dim)
            if self.use_scale
            else None
        )
        self.mode_embed = nn.Embedding(3, n_embd)
        nn.init.zeros_(self.mode_embed.weight)
        self.blocks = nn.ModuleList(
            [
                _DiTBlock(
                    n_embd, n_head, dropout=dropout,
                    rope_dim=rope_dim, rope_theta=rope_theta,
                    qk_norm=qk_norm, mlp_ratio=mlp_ratio,
                    attn_backend=attn_backend, norm_eps=norm_eps,
                )
                for _ in range(n_layer)
            ]
        )
        self.final = _FinalLayer(n_embd, out, norm_eps=norm_eps)

    def _broadcast_t(self, t: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        if t.ndim == 1:
            return t[:, None].expand(batch, seq_len)
        if t.ndim == 2:
            return t
        raise ValueError(f"t 须为 (B,) 或 (B, L)，收到 {tuple(t.shape)}")

    def _broadcast_w(
        self,
        w: torch.Tensor | None,
        batch: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if w is None:
            return torch.zeros(batch, seq_len, device=device, dtype=dtype)
        if w.ndim == 1:
            return w[:, None].expand(batch, seq_len)
        if w.ndim == 2:
            return w
        raise ValueError(f"w_sc 须为 (B,) 或 (B, L)，收到 {tuple(w.shape)}")

    def _cond(
        self,
        t: torch.Tensor,
        w_sc: torch.Tensor | None,
        m: torch.Tensor | None,
        *,
        batch: int,
        seq_len: int,
    ) -> torch.Tensor:
        t_pos = self._broadcast_t(t, batch, seq_len)
        if m is None:
            m = torch.full(
                (batch, seq_len), _MODE_DENOISE, device=t_pos.device, dtype=torch.long,
            )
        else:
            m = m.to(device=t_pos.device, dtype=torch.long)
            if m.ndim == 1:
                m = m[:, None].expand(batch, seq_len)
        # none 列钉 t=1；denoise / decode 用调用方给的 t。
        t_eff = torch.where(m == _MODE_NONE, torch.ones_like(t_pos), t_pos)
        cond = self.t_embedder(t_eff)
        if self.w_embedder is not None:
            w_pos = self._broadcast_w(
                w_sc, batch, seq_len, t_pos.device, t_pos.dtype,
            )
            w_emb = self.w_embedder(w_pos)
            denoise = (m == _MODE_DENOISE).unsqueeze(-1).to(dtype=cond.dtype)
            cond = cond + w_emb * denoise
        m_emb = self.mode_embed(m.clamp(0, 2))
        known = (m != _MODE_NONE).unsqueeze(-1).to(dtype=cond.dtype)
        return cond + m_emb * known

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        w_sc: torch.Tensor | None = None,
        m: torch.Tensor | None = None,
        *,
        attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x`` 为 ``(B, L, D)``；返回 x-pred，形状 ``(B, L, out_dim)``。"""
        bsz, seq_len, _ = x.shape
        if x.size(-1) != self.n_embd:
            raise ValueError(
                f"AdaLNZeroStack 入口宽 {x.size(-1)} 须等于 n_embd={self.n_embd}"
            )
        cond = self._cond(t, w_sc, m, batch=bsz, seq_len=seq_len)
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        attn_mask = _as_sdpa_mask(attn_mask)
        for block in self.blocks:
            x = block(x, cond, attn_mask=attn_mask, positions=positions)
        return self.final(x, cond)
