"""BELF / RELF 的 FlexAttention ``mask_mod``：可见性与现有布尔 mask 对齐。

静态 2L 复用 Cola ``cola_2l_mask_mod``。PAD / ``ctx_p_drop`` / RELF 窗场
走 ``create_block_mask``（``B=batch``），不用会打散融合的 ``score_mod`` 查表。
生成路径仍用 SDPA。
"""

from __future__ import annotations

from functools import partial

import torch

from models.lm.belf_relf_core.pack import group_causal_mask
from models.lm.cola.layers import cola_2l_mask_mod
from models.latent.cola_vae.layers import FLEX_ATTN_AVAILABLE, fused_flex_attention

try:
    from torch.nn.attention.flex_attention import create_block_mask
except ImportError:
    create_block_mask = None  # type: ignore[assignment]


def require_flex() -> None:
    if not FLEX_ATTN_AVAILABLE or create_block_mask is None:
        raise RuntimeError("attn_backend=flex 需要 PyTorch FlexAttention；请升级或改用 sdpa")


def belf_parallel_2l_mask_mod(b, h, q_idx, kv_idx, block_size: int, n: int):
    """与 ``pack_2l_parallel_blocks_mask`` / Cola 2L 相同。"""
    return cola_2l_mask_mod(b, h, q_idx, kv_idx, block_size, n)


def make_belf_train_mask_mod(
    n: int,
    block_size: int,
    *,
    is_pad: torch.Tensor | None = None,
    unknown: torch.Tensor | None = None,
    drop_left: torch.Tensor | None = None,
):
    """并行 2L + 规格额外：未知 Q 不见同块 PAD K；``drop_left`` 切断左段。"""
    w = int(block_size)
    n_int = int(n)

    def mask_mod(b, h, q_idx, kv_idx):
        vis = cola_2l_mask_mod(b, h, q_idx, kv_idx, w, n_int)
        q_right = q_idx >= n_int
        k_left = kv_idx < n_int
        k_right = ~k_left
        if is_pad is not None and unknown is not None:
            q_loc = (q_idx - n_int).clamp(min=0, max=n_int - 1)
            k_loc = (kv_idx - n_int).clamp(min=0, max=n_int - 1)
            same_blk = (q_loc // w) == (k_loc // w)
            hide_pad = (
                q_right & k_right & unknown[b, q_loc] & is_pad[b, k_loc] & same_blk
            )
            vis = vis & (~hide_pad)
        if drop_left is not None:
            hide_drop = drop_left[b] & q_right & k_left
            vis = vis & (~hide_drop)
        return vis

    return mask_mod


@torch.compiler.disable
def build_belf_flex_block_mask(
    left_len: int,
    block_size: int,
    device: torch.device,
    *,
    is_pad: torch.Tensor | None = None,
    unknown: torch.Tensor | None = None,
    drop_left: torch.Tensor | None = None,
):
    """训练并行 2L 的 ``BlockMask``。有 PAD / drop 时 ``B=batch``。"""
    require_flex()
    n = int(left_len)
    two = 2 * n
    extra = (is_pad is not None and unknown is not None) or drop_left is not None
    if extra:
        bsz = int(
            (is_pad if is_pad is not None else drop_left).size(0)
        )
        mask_mod = make_belf_train_mask_mod(
            n, block_size, is_pad=is_pad, unknown=unknown, drop_left=drop_left,
        )
        return create_block_mask(
            mask_mod, B=bsz, H=None, Q_LEN=two, KV_LEN=two, device=device,
        )
    mask_mod = partial(belf_parallel_2l_mask_mod, block_size=int(block_size), n=n)
    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=two, KV_LEN=two, device=device,
    )


def relf_windows_visible(
    left_len: int,
    window_size: int,
    step_size: int,
    u: torch.Tensor,
    active: torch.Tensor,
    *,
    k0: torch.Tensor | None = None,
    in_win: torch.Tensor | None = None,
    drop_left: torch.Tensor | None = None,
) -> torch.Tensor:
    """布尔 ``(B, two, two)``：与 RELF 训练 SDPA 路径同一套规则。"""
    device = u.device
    bsz, n_win = u.shape
    w_sz = int(window_size)
    right_len = n_win * w_sz
    two = int(left_len) + right_len
    vis = u.new_zeros(bsz, two, two, dtype=torch.bool)
    if left_len > 0:
        q = torch.arange(left_len, device=device)
        vis[:, :left_len, :left_len] = q[None, :] <= q[:, None]
    ridx = torch.arange(right_len, device=device)
    loc = ridx % w_sz
    win_vis = group_causal_mask(w_sz, int(step_size), device=device)
    same = (ridx[:, None] // w_sz) == (ridx[None, :] // w_sz)
    vis[:, left_len:, left_len:] = same & win_vis[loc[:, None], loc[None, :]]
    g = u if k0 is None else u + k0
    kv_end = g.clamp(min=0, max=left_len)[:, ridx // w_sz]
    if left_len > 0:
        left_idx = torch.arange(left_len, device=device)
        vis[:, left_len:, :left_len] = left_idx[None, None, :] < kv_end[:, :, None]
    act_r = active[:, ridx // w_sz]
    vis[:, left_len:, :] = vis[:, left_len:, :] & act_r[:, :, None]
    eye_r = ridx[:, None] == ridx[None, :]
    vis[:, left_len:, left_len:] = vis[:, left_len:, left_len:] | (
        (~act_r[:, :, None]) & eye_r
    )
    if in_win is not None:
        in_flat = in_win.reshape(bsz, right_len)
        vis[:, left_len:, left_len:] = (
            vis[:, left_len:, left_len:]
            & in_flat[:, None, :]
            & in_flat[:, :, None]
        )
        vis[:, left_len:, left_len:] = vis[:, left_len:, left_len:] | (
            (~in_flat)[:, :, None] & eye_r
        )
    if drop_left is not None and left_len > 0:
        vis[drop_left, left_len:, :left_len] = False
    return vis


def make_relf_windows_mask_mod(
    left_len: int,
    window_size: int,
    step_size: int,
    n_win: int,
    u: torch.Tensor,
    active: torch.Tensor,
    *,
    k0: torch.Tensor | None = None,
    in_win: torch.Tensor | None = None,
    drop_left: torch.Tensor | None = None,
):
    """与 ``_RelfBackbone._windows_attn_mask`` 同一套可见性。"""
    left = int(left_len)
    w_sz = int(window_size)
    step = int(step_size)
    right_len = int(n_win) * w_sz
    two = left + right_len
    last_r = max(right_len - 1, 0)
    last_w = max(int(n_win) - 1, 0)
    k0_t = k0
    in_w = in_win
    drop = drop_left

    def mask_mod(b, h, q_idx, kv_idx):
        del h
        q_left = q_idx < left
        k_left = kv_idx < left
        q_right = ~q_left
        k_right = ~k_left
        left_vis = q_left & k_left & (kv_idx <= q_idx)

        q_r = (q_idx - left).clamp(min=0, max=last_r)
        k_r = (kv_idx - left).clamp(min=0, max=last_r)
        q_win = (q_r // w_sz).clamp(max=last_w)
        k_win = (k_r // w_sz).clamp(max=last_w)
        q_loc = q_r % w_sz
        k_loc = k_r % w_sz
        same = q_win == k_win
        grp = (k_loc // step) <= (q_loc // step)
        rr_base = q_right & k_right & same & grp

        g = u[b, q_win]
        if k0_t is not None:
            g = g + k0_t[b, q_win]
        kv_end = g.clamp(min=0, max=left)
        rl = q_right & k_left & (kv_idx < kv_end)

        act = active[b, q_win]
        vis = left_vis | ((rr_base | rl) & act)
        eye = q_right & k_right & (q_r == k_r)
        vis = vis | (q_right & (~act) & eye)

        if in_w is not None:
            in_q = in_w[b, q_win, q_loc]
            in_k = in_w[b, k_win, k_loc]
            vis = vis & ((~q_right) | (~k_right) | (in_q & in_k))
            vis = vis | (q_right & k_right & (~in_q) & eye)
        if drop is not None and left > 0:
            vis = vis & ~(drop[b] & q_right & k_left)
        return vis

    return mask_mod, two


@torch.compiler.disable
def build_relf_flex_block_mask(
    left_len: int,
    window_size: int,
    step_size: int,
    u: torch.Tensor,
    active: torch.Tensor,
    *,
    k0: torch.Tensor | None = None,
    in_win: torch.Tensor | None = None,
    drop_left: torch.Tensor | None = None,
):
    """RELF 窗 2L：每步 ``B=batch``（``u`` / ``k0`` 随样本变）。"""
    require_flex()
    n_win = int(u.size(1))
    mask_mod, two = make_relf_windows_mask_mod(
        left_len, window_size, step_size, n_win, u, active,
        k0=k0, in_win=in_win, drop_left=drop_left,
    )
    bsz = int(u.size(0))
    return create_block_mask(
        mask_mod, B=bsz, H=None, Q_LEN=two, KV_LEN=two, device=u.device,
    )


def materialize_mask_mod(
    mask_mod,
    *,
    q_len: int,
    kv_len: int,
    device: torch.device,
    batch: int | None = None,
) -> torch.Tensor:
    """把 ``mask_mod`` 铺成布尔 ``(Q, K)`` 或 ``(B, Q, K)``，供对照测试。"""
    q = torch.arange(q_len, device=device)
    k = torch.arange(kv_len, device=device)
    qq = q[:, None].expand(q_len, kv_len)
    kk = k[None, :].expand(q_len, kv_len)
    if batch is None:
        return mask_mod(0, 0, qq, kk)
    rows = [
        mask_mod(b, 0, qq, kk) for b in range(int(batch))
    ]
    return torch.stack(rows, dim=0)


__all__ = [
    "FLEX_ATTN_AVAILABLE",
    "build_belf_flex_block_mask",
    "build_relf_flex_block_mask",
    "belf_parallel_2l_mask_mod",
    "fused_flex_attention",
    "make_belf_train_mask_mod",
    "make_relf_windows_mask_mod",
    "materialize_mask_mod",
    "relf_windows_visible",
    "require_flex",
]
