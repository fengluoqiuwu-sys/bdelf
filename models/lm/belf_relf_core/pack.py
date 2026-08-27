"""2L pack ``[clean|noisy]`` 与组因果掩码。

组内双向、组间单向（后组可见前组）。左右组大小可不同：
BELF 两侧都是 ``W``；RELF 左段 ``group=1``、右段 ``group=S``。
"""

from __future__ import annotations

import torch


def pack_2l(h_left: torch.Tensor, h_right: torch.Tensor) -> torch.Tensor:
    """沿长度维拼接 ``[h_left | h_right]``。左段 stop-grad 由调用方做。"""
    return torch.cat([h_left, h_right], dim=1)


def group_causal_mask(
    seq_len: int,
    group_size: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """布尔 ``(L, L)``：True=可见。组内双向、组间因果。"""
    if int(group_size) < 1:
        raise ValueError(f"group_size 须 >= 1，收到 {group_size}")
    q = torch.arange(seq_len, device=device)[:, None]
    kv = torch.arange(seq_len, device=device)[None, :]
    return (kv // int(group_size)) <= (q // int(group_size))


def pack_2l_mask(
    left_len: int,
    right_len: int,
    left_group: int,
    right_group: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """2L ``[clean|noisy]`` 可见性，左右组大小可不同。

    干净半段：组因果，不看噪声半段。
    噪声 Q 的组 ``b``：看干净 K 的更早组 + 噪声半段中自身及更早组（档间单向）。

    可见规则对齐 Cola ``build_cola_2l_mask``（噪声只看更早干净组 + 自身噪声组；
    干净半段组因果、不看噪声）。组号：左段从 0 起，右段接到左段组号之后，
    因此前缀相对当前块/窗一律是更早组（右段可见全部左段）。
    """
    if int(left_group) < 1 or int(right_group) < 1:
        raise ValueError(
            f"组大小须 >= 1，收到 left_group={left_group}, right_group={right_group}"
        )
    two = int(left_len) + int(right_len)
    n_left = int(left_len)
    lg = int(left_group)
    rg = int(right_group)
    idx = torch.arange(two, device=device)
    is_clean = idx < n_left
    local = torch.where(is_clean, idx, idx - n_left)
    # 左段组号从 0 起；右段接到左段之后，使前缀相对当前块/窗都是「更早组」。
    n_left_groups = (n_left + lg - 1) // lg if n_left > 0 else 0
    blk = torch.where(is_clean, local // lg, n_left_groups + local // rg)
    q_blk = blk[:, None]
    k_blk = blk[None, :]
    q_clean = is_clean[:, None]
    k_clean = is_clean[None, :]
    noisy_q_clean_k = (~q_clean) & k_clean & (q_blk > k_blk)
    noisy_q_noisy_k = (~q_clean) & (~k_clean) & (q_blk >= k_blk)
    clean_q_clean_k = q_clean & k_clean & (q_blk >= k_blk)
    return noisy_q_clean_k | noisy_q_noisy_k | clean_q_clean_k


def pack_2l_parallel_blocks_mask(
    seq_len: int,
    block_size: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """训练用并行块 2L：左右对齐同一文档块号。

    noisy Q 只看更早 clean 块（``k_blk < q_blk``），噪声块之间仅同组可见；
    clean 组因果、不看噪声。与 ``pack_2l_mask``（前缀|当前块/窗）不同，
    专给 ``[全长 clean | 全长 noisy]``。
    """
    n = int(seq_len)
    w = int(block_size)
    if n < 1 or w < 1:
        raise ValueError(f"seq_len 与 block_size 须 >= 1，收到 {seq_len}, {block_size}")
    two = 2 * n
    idx = torch.arange(two, device=device)
    is_clean = idx < n
    local = torch.where(is_clean, idx, idx - n)
    blk = local // w
    q_blk = blk[:, None]
    k_blk = blk[None, :]
    q_clean = is_clean[:, None]
    k_clean = is_clean[None, :]
    noisy_q_clean_k = (~q_clean) & k_clean & (q_blk > k_blk)
    noisy_q_noisy_k = (~q_clean) & (~k_clean) & (q_blk == k_blk)
    clean_q_clean_k = q_clean & k_clean & (q_blk >= k_blk)
    return noisy_q_clean_k | noisy_q_noisy_k | clean_q_clean_k
