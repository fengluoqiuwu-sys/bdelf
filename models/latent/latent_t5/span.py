"""T5 span corruption（encoder 侧 sentinel 替换）。"""

from __future__ import annotations

import torch


def span_corruption_mask(
    shape: tuple[int, ...],
    *,
    mask_ratio: float,
    mean_span_len: int,
    device: torch.device,
) -> torch.Tensor:
    """按 span 级覆盖约 ``mask_ratio``（平均 span 长 ``mean_span_len``）。

    与文档一致：逐 span 涂到覆盖达标（允许重叠）；不抽第 0 位。
    在 CPU 上跑循环，避免逐步 ``.item()`` 造成 GPU 同步。
    """
    bsz, seq_len = shape
    mask = torch.zeros(bsz, seq_len, dtype=torch.bool)
    target = max(1, int(seq_len * mask_ratio))
    span_len = max(1, min(int(mean_span_len), seq_len))
    max_start = max(1, seq_len - span_len + 1)
    for b in range(bsz):
        covered = 0
        attempts = 0
        row = mask[b]
        while covered < target and attempts < seq_len * 4:
            attempts += 1
            start = int(torch.randint(0, max_start, (1,)).item())
            end = min(start + span_len, seq_len)
            sl = row[start:end]
            covered += int((~sl).sum().item())
            row[start:end] = True
    mask[:, 0] = False
    return mask.to(device=device, non_blocking=True)


def apply_span_sentinels(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    vocab_size: int,
    num_sentinels: int,
) -> torch.Tensor:
    """把 mask 位置换成随机 sentinel id（``vocab_size + [0, num_sentinels)``）。"""
    if num_sentinels <= 0:
        raise ValueError("num_sentinels must be positive")
    sentinel_offsets = torch.randint(
        0, num_sentinels, mask.shape, device=tokens.device, dtype=torch.long,
    )
    sentinel_ids = vocab_size + sentinel_offsets
    return torch.where(mask, sentinel_ids, tokens)
