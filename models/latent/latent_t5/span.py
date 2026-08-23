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
    """生成布尔 mask，约覆盖 ``mask_ratio`` 的 token（span 级）。"""
    bsz, seq_len = shape
    mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
    target = max(1, int(seq_len * mask_ratio))
    span_len = max(1, mean_span_len)
    for b in range(bsz):
        covered = 0
        attempts = 0
        while covered < target and attempts < seq_len * 4:
            attempts += 1
            length = min(span_len, seq_len)
            if length <= 0:
                break
            start = torch.randint(0, max(1, seq_len - length + 1), (1,), device=device).item()
            end = min(start + length, seq_len)
            new = mask[b, start:end]
            covered += int((~new).sum().item())
            mask[b, start:end] = True
    mask[:, 0] = False
    return mask


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
