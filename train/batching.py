"""Training-time dataset wrappers and deterministic batch indexing."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class TokenChunkDataset(Dataset):
    """Wraps a preprocessed split; yields ``input_ids`` tensors."""

    def __init__(self, split_ds: Dataset) -> None:
        self.split_ds = split_ds

    def __len__(self) -> int:
        return len(self.split_ds)

    def __getitem__(self, index: int) -> torch.Tensor:
        item = self.split_ds[index]
        if isinstance(item, dict):
            return item["input_ids"]
        return item


def collate_input_ids(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch)


def build_eval_subset(
    dataset: Dataset,
    sample_count: int | None,
    seed: int,
) -> tuple[Dataset, int]:
    """Return eval dataset, optionally subsampled without mutating the source split."""
    total = len(dataset)
    if sample_count is None or sample_count >= total:
        return dataset, total
    gen = torch.Generator()
    gen.manual_seed(seed)
    indices = torch.randperm(total, generator=gen)[:sample_count].tolist()
    return Subset(dataset, indices), sample_count


_PERM_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
_PERM_CACHE_MAX = 3


def _get_epoch_perm(dataset_len: int, seed: int, epoch: int) -> np.ndarray:
    key = (dataset_len, seed, epoch)
    perm = _PERM_CACHE.get(key)
    if perm is None:
        gen = torch.Generator()
        gen.manual_seed(seed + epoch)
        perm = torch.randperm(dataset_len, generator=gen).numpy()
        if len(_PERM_CACHE) >= _PERM_CACHE_MAX:
            _PERM_CACHE.pop(next(iter(_PERM_CACHE)))
        _PERM_CACHE[key] = perm
    return perm


def get_train_batch_indices(
    step: int,
    dataset_len: int,
    batch_size: int,
    world_size: int,
    seed: int,
) -> np.ndarray:
    global_batch = batch_size * world_size
    if dataset_len < global_batch:
        raise ValueError(
            f"Dataset size {dataset_len} is less than global batch {global_batch} "
            f"(batch_size={batch_size} x world_size={world_size})"
        )

    global_pos = step * global_batch
    epoch = global_pos // dataset_len
    offset = global_pos % dataset_len

    parts: list[np.ndarray] = []
    filled = 0
    while filled < global_batch:
        perm = _get_epoch_perm(dataset_len, seed, epoch)
        take = min(global_batch - filled, dataset_len - offset)
        parts.append(perm[offset : offset + take])
        filled += take
        offset += take
        if offset >= dataset_len:
            epoch += 1
            offset = 0
    indices = parts[0] if len(parts) == 1 else np.concatenate(parts)
    return np.ascontiguousarray(indices, dtype=np.int64)


def fetch_train_batch(
    dataset: TokenChunkDataset,
    step: int,
    batch_size: int,
    world_size: int,
    rank: int,
    seed: int,
) -> torch.Tensor:
    indices = get_train_batch_indices(step, len(dataset), batch_size, world_size, seed)
    start = rank * batch_size
    rank_indices = indices[start : start + batch_size]
    rows = [dataset[int(i)] for i in rank_indices]
    return collate_input_ids(rows)
