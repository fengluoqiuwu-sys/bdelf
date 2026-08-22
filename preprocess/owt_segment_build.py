"""OWT 句段预处理构建：按文档切分、pad、block shuffle、写 shard。"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Set

import numpy as np
from tqdm import tqdm

from dataset import FL_Dataset
from preprocess.owt_split import (
    bucket_counts_from_lengths,
    bucket_pad_length,
    chunk_document,
    pad_to,
)
from preprocess.preprocess import (
    FL_PreprocessConfig,
    _DTYPE,
    _ShardWriter,
    _SplitCacheMeta,
    _TaggedDocBatch,
    _cleanup_shuffle_tmp,
    _cleanup_split,
    _worker_count,
)
from tokenizer import FL_Tokenizer, get_token_layout, get_tokenizer

# 块 shuffle：每块连续读盘，块顺序随机；train 约 40GB 时通常比逐条随机读快一个数量级。
_SHUFFLE_BLOCK_ROWS = 65536

_WORKER_TOKENIZER: FL_Tokenizer | None = None
_WORKER_BOS: int = 0
_WORKER_EOS: int = 0
_WORKER_PAD: int = 0
_WORKER_PROCESS_D: int = 0
_WORKER_MIN_LEN: int = 128
_WORKER_PAD_MODE: str = "fixed"
_WORKER_FIXED_PAD: int = 0


@dataclass(frozen=True)
class _OwtWorkerConfig:
    tokenizer_name: str
    bos_id: int
    eos_id: int
    pad_id: int
    process_d: int
    min_chunk_len: int
    pad_mode: str
    fixed_pad_len: int


@dataclass(frozen=True)
class _ShuffleHooks:
    """shuffle 生命周期回调（由 preprocess 写 manifest）。"""

    on_start: Callable[[str, _SplitCacheMeta], None] | None = None
    on_progress: Callable[[str, int, int], None] | None = None
    on_complete: Callable[[str, _SplitCacheMeta], None] | None = None


def _init_owt_worker(cfg: _OwtWorkerConfig) -> None:
    global _WORKER_TOKENIZER, _WORKER_BOS, _WORKER_EOS, _WORKER_PAD
    global _WORKER_PROCESS_D, _WORKER_MIN_LEN, _WORKER_PAD_MODE, _WORKER_FIXED_PAD
    os.environ["BDELF_QUIET_TOKENIZER"] = "1"
    _WORKER_TOKENIZER = get_tokenizer(cfg.tokenizer_name)
    _WORKER_TOKENIZER.model_max_length = int(1e9)
    _WORKER_BOS = cfg.bos_id
    _WORKER_EOS = cfg.eos_id
    _WORKER_PAD = cfg.pad_id
    _WORKER_PROCESS_D = cfg.process_d
    _WORKER_MIN_LEN = cfg.min_chunk_len
    _WORKER_PAD_MODE = cfg.pad_mode
    _WORKER_FIXED_PAD = cfg.fixed_pad_len


def _pad_chunk_ids(chunk: List[int], pad_id: int) -> tuple[List[int], int]:
    valid_len = len(chunk)
    if _WORKER_PAD_MODE == "fixed":
        pad_len = _WORKER_FIXED_PAD
    else:
        pad_len = bucket_pad_length(valid_len)
        if pad_len is None:
            return [], 0
    if valid_len > pad_len:
        raise RuntimeError(f"chunk len {valid_len} > pad_len {pad_len}")
    return pad_to(chunk, pad_len, pad_id), valid_len


def _process_owt_batch(batch: _TaggedDocBatch) -> tuple[str, int, np.ndarray, np.ndarray]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("OWT worker tokenizer is not initialized.")
    rows: List[np.ndarray] = []
    lengths: List[int] = []
    for text in batch.texts:
        chunks = chunk_document(
            _WORKER_TOKENIZER,
            text,
            d=_WORKER_PROCESS_D,
            bos_id=_WORKER_BOS,
            eos_id=_WORKER_EOS,
            min_len=_WORKER_MIN_LEN,
        )
        for chunk in chunks:
            padded, valid = _pad_chunk_ids(chunk, _WORKER_PAD)
            if not padded:
                continue
            rows.append(np.asarray(padded, dtype=_DTYPE))
            lengths.append(valid)
    if not rows:
        empty = np.empty((0, _WORKER_FIXED_PAD), dtype=_DTYPE)
        return batch.split, len(batch.texts), empty, np.empty(0, dtype=_DTYPE)
    return (
        batch.split,
        len(batch.texts),
        np.stack(rows, axis=0),
        np.asarray(lengths, dtype=_DTYPE),
    )


def _iter_owt_pipelined(
    executor: ProcessPoolExecutor,
    doc_batches: Iterator[_TaggedDocBatch],
    *,
    workers: int,
) -> Iterator[tuple[str, int, np.ndarray, np.ndarray]]:
    max_inflight = max(2, workers * 2)
    inflight: deque[Future] = deque()

    for batch in doc_batches:
        inflight.append(executor.submit(_process_owt_batch, batch))
        while len(inflight) >= max_inflight:
            yield inflight.popleft().result()

    while inflight:
        yield inflight.popleft().result()


def _read_split_range(
    cache_dir: Path,
    split: str,
    meta: _SplitCacheMeta,
    chunk_length: int,
    start: int,
    end: int,
    *,
    shard_starts: np.ndarray,
    maps: List[np.memmap | None],
    len_mmap: np.memmap | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """读取 split 内全局行号 [start, end)（源顺序连续段，可跨 shard）。"""
    if start >= end:
        empty = np.empty((0, chunk_length), dtype=_DTYPE)
        return empty, np.empty(0, dtype=_DTYPE) if len_mmap is not None else None

    row_parts: List[np.ndarray] = []
    len_parts: List[np.ndarray] = []
    pos = start
    while pos < end:
        shard_idx = int(np.searchsorted(shard_starts, pos, side="right") - 1)
        shard = meta.shards[shard_idx]
        shard_global_end = int(shard_starts[shard_idx]) + shard.count
        chunk_end = min(end, shard_global_end)
        local_start = pos - int(shard_starts[shard_idx])
        local_end = chunk_end - int(shard_starts[shard_idx])

        mmap = maps[shard_idx]
        if mmap is None:
            mmap = np.memmap(
                cache_dir / shard.file,
                dtype=_DTYPE,
                mode="r",
                shape=(shard.count, chunk_length),
            )
            maps[shard_idx] = mmap
        row_parts.append(np.asarray(mmap[local_start:local_end]))
        if len_mmap is not None:
            len_parts.append(np.asarray(len_mmap[pos:chunk_end]))
        pos = chunk_end

    rows = row_parts[0] if len(row_parts) == 1 else np.concatenate(row_parts, axis=0)
    lengths = None
    if len_mmap is not None:
        lengths = (
            len_parts[0] if len(len_parts) == 1 else np.concatenate(len_parts, axis=0)
        )
    return rows, lengths


def _split_shuffle_seed(config: FL_PreprocessConfig, split: str) -> int:
    raw = f"{config.shuffle_seed}:{split}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:4], "big")


def _shuffle_split_cache_block(
    cache_dir: Path,
    split: str,
    meta: _SplitCacheMeta,
    *,
    chunk_length: int,
    shuffle_seed: int,
    bucket_lengths: List[int],
    hooks: _ShuffleHooks | None = None,
) -> _SplitCacheMeta:
    """块级 shuffle：打乱固定大小块顺序，块内保持源顺序（I/O 友好）。"""
    if meta.count <= 1:
        return meta

    rng = np.random.default_rng(shuffle_seed)
    block_size = _SHUFFLE_BLOCK_ROWS
    n_blocks = (meta.count + block_size - 1) // block_size
    block_perm = rng.permutation(n_blocks)

    shard_counts = np.asarray([s.count for s in meta.shards], dtype=np.int64)
    shard_starts = np.concatenate(([0], np.cumsum(shard_counts[:-1])))
    src_maps: List[np.memmap | None] = [None] * len(meta.shards)
    len_mmap: np.memmap | None = None
    if meta.has_lengths:
        len_path = cache_dir / f"{split}.len"
        len_mmap = np.memmap(
            len_path,
            dtype=_DTYPE,
            mode="r",
            shape=(meta.count,),
        )

    tmp_dir = cache_dir / f".shuffle_{split}"
    _cleanup_shuffle_tmp(cache_dir, split)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    writer = _ShardWriter(
        tmp_dir,
        split,
        chunk_length=chunk_length,
        record_lengths=meta.has_lengths,
    )

    if hooks and hooks.on_start:
        hooks.on_start(split, meta)

    progress = tqdm(
        total=meta.count,
        desc=f"[preprocess] shuffle {split}",
        unit="chunk",
        dynamic_ncols=True,
    )
    progress_rows = 0
    manifest_rows = 0
    manifest_stride = 1_048_576

    try:
        for block_id in block_perm:
            src_start = int(block_id) * block_size
            src_end = min(src_start + block_size, meta.count)
            rows, lengths = _read_split_range(
                cache_dir,
                split,
                meta,
                chunk_length,
                src_start,
                src_end,
                shard_starts=shard_starts,
                maps=src_maps,
                len_mmap=len_mmap,
            )
            writer.append(rows, lengths)
            block_rows = src_end - src_start
            progress.update(block_rows)
            progress_rows += block_rows

            if (
                hooks
                and hooks.on_progress
                and progress_rows - manifest_rows >= manifest_stride
            ):
                hooks.on_progress(split, progress_rows, meta.count)
                manifest_rows = progress_rows
    finally:
        progress.close()

    new_meta = writer.finalize()
    _cleanup_split(cache_dir, split)
    for path in tmp_dir.iterdir():
        path.rename(cache_dir / path.name)
    tmp_dir.rmdir()

    if bucket_lengths and new_meta.has_lengths:
        out_len = np.memmap(
            cache_dir / f"{split}.len",
            dtype=_DTYPE,
            mode="r",
            shape=(new_meta.count,),
        )
        new_meta.bucket_counts = bucket_counts_from_lengths(
            out_len.tolist(), bucket_lengths
        )

    if hooks and hooks.on_complete:
        hooks.on_complete(split, new_meta)

    return new_meta


def shuffle_owt_splits(
    cache_dir: Path,
    config: FL_PreprocessConfig,
    metas: Dict[str, _SplitCacheMeta],
    *,
    hooks: _ShuffleHooks | None = None,
) -> Dict[str, _SplitCacheMeta]:
    """对已切分（未 shuffle 或 shuffle 中断）的 split 做 block shuffle。"""
    out: Dict[str, _SplitCacheMeta] = {}
    for split in sorted(metas):
        meta = metas[split]
        if meta.count <= 1:
            out[split] = meta
            if hooks and hooks.on_complete:
                hooks.on_complete(split, meta)
            continue
        tqdm.write(
            f"[preprocess] block shuffle split={split!r}: "
            f"{meta.count:,} chunks, block={_SHUFFLE_BLOCK_ROWS:,}"
        )
        out[split] = _shuffle_split_cache_block(
            cache_dir,
            split,
            meta,
            chunk_length=config.chunk_length,
            shuffle_seed=_split_shuffle_seed(config, split),
            bucket_lengths=list(config.bucket_lengths),
            hooks=hooks,
        )
        tqdm.write(
            f"[preprocess] split={split!r}: done, {out[split].count:,} chunks"
        )
    return out


def _run_owt_segment_loop(
    tagged_batches: Iterator[_TaggedDocBatch],
    *,
    splits: Set[str],
    cache_dir: Path,
    config: FL_PreprocessConfig,
    doc_totals: Dict[str, int],
    workers: int,
    worker_cfg: _OwtWorkerConfig,
) -> Dict[str, _SplitCacheMeta]:
    writers = {
        split: _ShardWriter(
            cache_dir,
            split,
            chunk_length=config.chunk_length,
            record_lengths=True,
        )
        for split in splits
    }
    progress: Dict[str, tqdm] = {}
    for split, total in doc_totals.items():
        progress[split] = tqdm(
            total=total,
            desc=f"[preprocess] {split}",
            unit="doc",
            dynamic_ncols=True,
        )

    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_owt_worker,
            initargs=(worker_cfg,),
        ) as executor:
            for split, doc_count, rows, lengths in _iter_owt_pipelined(
                executor, tagged_batches, workers=workers
            ):
                if rows.size > 0:
                    writers[split].append(rows, lengths)
                bar = progress[split]
                bar.update(doc_count)
                bar.set_postfix(
                    chunks=f"{writers[split]._total:,}", refresh=False
                )
    finally:
        metas: Dict[str, _SplitCacheMeta] = {}
        for split, writer in writers.items():
            progress[split].close()
            metas[split] = writer.finalize()
            tqdm.write(
                f"[preprocess] split={split!r}: built, {metas[split].count:,} chunks "
                f"(shuffle pending)"
            )
        return metas


def build_owt_segment_splits(
    _source: FL_Dataset | None,
    *,
    splits: Set[str],
    cache_dir: Path,
    config: FL_PreprocessConfig,
    tagged_batches: Iterator[_TaggedDocBatch],
    doc_totals: Dict[str, int],
) -> Dict[str, _SplitCacheMeta]:
    layout = get_token_layout(config.tokenizer)
    workers = _worker_count()
    worker_cfg = _OwtWorkerConfig(
        tokenizer_name=config.tokenizer,
        bos_id=layout.bos_token_id,
        eos_id=layout.eos_token_id,
        pad_id=layout.pad_token_id,
        process_d=config.process_d,
        min_chunk_len=config.min_chunk_len,
        pad_mode=config.pad_mode,
        fixed_pad_len=config.chunk_length,
    )
    tqdm.write(
        f"[preprocess] owt_segment: splits={sorted(splits)}, "
        f"process_d={config.process_d}, pad_mode={config.pad_mode!r}, "
        f"chunk_length={config.chunk_length}, workers={workers}"
    )
    return _run_owt_segment_loop(
        tagged_batches,
        splits=splits,
        cache_dir=cache_dir,
        config=config,
        doc_totals=doc_totals,
        workers=workers,
        worker_cfg=worker_cfg,
    )
