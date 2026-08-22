"""Preprocessing pipeline and preprocessed datasets for language-model training."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Set, Union

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from config_util import load_yaml_config
from dataset import FL_Dataset, get_dataset
from tokenizer import FL_TokenLayout, FL_Tokenizer, get_token_layout, get_tokenizer

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "preprocess"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "preprocessed_datasets"

OverflowMode = Literal["wrap", "discard", "pad_eos"]
PadMode = Literal["fixed", "bucket"]
Strategy = Literal["stream", "owt_segment"]
_MANIFEST_VERSION_STREAM = 2
_MANIFEST_VERSION_OWT = 3
_OVERFLOW_MODES = frozenset({"wrap", "discard", "pad_eos"})
_DTYPE = np.int32
_DOCS_PER_TASK = 512
# Split token/chunk storage into multiple files above this size (bytes).
_SHARD_MAX_BYTES = 1 << 30

_WORKER_TOKENIZER: FL_Tokenizer | None = None


@dataclass(frozen=True)
class _TaggedDocBatch:
    split: str
    texts: List[str]


@dataclass
class _SplitPipeline:
    split: str
    chunker: "_StreamingChunker"
    writer: "_ShardWriter"


@dataclass
class FL_PreprocessConfig:
    """Abstract preprocessing config."""

    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "chunk_length",
            "overflow_mode",
            "seed",
            "text_column",
        }
    )

    name: str = "prototype"
    tokenizer: str = "gpt2"
    strategy: Strategy = "stream"
    chunk_length: int = 1024
    overflow_mode: OverflowMode = "discard"
    seed: int = 42
    text_column: str = "text"
    process_d: int = 0
    min_chunk_len: int = 128
    pad_mode: PadMode = "fixed"
    bucket_lengths: List[int] = field(default_factory=list)
    shuffle_seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "FL_PreprocessConfig":
        config = load_yaml_config(cls, path, required=cls._YAML_REQUIRED)
        if config.overflow_mode not in _OVERFLOW_MODES:
            raise ValueError(
                f"{path}: overflow_mode must be one of "
                f"{sorted(_OVERFLOW_MODES)}, got {config.overflow_mode!r}"
            )
        if config.chunk_length < 2:
            raise ValueError(f"{path}: chunk_length must be >= 2")
        if config.strategy not in ("stream", "owt_segment"):
            raise ValueError(
                f"{path}: strategy must be 'stream' or 'owt_segment', "
                f"got {config.strategy!r}"
            )
        if config.strategy == "owt_segment":
            if config.process_d < 512:
                raise ValueError(f"{path}: process_d must be >= 512")
            if config.pad_mode not in ("fixed", "bucket"):
                raise ValueError(f"{path}: pad_mode must be 'fixed' or 'bucket'")
            if config.pad_mode == "bucket" and not config.bucket_lengths:
                config.bucket_lengths = [256, 512, 1024, 2048]
        return config

    def manifest_version(self) -> int:
        return (
            _MANIFEST_VERSION_OWT
            if self.strategy == "owt_segment"
            else _MANIFEST_VERSION_STREAM
        )


@dataclass(frozen=True)
class _SplitShardMeta:
    file: str
    count: int


@dataclass
class _SplitCacheMeta:
    count: int
    shards: List[_SplitShardMeta]
    has_lengths: bool = False
    bucket_counts: Dict[int, int] = field(default_factory=dict)


def list_preprocess() -> List[str]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(
        path.stem
        for path in CONFIG_DIR.glob("*.yaml")
        if path.stem != "prototype"
    )


def get_preprocess(name: str) -> FL_PreprocessConfig:
    if name == "prototype":
        raise ValueError("Prototype preprocess config cannot be instantiated.")

    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        available = ", ".join(list_preprocess()) or "<none>"
        raise FileNotFoundError(
            f"Config {name}.yaml does not exist. Available: {available}"
        )
    return FL_PreprocessConfig.from_yaml(config_path)


def get_preprocessed(
    preprocess_name: str,
    dataset: Union[str, FL_Dataset],
) -> "FL_PreprocessedDataset":
    source = get_dataset(dataset) if isinstance(dataset, str) else dataset
    return FL_PreprocessedDataset(get_preprocess(preprocess_name), source)


def _dataset_fingerprint_payload(dc) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "name": dc.name,
        "repo_id": dc.repo_id,
        "revision": dc.revision,
        "subset": dc.subset,
        "split": dc.split,
    }
    if dc.dev_count is not None and dc.test_count is not None:
        payload["dev_count"] = dc.dev_count
        payload["test_count"] = dc.test_count
        payload["holdout_seed"] = dc.holdout_seed
    else:
        payload["eval_count"] = dc.eval_count
        payload["eval_seed"] = dc.eval_seed
    return payload


def _preprocess_fingerprint_payload(config: FL_PreprocessConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "name": config.name,
        "tokenizer": config.tokenizer,
        "strategy": config.strategy,
        "chunk_length": config.chunk_length,
        "seed": config.seed,
        "text_column": config.text_column,
    }
    if config.strategy == "stream":
        payload["overflow_mode"] = config.overflow_mode
    else:
        payload.update(
            {
                "process_d": config.process_d,
                "min_chunk_len": config.min_chunk_len,
                "pad_mode": config.pad_mode,
                "bucket_lengths": list(config.bucket_lengths),
                "shuffle_seed": config.shuffle_seed,
            }
        )
    return payload


def _fingerprint(config: FL_PreprocessConfig, source: FL_Dataset) -> str:
    dc = source.config
    payload = {
        "preprocess": _preprocess_fingerprint_payload(config),
        "dataset": _dataset_fingerprint_payload(dc),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_dir(config: FL_PreprocessConfig, source: FL_Dataset) -> Path:
    return CACHE_DIR / f"{source.config.name}_{config.name}_{_fingerprint(config, source)}"



def _available_cpu_count() -> int:
    """可见 CPU 数（Slurm/cgroup 下为分配核数，而非整机逻辑核）。"""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _worker_count() -> int:
    n = _available_cpu_count()
    # 主进程顺序读 parquet + 协调；留 1 核避免与 worker 抢满 CPU
    return max(1, n - 1) if n > 1 else 1


def _log_preprocess(message: str) -> None:
    tqdm.write(f"[preprocess] {message}")


def _init_tokenizer_worker(tokenizer_name: str) -> None:
    global _WORKER_TOKENIZER
    os.environ["BDELF_QUIET_TOKENIZER"] = "1"
    _WORKER_TOKENIZER = get_tokenizer(tokenizer_name)
    # Tokenize full documents first, then chunk by token length; ignore model_max_length here.
    _WORKER_TOKENIZER.model_max_length = int(1e9)


def _encode_texts(texts: List[str], tokenizer: FL_Tokenizer) -> np.ndarray:
    """Tokenize full documents, return a concatenated token stream for chunking."""
    if not texts:
        return np.empty(0, dtype=_DTYPE)

    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    parts: List[np.ndarray] = []
    for token_ids in encoded:
        if token_ids:
            parts.append(np.asarray(token_ids, dtype=_DTYPE))
    if not parts:
        return np.empty(0, dtype=_DTYPE)
    return np.concatenate(parts)


def _shard_capacity(chunk_length: int) -> int:
    row_bytes = chunk_length * np.dtype(_DTYPE).itemsize
    return max(1, _SHARD_MAX_BYTES // row_bytes)


def _tokenize_texts_shard(texts: List[str]) -> np.ndarray:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker is not initialized.")
    return _encode_texts(texts, _WORKER_TOKENIZER)


def _iter_doc_batches(
    texts: Iterator[str],
    *,
    batch_size: int,
) -> Iterator[List[str]]:
    batch: List[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_doc_batches_from_dataset(
    hf_dataset,
    text_column: str,
) -> Iterator[List[str]]:
    def _rows() -> Iterator[str]:
        for row in hf_dataset:
            text = row.get(text_column) if isinstance(row, dict) else row[text_column]
            if text is None:
                continue
            stripped = str(text).strip()
            if stripped:
                yield stripped

    yield from _iter_doc_batches(_rows(), batch_size=_DOCS_PER_TASK)


def _iter_tagged_doc_batches(
    source: FL_Dataset,
    splits: Set[str],
    *,
    text_column: str,
    total_rows: int,
) -> Iterator[_TaggedDocBatch]:
    pending: Dict[str, List[str]] = {split: [] for split in splits}

    with tqdm(
        total=total_rows,
        desc="[preprocess] reading",
        unit="row",
        dynamic_ncols=True,
    ) as row_progress:
        for row_index, text in source.iter_parquet_rows(text_column=text_column):
            row_progress.update(1)
            if text is None:
                continue
            split = source.holdout_row_split(row_index)
            if split not in splits:
                continue
            pending[split].append(text)
            if len(pending[split]) >= _DOCS_PER_TASK:
                yield _TaggedDocBatch(split, pending[split])
                pending[split] = []

    for split, texts in pending.items():
        if texts:
            yield _TaggedDocBatch(split, texts)


def _tokenize_tagged_batch(batch: _TaggedDocBatch) -> tuple[str, int, np.ndarray]:
    return batch.split, len(batch.texts), _tokenize_texts_shard(batch.texts)


def _iter_token_streams_pipelined(
    executor: ProcessPoolExecutor,
    doc_batches: Iterator[_TaggedDocBatch],
    *,
    workers: int,
) -> Iterator[tuple[str, int, np.ndarray]]:
    """Sequential submit + bounded in-flight tokenize."""
    max_inflight = max(2, workers * 2)
    inflight: deque[Future] = deque()

    for batch in doc_batches:
        inflight.append(executor.submit(_tokenize_tagged_batch, batch))
        while len(inflight) >= max_inflight:
            split, doc_count, tokens = inflight.popleft().result()
            yield split, doc_count, tokens

    while inflight:
        split, doc_count, tokens = inflight.popleft().result()
        yield split, doc_count, tokens


def _split_doc_total(source: FL_Dataset, split: str) -> int:
    total = source.count_raw_rows()
    if source.config.dev_count is not None and source.config.test_count is not None:
        sets = source._get_tri_holdout_sets()
        if split == "train":
            return total - len(sets["test"]) - len(sets["dev"])
        if split == "dev":
            return len(sets["dev"])
        if split == "test":
            return len(sets["test"])
        raise ValueError(f"Unknown split '{split}'.")
    eval_count = len(source.holdout_eval_indices())
    if split == "train":
        return total - eval_count
    if split == "eval":
        return eval_count
    raise ValueError(f"Unknown split '{split}'.")


def _run_preprocess_loop(
    doc_batches: Iterator[List[str]],
    *,
    total: int,
    split: str,
    cache_dir: Path,
    config: FL_PreprocessConfig,
    special: FL_TokenLayout,
    workers: int,
    tokenizer_name: str,
) -> _SplitCacheMeta:
    def _tagged() -> Iterator[_TaggedDocBatch]:
        for texts in doc_batches:
            yield _TaggedDocBatch(split, texts)

    pipelines = {
        split: _make_split_pipeline(
            split, cache_dir=cache_dir, config=config, special=special
        )
    }
    metas = _run_tagged_preprocess_loop(
        _tagged(),
        pipelines=pipelines,
        doc_totals={split: total},
        workers=workers,
        tokenizer_name=tokenizer_name,
    )
    return metas[split]


def _make_split_pipeline(
    split: str,
    *,
    cache_dir: Path,
    config: FL_PreprocessConfig,
    special: FL_TokenLayout,
) -> _SplitPipeline:
    return _SplitPipeline(
        split=split,
        chunker=_StreamingChunker(
            chunk_length=config.chunk_length,
            overflow_mode=config.overflow_mode,
            special=special,
        ),
        writer=_ShardWriter(
            cache_dir,
            split,
            chunk_length=config.chunk_length,
            record_lengths=config.overflow_mode == "pad_eos",
        ),
    )


def _run_tagged_preprocess_loop(
    tagged_batches: Iterator[_TaggedDocBatch],
    *,
    pipelines: Dict[str, _SplitPipeline],
    doc_totals: Dict[str, int],
    workers: int,
    tokenizer_name: str,
) -> Dict[str, _SplitCacheMeta]:
    progress: Dict[str, tqdm] = {}
    metas: Dict[str, _SplitCacheMeta] = {}
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
            initializer=_init_tokenizer_worker,
            initargs=(tokenizer_name,),
        ) as executor:
            for split, doc_count, tokens in _iter_token_streams_pipelined(
                executor, tagged_batches, workers=workers
            ):
                pipeline = pipelines[split]
                rows, lengths = pipeline.chunker.feed(tokens)
                pipeline.writer.append(rows, lengths)
                bar = progress[split]
                bar.update(doc_count)
                bar.set_postfix(chunks=f"{pipeline.writer._total:,}", refresh=False)
    finally:
        for split, pipeline in pipelines.items():
            rows, lengths = pipeline.chunker.finish()
            pipeline.writer.append(rows, lengths)
            progress[split].close()
            meta = pipeline.writer.finalize()
            metas[split] = meta
            tqdm.write(f"[preprocess] split={split!r}: done, {meta.count:,} chunks")
    return metas


def _stream_preprocess_parquet(
    source: FL_Dataset,
    *,
    splits: Set[str],
    cache_dir: Path,
    config: FL_PreprocessConfig,
    special: FL_TokenLayout,
    workers: int,
) -> Dict[str, _SplitCacheMeta]:
    total_rows = source.count_raw_rows()
    doc_totals = {
        split: _split_doc_total(source, split) for split in sorted(splits)
    }
    tqdm.write(
        f"[preprocess] single parquet scan: rows={total_rows:,}, "
        f"splits={sorted(splits)}, workers={workers}, task={_DOCS_PER_TASK}, "
        f"chunk={config.chunk_length} tokens"
    )
    pipelines = {
        split: _make_split_pipeline(
            split, cache_dir=cache_dir, config=config, special=special
        )
        for split in splits
    }
    tagged_batches = _iter_tagged_doc_batches(
        source, splits, text_column=config.text_column, total_rows=total_rows
    )
    return _run_tagged_preprocess_loop(
        tagged_batches,
        pipelines=pipelines,
        doc_totals=doc_totals,
        workers=workers,
        tokenizer_name=config.tokenizer,
    )


def _stream_preprocess_split_dataset(
    hf_dataset,
    *,
    split: str,
    cache_dir: Path,
    config: FL_PreprocessConfig,
    special: FL_TokenLayout,
    workers: int,
) -> _SplitCacheMeta:
    total = len(hf_dataset)
    doc_batches = _iter_doc_batches_from_dataset(hf_dataset, config.text_column)
    tqdm.write(
        f"[preprocess] split={split!r}: {total:,} texts "
        f"(workers={workers}, task={_DOCS_PER_TASK}, "
        f"chunk={config.chunk_length} tokens)"
    )
    return _run_preprocess_loop(
        doc_batches,
        total=total,
        split=split,
        cache_dir=cache_dir,
        config=config,
        special=special,
        workers=workers,
        tokenizer_name=config.tokenizer,
    )


class _StreamingChunker:
    """Split a token stream into fixed-width rows after full-document tokenization."""

    def __init__(
        self,
        *,
        chunk_length: int,
        overflow_mode: OverflowMode,
        special: FL_TokenLayout,
    ) -> None:
        self.chunk_length = chunk_length
        self.overflow_mode = overflow_mode
        self.special = special
        self.content = chunk_length - 1
        self._buffer = np.empty(0, dtype=_DTYPE)
        self._stream_prefix = np.empty(0, dtype=_DTYPE)

    def _track_stream_prefix(self, tokens: np.ndarray) -> None:
        if tokens.size == 0 or self._stream_prefix.size >= self.content:
            return
        take = min(self.content - self._stream_prefix.size, tokens.size)
        self._stream_prefix = np.concatenate([self._stream_prefix, tokens[:take]])

    def feed(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if tokens.size == 0:
            return np.empty((0, self.chunk_length), dtype=_DTYPE), None

        self._track_stream_prefix(tokens)
        self._buffer = np.concatenate((self._buffer, tokens))
        chunks: List[np.ndarray] = []
        while self._buffer.size >= self.content:
            row = np.empty(self.chunk_length, dtype=_DTYPE)
            row[0] = self.special.bos_token_id
            row[1:] = self._buffer[: self.content]
            chunks.append(row)
            self._buffer = self._buffer[self.content :]

        if not chunks:
            return np.empty((0, self.chunk_length), dtype=_DTYPE), None
        return np.stack(chunks, axis=0), None

    def finish(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self._buffer.size == 0:
            return np.empty((0, self.chunk_length), dtype=_DTYPE), None

        if self.overflow_mode == "discard":
            return np.empty((0, self.chunk_length), dtype=_DTYPE), None

        row = np.empty(self.chunk_length, dtype=_DTYPE)

        if self.overflow_mode == "wrap":
            need = self.content - self._buffer.size
            wrap = _cyclic_take(self._stream_prefix, need)
            body = np.concatenate((self._buffer, wrap))
            row[0] = self.special.bos_token_id
            row[1:] = body
            return row.reshape(1, self.chunk_length), None

        if self.overflow_mode == "pad_eos":
            body = np.concatenate(
                ([self.special.bos_token_id], self._buffer, [self.special.eos_token_id])
            ).astype(_DTYPE)
            valid = min(body.size, self.chunk_length)
            row.fill(self.special.pad_token_id)
            row[:valid] = body[:valid]
            return row.reshape(1, self.chunk_length), np.asarray([valid], dtype=_DTYPE)

        raise ValueError(f"Unknown overflow_mode: {self.overflow_mode!r}")


def _cyclic_take(source: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=_DTYPE)
    if source.size == 0:
        return np.zeros(count, dtype=_DTYPE)
    reps = (count + source.size - 1) // source.size
    return np.tile(source, reps)[:count]


class _ShardWriter:
    """Write chunk rows into one or more shard files."""

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        *,
        chunk_length: int,
        record_lengths: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.split = split
        self.chunk_length = chunk_length
        self._record_lengths = record_lengths
        self._shard_capacity = _shard_capacity(chunk_length)
        self._shard_idx = 0
        self._shard_rows = 0
        self._mmap: np.memmap | None = None
        self._lengths: List[int] = []
        self._shards: List[_SplitShardMeta] = []
        self._total = 0

    def _open_shard(self) -> None:
        if self._mmap is not None:
            self._mmap.flush()
        path = self.cache_dir / f"{self.split}.{self._shard_idx:05d}.bin"
        self._mmap = np.memmap(
            path,
            dtype=_DTYPE,
            mode="w+",
            shape=(self._shard_capacity, self.chunk_length),
        )
        self._shard_rows = 0
        self._shards.append(
            _SplitShardMeta(file=path.name, count=0)
        )

    def append(
        self,
        rows: np.ndarray,
        lengths: np.ndarray | None,
    ) -> None:
        if rows.size == 0:
            return

        offset = 0
        while offset < rows.shape[0]:
            if self._mmap is None or self._shard_rows >= self._shard_capacity:
                if self._mmap is not None:
                    self._shards[-1] = _SplitShardMeta(
                        file=self._shards[-1].file,
                        count=self._shard_rows,
                    )
                    self._shard_idx += 1
                self._open_shard()

            take = min(rows.shape[0] - offset, self._shard_capacity - self._shard_rows)
            end = offset + take
            self._mmap[self._shard_rows : self._shard_rows + take] = rows[offset:end]
            self._shard_rows += take
            self._total += take

            if self._record_lengths:
                if lengths is None:
                    self._lengths.extend([self.chunk_length] * take)
                else:
                    self._lengths.extend(int(v) for v in lengths[offset:end])

            offset = end

    def finalize(self) -> _SplitCacheMeta:
        if self._mmap is None:
            return _SplitCacheMeta(count=0, shards=[], has_lengths=False)

        self._shards[-1] = _SplitShardMeta(
            file=self._shards[-1].file,
            count=self._shard_rows,
        )
        self._mmap.flush()
        del self._mmap
        self._mmap = None

        for shard in self._shards:
            path = self.cache_dir / shard.file
            if shard.count == 0:
                path.unlink(missing_ok=True)
                continue
            row_bytes = self.chunk_length * np.dtype(_DTYPE).itemsize
            with open(path, "rb+") as f:
                f.truncate(shard.count * row_bytes)
            np.memmap(
                path,
                dtype=_DTYPE,
                mode="r+",
                shape=(shard.count, self.chunk_length),
            ).flush()

        has_lengths = self._record_lengths and len(self._lengths) == self._total
        if has_lengths:
            len_path = self.cache_dir / f"{self.split}.len"
            len_mmap = np.memmap(
                len_path,
                dtype=_DTYPE,
                mode="w+",
                shape=(self._total,),
            )
            len_mmap[:] = np.asarray(self._lengths, dtype=_DTYPE)
            len_mmap.flush()

        return _SplitCacheMeta(
            count=self._total,
            shards=[s for s in self._shards if s.count > 0],
            has_lengths=has_lengths,
        )


def _cleanup_split(cache_dir: Path, split: str) -> None:
    for path in cache_dir.glob(f"{split}.*"):
        path.unlink(missing_ok=True)
    (cache_dir / f"{split}.len").unlink(missing_ok=True)


def _cleanup_shuffle_tmp(cache_dir: Path, split: str) -> None:
    tmp_dir = cache_dir / f".shuffle_{split}"
    if not tmp_dir.exists():
        return
    for child in tmp_dir.iterdir():
        child.unlink(missing_ok=True)
    tmp_dir.rmdir()


def _infer_split_meta_from_disk(
    cache_dir: Path,
    split: str,
    *,
    chunk_length: int,
) -> _SplitCacheMeta | None:
    """无 manifest 时从磁盘推断 split 元数据（用于切分已完成、shuffle 未完成）。"""
    shards: List[_SplitShardMeta] = []
    row_bytes = chunk_length * np.dtype(_DTYPE).itemsize
    for path in sorted(cache_dir.glob(f"{split}.*.bin")):
        size = path.stat().st_size
        if size <= 0 or size % row_bytes != 0:
            return None
        shards.append(_SplitShardMeta(file=path.name, count=size // row_bytes))
    if not shards:
        return None
    count = sum(shard.count for shard in shards)
    len_path = cache_dir / f"{split}.len"
    has_lengths = (
        len_path.exists() and len_path.stat().st_size == count * np.dtype(_DTYPE).itemsize
    )
    return _SplitCacheMeta(count=count, shards=shards, has_lengths=has_lengths)


def _split_meta_from_manifest(raw: Dict[str, Any]) -> _SplitCacheMeta:
    shards = [
        _SplitShardMeta(file=item["file"], count=int(item["count"]))
        for item in raw.get("shards", [])
    ]
    bucket_raw = raw.get("bucket_counts") or {}
    bucket_counts = {int(k): int(v) for k, v in bucket_raw.items()}
    return _SplitCacheMeta(
        count=int(raw.get("count", 0)),
        shards=shards,
        has_lengths=bool(raw.get("has_lengths", False)),
        bucket_counts=bucket_counts,
    )


def _split_meta_to_manifest(meta: _SplitCacheMeta) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "count": meta.count,
        "has_lengths": meta.has_lengths,
        "shards": [{"file": s.file, "count": s.count} for s in meta.shards],
    }
    if meta.bucket_counts:
        out["bucket_counts"] = dict(meta.bucket_counts)
    return out


def _manifest_payload_base(
    config: FL_PreprocessConfig,
    fingerprint: str,
    *,
    status: str,
    split_counts: Dict[str, int],
    splits: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": config.manifest_version(),
        "status": status,
        "fingerprint": fingerprint,
        "strategy": config.strategy,
        "chunk_length": config.chunk_length,
        "split_counts": split_counts,
        "splits": splits,
    }
    if config.strategy == "stream":
        payload["overflow_mode"] = config.overflow_mode
    else:
        payload.update(
            {
                "process_d": config.process_d,
                "min_chunk_len": config.min_chunk_len,
                "pad_mode": config.pad_mode,
                "bucket_lengths": list(config.bucket_lengths),
                "shuffle_seed": config.shuffle_seed,
                "shuffle_mode": "block",
                "shuffle_block_rows": 65536,
            }
        )
    return payload


def _verify_split_cache(
    cache_dir: Path,
    split: str,
    meta: _SplitCacheMeta,
    *,
    chunk_length: int | None = None,
) -> bool:
    if meta.count != sum(shard.count for shard in meta.shards):
        return False
    if meta.count == 0:
        return not meta.shards and not meta.has_lengths
    row_bytes_expected = None
    for shard in meta.shards:
        path = cache_dir / shard.file
        if not path.exists() or shard.count <= 0:
            return False
        size = path.stat().st_size
        row_bytes = size // shard.count
        if size != row_bytes * shard.count:
            return False
        if row_bytes_expected is None:
            row_bytes_expected = row_bytes
        elif row_bytes != row_bytes_expected:
            return False
    if chunk_length is not None and row_bytes_expected is not None:
        expected = chunk_length * np.dtype(_DTYPE).itemsize
        if row_bytes_expected != expected:
            return False
    if meta.has_lengths:
        len_path = cache_dir / f"{split}.len"
        if not len_path.exists():
            return False
        if len_path.stat().st_size != meta.count * np.dtype(_DTYPE).itemsize:
            return False
    return True


def _write_manifest(cache_dir: Path, payload: Dict[str, Any]) -> None:
    tmp = cache_dir / "manifest.yaml.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    os.replace(tmp, cache_dir / "manifest.yaml")


def _cleanup_cache_dir(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    for path in cache_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir() and path.name.startswith(".shuffle_"):
            for child in path.iterdir():
                child.unlink(missing_ok=True)
            path.rmdir()


def _owt_segment_preprocess_parquet(
    source: FL_Dataset,
    *,
    splits: Set[str],
    cache_dir: Path,
    config: FL_PreprocessConfig,
) -> Dict[str, _SplitCacheMeta]:
    from preprocess.owt_segment_build import build_owt_segment_splits

    total_rows = source.count_raw_rows()
    doc_totals = {
        split: _split_doc_total(source, split) for split in sorted(splits)
    }
    tagged_batches = _iter_tagged_doc_batches(
        source, splits, text_column=config.text_column, total_rows=total_rows
    )
    return build_owt_segment_splits(
        source,
        splits=splits,
        cache_dir=cache_dir,
        config=config,
        tagged_batches=tagged_batches,
        doc_totals=doc_totals,
    )


def _owt_segment_preprocess_split_dataset(
    hf_dataset,
    *,
    split: str,
    cache_dir: Path,
    config: FL_PreprocessConfig,
) -> _SplitCacheMeta:
    from preprocess.owt_segment_build import build_owt_segment_splits

    total = len(hf_dataset)

    def _tagged() -> Iterator[_TaggedDocBatch]:
        for texts in _iter_doc_batches_from_dataset(hf_dataset, config.text_column):
            yield _TaggedDocBatch(split, texts)

    metas = build_owt_segment_splits(
        None,
        splits={split},
        cache_dir=cache_dir,
        config=config,
        tagged_batches=_tagged(),
        doc_totals={split: total},
    )
    return metas[split]


def _owt_shuffle_splits(
    cache_dir: Path,
    config: FL_PreprocessConfig,
    split_names: List[str],
    split_entries: Dict[str, Any],
    split_counts: Dict[str, int],
    *,
    write_partial: Callable[[], None],
) -> None:
    from preprocess.owt_segment_build import _ShuffleHooks, shuffle_owt_splits

    metas = {
        split: _split_meta_from_manifest(split_entries[split]) for split in split_names
    }

    def _on_start(split: str, meta: _SplitCacheMeta) -> None:
        split_entries[split] = {
            "status": "shuffling",
            "shuffle_rows": 0,
            **_split_meta_to_manifest(meta),
        }
        write_partial()

    def _on_progress(split: str, rows_done: int, total: int) -> None:
        entry = split_entries.get(split, {})
        entry["status"] = "shuffling"
        entry["shuffle_rows"] = rows_done
        split_entries[split] = entry
        write_partial()

    def _on_complete(split: str, meta: _SplitCacheMeta) -> None:
        split_entries[split] = {
            "status": "complete",
            **_split_meta_to_manifest(meta),
        }
        split_counts[split] = meta.count
        write_partial()

    hooks = _ShuffleHooks(
        on_start=_on_start,
        on_progress=_on_progress,
        on_complete=_on_complete,
    )
    shuffle_owt_splits(cache_dir, config, metas, hooks=hooks)


def _split_needs_shuffle(
    cache_dir: Path,
    split: str,
    prior: Dict[str, Any] | None,
    *,
    chunk_length: int,
) -> _SplitCacheMeta | None:
    """若 split 已切分未 shuffle（或 shuffle 中断），返回可验证的 meta。"""
    if prior:
        status = prior.get("status")
        if status == "complete":
            meta = _split_meta_from_manifest(prior)
            if _verify_split_cache(
                cache_dir, split, meta, chunk_length=chunk_length
            ):
                return None
        elif status in ("built", "shuffling"):
            meta = _split_meta_from_manifest(prior)
            if _verify_split_cache(
                cache_dir, split, meta, chunk_length=chunk_length
            ):
                return meta

    inferred = _infer_split_meta_from_disk(
        cache_dir, split, chunk_length=chunk_length
    )
    if inferred and _verify_split_cache(
        cache_dir, split, inferred, chunk_length=chunk_length
    ):
        return inferred
    return None


def _build_cache(
    config: FL_PreprocessConfig,
    source: FL_Dataset,
    cache_dir: Path,
) -> Dict[str, int]:
    special = get_token_layout(config.tokenizer)
    workers = _worker_count()
    fingerprint = _fingerprint(config, source)
    expected_version = config.manifest_version()
    manifest = _load_manifest(cache_dir)
    if manifest and manifest.get("fingerprint") != fingerprint:
        _cleanup_cache_dir(cache_dir)
    elif manifest and manifest.get("version") != expected_version:
        _cleanup_cache_dir(cache_dir)

    existing = _load_manifest(cache_dir) or {}
    split_entries: Dict[str, Any] = dict(existing.get("splits", {}))
    split_counts: Dict[str, int] = {}
    splits_to_build: List[str] = []
    splits_to_shuffle: List[str] = []
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split in source.get_splits():
        prior = split_entries.get(split)
        if prior and prior.get("status") == "complete":
            meta = _split_meta_from_manifest(prior)
            if _verify_split_cache(
                cache_dir, split, meta, chunk_length=config.chunk_length
            ):
                split_counts[split] = meta.count
                continue

        shuffle_meta = _split_needs_shuffle(
            cache_dir,
            split,
            prior,
            chunk_length=config.chunk_length,
        )
        if shuffle_meta is not None:
            split_entries[split] = {
                "status": "built",
                **_split_meta_to_manifest(shuffle_meta),
            }
            split_counts[split] = shuffle_meta.count
            splits_to_shuffle.append(split)
            continue

        _cleanup_split(cache_dir, split)
        _cleanup_shuffle_tmp(cache_dir, split)
        splits_to_build.append(split)

    def _write_partial() -> None:
        _write_manifest(
            cache_dir,
            _manifest_payload_base(
                config,
                fingerprint,
                status="partial",
                split_counts=dict(split_counts),
                splits=split_entries,
            ),
        )

    if splits_to_shuffle and config.strategy == "owt_segment":
        _log_preprocess(
            f"Skipping tokenize; resuming shuffle for splits={splits_to_shuffle}"
        )
        _owt_shuffle_splits(
            cache_dir,
            config,
            splits_to_shuffle,
            split_entries,
            split_counts,
            write_partial=_write_partial,
        )

    if splits_to_build and source.can_stream_parquet():
        if config.strategy == "owt_segment":
            metas = _owt_segment_preprocess_parquet(
                source,
                splits=set(splits_to_build),
                cache_dir=cache_dir,
                config=config,
            )
        else:
            metas = _stream_preprocess_parquet(
                source,
                splits=set(splits_to_build),
                cache_dir=cache_dir,
                config=config,
                special=special,
                workers=workers,
            )
        for split, meta in metas.items():
            if config.strategy == "owt_segment":
                split_entries[split] = {
                    "status": "built",
                    **_split_meta_to_manifest(meta),
                }
            else:
                split_entries[split] = {
                    "status": "complete",
                    **_split_meta_to_manifest(meta),
                }
            split_counts[split] = meta.count
            _write_partial()

        if config.strategy == "owt_segment":
            _owt_shuffle_splits(
                cache_dir,
                config,
                list(metas.keys()),
                split_entries,
                split_counts,
                write_partial=_write_partial,
            )
    elif splits_to_build:
        for split in splits_to_build:
            _log_preprocess(f"Loading split={split!r} ...")
            load_started = time.time()
            hf_dataset = source.load_split(split)
            tqdm.write(
                f"[preprocess] split={split!r} loaded {len(hf_dataset):,} rows "
                f"({time.time() - load_started:.1f}s)"
            )
            if config.strategy == "owt_segment":
                meta = _owt_segment_preprocess_split_dataset(
                    hf_dataset,
                    split=split,
                    cache_dir=cache_dir,
                    config=config,
                )
            else:
                meta = _stream_preprocess_split_dataset(
                    hf_dataset,
                    split=split,
                    cache_dir=cache_dir,
                    config=config,
                    special=special,
                    workers=workers,
                )
            if config.strategy == "owt_segment":
                split_entries[split] = {
                    "status": "built",
                    **_split_meta_to_manifest(meta),
                }
            else:
                split_entries[split] = {
                    "status": "complete",
                    **_split_meta_to_manifest(meta),
                }
            split_counts[split] = meta.count
            _write_partial()

            if config.strategy == "owt_segment":
                _owt_shuffle_splits(
                    cache_dir,
                    config,
                    [split],
                    split_entries,
                    split_counts,
                    write_partial=_write_partial,
                )

    _write_manifest(
        cache_dir,
        _manifest_payload_base(
            config,
            fingerprint,
            status="complete",
            split_counts=split_counts,
            splits=split_entries,
        ),
    )
    return split_counts


def _load_manifest(cache_dir: Path) -> Dict[str, Any] | None:
    path = cache_dir / "manifest.yaml"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_cache(
    config: FL_PreprocessConfig,
    source: FL_Dataset,
    cache_dir: Path,
) -> tuple[Dict[str, int], Dict[str, _SplitCacheMeta]]:
    fingerprint = _fingerprint(config, source)
    manifest = _load_manifest(cache_dir)
    if manifest and manifest.get("fingerprint") == fingerprint:
        expected_version = config.manifest_version()
        if manifest.get("status") == "complete" and manifest.get("version") == expected_version:
            split_names = set(source.get_splits())
            if set(manifest.get("splits", {})) == split_names:
                splits = {
                    name: _split_meta_from_manifest(raw)
                    for name, raw in manifest.get("splits", {}).items()
                }
                if all(
                    _verify_split_cache(
                        cache_dir,
                        name,
                        meta,
                        chunk_length=int(manifest.get("chunk_length", config.chunk_length)),
                    )
                    for name, meta in splits.items()
                ):
                    return dict(manifest.get("split_counts", {})), splits

    cache_dir.mkdir(parents=True, exist_ok=True)
    _log_preprocess(
        f"Cache miss; building: dataset={source.config.name!r} "
        f"preprocess={config.name!r}"
    )
    _log_preprocess(f"Output directory: {cache_dir}")
    _log_preprocess(
        f"Parallel workers: {_worker_count()} "
        f"(tokenize stage uses all CPU cores; first run may take a while)"
    )
    split_counts = _build_cache(config, source, cache_dir)
    _log_preprocess(f"Cache build complete: {split_counts}")
    manifest = _load_manifest(cache_dir) or {}
    splits = {
        name: _split_meta_from_manifest(raw)
        for name, raw in manifest.get("splits", {}).items()
    }
    return split_counts, splits


class _PreprocessedSplitDataset(Dataset):
    """Memory-mapped view of one preprocessed split (possibly multi-file)."""

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        *,
        chunk_length: int,
        meta: _SplitCacheMeta,
    ) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.split = split
        self.chunk_length = chunk_length
        self.meta = meta
        if meta.count == 0:
            self._shard_counts = np.empty(0, dtype=np.int64)
            self._shard_starts = np.array([0], dtype=np.int64)
        else:
            self._shard_counts = np.asarray(
                [shard.count for shard in meta.shards], dtype=np.int64
            )
            self._shard_starts = np.concatenate(
                ([0], np.cumsum(self._shard_counts[:-1]))
            )
        self._maps: List[np.memmap | None] = [None] * len(meta.shards)
        self._lengths: np.memmap | None = None
        if meta.has_lengths:
            len_path = cache_dir / f"{split}.len"
            self._lengths = np.memmap(
                len_path, dtype=_DTYPE, mode="r", shape=(meta.count,)
            )

    def __len__(self) -> int:
        return int(self.meta.count)

    def _map_shard(self, shard_idx: int) -> np.memmap:
        cached = self._maps[shard_idx]
        if cached is not None:
            return cached
        shard = self.meta.shards[shard_idx]
        mmap = np.memmap(
            self.cache_dir / shard.file,
            dtype=_DTYPE,
            mode="r",
            shape=(shard.count, self.chunk_length),
        )
        self._maps[shard_idx] = mmap
        return mmap

    def _resolve(self, index: int) -> tuple[int, int]:
        shard_idx = int(
            np.searchsorted(self._shard_starts, index, side="right") - 1
        )
        local_idx = index - int(self._shard_starts[shard_idx])
        return shard_idx, local_idx

    def locate(self, index: int) -> int:
        """Validate a global sample index for resuming training."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Sample index {index} out of range [0, {len(self)}) "
                f"for split '{self.split}'"
            )
        return index

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        index = self.locate(index)
        shard_idx, local_idx = self._resolve(index)
        row = self._map_shard(shard_idx)[local_idx]
        item: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(row, dtype=torch.long)
        }
        if self._lengths is not None:
            item["length"] = torch.tensor(int(self._lengths[index]), dtype=torch.long)
        return item

    def __getitems__(self, indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Batch path for DataLoader (PyTorch 2+); groups reads by shard."""
        if not indices:
            return []
        resolved = [self._resolve(self.locate(i)) for i in indices]
        out: List[Dict[str, torch.Tensor] | None] = [None] * len(indices)
        by_shard: Dict[int, List[tuple[int, int]]] = {}
        for pos, (shard_idx, local_idx) in enumerate(resolved):
            by_shard.setdefault(shard_idx, []).append((pos, local_idx))

        for shard_idx, entries in by_shard.items():
            mmap = self._map_shard(shard_idx)
            for pos, local_idx in entries:
                row = mmap[local_idx]
                item: Dict[str, torch.Tensor] = {
                    "input_ids": torch.tensor(row, dtype=torch.long)
                }
                global_index = indices[pos]
                if self._lengths is not None:
                    item["length"] = torch.tensor(
                        int(self._lengths[global_index]), dtype=torch.long
                    )
                out[pos] = item
        return out  # type: ignore[return-value]


class FL_PreprocessedDataset:
    """Preprocessed dataset with the same splits as its source ``FL_Dataset``."""

    def __init__(self, config: FL_PreprocessConfig, source: FL_Dataset) -> None:
        self.config = config
        self.source = source
        self._split_views: Dict[str, _PreprocessedSplitDataset] = {}

        source.ensure_downloaded()

        self.cache_dir = _cache_dir(config, source)
        self.split_counts, self._split_meta = _ensure_cache(
            config, source, self.cache_dir
        )

    def get_splits(self) -> List[str]:
        return self.source.get_splits()

    def load_split(self, split: str) -> _PreprocessedSplitDataset:
        if split not in self.get_splits():
            raise ValueError(
                f"Unknown split '{split}'. Supported: {self.get_splits()}"
            )
        if split not in self._split_meta:
            raise FileNotFoundError(
                f"Preprocessed split '{split}' is not available in cache."
            )
        if split not in self._split_views:
            self._split_views[split] = _PreprocessedSplitDataset(
                self.cache_dir,
                split,
                chunk_length=self.config.chunk_length,
                meta=self._split_meta[split],
            )
        return self._split_views[split]

    def get_split_counts(self) -> Dict[str, int]:
        return dict(self.split_counts)
