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
from typing import Any, Dict, Iterator, List, Literal, Set, Union

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
_MANIFEST_VERSION = 2
_OVERFLOW_MODES = frozenset({"wrap", "discard", "pad_eos"})
_DTYPE = np.int32
_DOCS_PER_TASK = 512
_MAX_TOKENIZE_WORKERS = 16
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
    chunk_length: int = 1024
    overflow_mode: OverflowMode = "discard"
    seed: int = 42
    text_column: str = "text"
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
        return config


@dataclass(frozen=True)
class _SplitShardMeta:
    file: str
    count: int


@dataclass
class _SplitCacheMeta:
    count: int
    shards: List[_SplitShardMeta]
    has_lengths: bool = False


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


def _fingerprint(config: FL_PreprocessConfig, source: FL_Dataset) -> str:
    dc = source.config
    payload = {
        "preprocess": {
            "name": config.name,
            "tokenizer": config.tokenizer,
            "chunk_length": config.chunk_length,
            "overflow_mode": config.overflow_mode,
            "seed": config.seed,
            "text_column": config.text_column,
        },
        "dataset": {
            "name": dc.name,
            "repo_id": dc.repo_id,
            "revision": dc.revision,
            "subset": dc.subset,
            "split": dc.split,
            "eval_count": dc.eval_count,
            "eval_seed": dc.eval_seed,
        },
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_dir(config: FL_PreprocessConfig, source: FL_Dataset) -> Path:
    return CACHE_DIR / f"{source.config.name}_{config.name}_{_fingerprint(config, source)}"



def _worker_count() -> int:
    return min(_MAX_TOKENIZE_WORKERS, max(1, os.cpu_count() or 1))


def _log_preprocess(message: str) -> None:
    tqdm.write(f"[preprocess] {message}")


def _init_tokenizer_worker(tokenizer_name: str) -> None:
    global _WORKER_TOKENIZER
    os.environ["BDELF_QUIET_TOKENIZER"] = "1"
    _WORKER_TOKENIZER = get_tokenizer(tokenizer_name)
    # 先完整 tokenize，再按 token 长度切块；预处理阶段不应受 model_max_length 约束。
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
    eval_indices = source.holdout_eval_indices()
    pending: Dict[str, List[str]] = {split: [] for split in splits}

    with tqdm(
        total=total_rows,
        desc="[preprocess] 读取",
        unit="row",
        dynamic_ncols=True,
    ) as row_progress:
        for row_index, text in source.iter_parquet_rows(text_column=text_column):
            row_progress.update(1)
            if text is None:
                continue
            split = "eval" if row_index in eval_indices else "train"
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
            tqdm.write(f"[preprocess] split={split!r}: 完成，共 {meta.count:,} chunks")
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
        f"[preprocess] 单次顺序扫描 parquet: rows={total_rows:,}, "
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
        f"[preprocess] split={split!r}: {total:,} 条文本 "
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


def _split_meta_from_manifest(raw: Dict[str, Any]) -> _SplitCacheMeta:
    shards = [
        _SplitShardMeta(file=item["file"], count=int(item["count"]))
        for item in raw.get("shards", [])
    ]
    return _SplitCacheMeta(
        count=int(raw.get("count", 0)),
        shards=shards,
        has_lengths=bool(raw.get("has_lengths", False)),
    )


def _split_meta_to_manifest(meta: _SplitCacheMeta) -> Dict[str, Any]:
    return {
        "count": meta.count,
        "has_lengths": meta.has_lengths,
        "shards": [{"file": s.file, "count": s.count} for s in meta.shards],
    }


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


def _build_cache(
    config: FL_PreprocessConfig,
    source: FL_Dataset,
    cache_dir: Path,
) -> Dict[str, int]:
    special = get_token_layout(config.tokenizer)
    workers = _worker_count()
    fingerprint = _fingerprint(config, source)
    manifest = _load_manifest(cache_dir)
    if manifest and manifest.get("fingerprint") != fingerprint:
        _cleanup_cache_dir(cache_dir)
    elif manifest and manifest.get("version") != _MANIFEST_VERSION:
        _cleanup_cache_dir(cache_dir)

    existing = _load_manifest(cache_dir) or {}
    split_entries: Dict[str, Any] = dict(existing.get("splits", {}))
    split_counts: Dict[str, int] = {}
    splits_to_build: List[str] = []
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
        _cleanup_split(cache_dir, split)
        splits_to_build.append(split)

    if splits_to_build and source.can_stream_parquet():
        metas = _stream_preprocess_parquet(
            source,
            splits=set(splits_to_build),
            cache_dir=cache_dir,
            config=config,
            special=special,
            workers=workers,
        )
        for split, meta in metas.items():
            split_entries[split] = {
                "status": "complete",
                **_split_meta_to_manifest(meta),
            }
            split_counts[split] = meta.count
            _write_manifest(
                cache_dir,
                {
                    "version": _MANIFEST_VERSION,
                    "status": "partial",
                    "fingerprint": fingerprint,
                    "chunk_length": config.chunk_length,
                    "overflow_mode": config.overflow_mode,
                    "split_counts": dict(split_counts),
                    "splits": split_entries,
                },
            )
    else:
        for split in splits_to_build:
            _log_preprocess(f"加载 split={split!r} ...")
            load_started = time.time()
            hf_dataset = source.load_split(split)
            tqdm.write(
                f"[preprocess] split={split!r} 已加载 {len(hf_dataset):,} 条 "
                f"({time.time() - load_started:.1f}s)"
            )
            meta = _stream_preprocess_split_dataset(
                hf_dataset,
                split=split,
                cache_dir=cache_dir,
                config=config,
                special=special,
                workers=workers,
            )
            split_entries[split] = {
                "status": "complete",
                **_split_meta_to_manifest(meta),
            }
            split_counts[split] = meta.count
            _write_manifest(
                cache_dir,
                {
                    "version": _MANIFEST_VERSION,
                    "status": "partial",
                    "fingerprint": fingerprint,
                    "chunk_length": config.chunk_length,
                    "overflow_mode": config.overflow_mode,
                    "split_counts": dict(split_counts),
                    "splits": split_entries,
                },
            )

    _write_manifest(
        cache_dir,
        {
            "version": _MANIFEST_VERSION,
            "status": "complete",
            "fingerprint": fingerprint,
            "chunk_length": config.chunk_length,
            "overflow_mode": config.overflow_mode,
            "split_counts": split_counts,
            "splits": split_entries,
        },
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
        if manifest.get("status") == "complete" and manifest.get("version") == _MANIFEST_VERSION:
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
        f"缓存未命中，开始构建: dataset={source.config.name!r} "
        f"preprocess={config.name!r}"
    )
    _log_preprocess(f"输出目录: {cache_dir}")
    _log_preprocess(
        f"并行 worker 数: {_worker_count()}（tokenize 阶段会占满 CPU，首次运行可能耗时较长）"
    )
    split_counts = _build_cache(config, source, cache_dir)
    _log_preprocess(f"缓存构建完成: {split_counts}")
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
