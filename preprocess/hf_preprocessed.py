"""从 Hub 已切分 parquet 物化成本地 memmap 缓存（与原文切分同指纹、同行序）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from preprocess.owt_split import bucket_counts_from_lengths
from preprocess.preprocess import (
    FL_PreprocessConfig,
    _DTYPE,
    _ShardWriter,
    _SplitCacheMeta,
    _cleanup_cache_dir,
    _fingerprint,
    _log_preprocess,
    _manifest_payload_base,
    _split_meta_to_manifest,
    _write_manifest,
)
from tokenizer import get_token_layout


def _parquet_files(data_dir: Path, hub_split: str) -> List[Path]:
    files = sorted(data_dir.glob(f"{hub_split}-*.parquet"))
    if files:
        return files
    return sorted(data_dir.glob(f"{hub_split}*.parquet"))


def _hub_split_for_local(local_split: str) -> str:
    if local_split == "dev":
        return "validation"
    return local_split


def _table_to_padded(
    table,
    *,
    chunk_length: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """变长 input_ids + length → 定宽 memmap 行（右 pad）。"""
    lengths = np.asarray(table.column("length").to_numpy(), dtype=_DTYPE)
    n = int(lengths.shape[0])
    rows = np.full((n, chunk_length), pad_id, dtype=_DTYPE)
    col = table.column("input_ids")
    combined = col.combine_chunks() if col.num_chunks != 1 else col.chunk(0)
    offsets = np.asarray(combined.offsets)
    values = np.asarray(combined.values)
    for i in range(n):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        L = int(lengths[i])
        if end - start != L:
            raise RuntimeError(
                f"input_ids 长度 {end - start} 与 length={L} 不一致（row={i}）"
            )
        if L < 1 or L > chunk_length:
            raise RuntimeError(f"length={L} 超出 [1, {chunk_length}]（row={i}）")
        rows[i, :L] = values[start:end]
    return rows, lengths


def materialize_split_from_parquet(
    parquet_files: List[Path],
    *,
    cache_dir: Path,
    split: str,
    config: FL_PreprocessConfig,
    pad_id: int,
) -> _SplitCacheMeta:
    if not parquet_files:
        raise FileNotFoundError(f"没有 {split} 的 parquet 分片")
    writer = _ShardWriter(
        cache_dir,
        split,
        chunk_length=config.chunk_length,
        record_lengths=True,
    )
    total_rows = 0
    for path in parquet_files:
        pf = pq.ParquetFile(path)
        n_groups = pf.num_row_groups
        desc = f"[preprocess] hf {split} {path.name}"
        for rg in tqdm(range(n_groups), desc=desc, unit="rg", leave=False):
            table = pf.read_row_group(rg, columns=["input_ids", "length"])
            rows, lengths = _table_to_padded(
                table,
                chunk_length=config.chunk_length,
                pad_id=pad_id,
            )
            writer.append(rows, lengths)
            total_rows += int(rows.shape[0])
    meta = writer.finalize()
    if config.pad_mode == "bucket" and config.bucket_lengths and meta.has_lengths:
        len_mmap = np.memmap(
            cache_dir / f"{split}.len",
            dtype=_DTYPE,
            mode="r",
            shape=(meta.count,),
        )
        meta.bucket_counts = bucket_counts_from_lengths(
            len_mmap.tolist(), config.bucket_lengths
        )
    if meta.count != total_rows:
        raise RuntimeError(
            f"{split}: writer count {meta.count} != parquet rows {total_rows}"
        )
    return meta


def download_hf_preprocessed(
    repo_id: str,
    *,
    revision: str,
) -> Path:
    import hf_config  # noqa: F401
    from huggingface_hub import snapshot_download

    _log_preprocess(f"Downloading preprocessed dataset {repo_id}@{revision}")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["data/*.parquet", "meta.json"],
    )
    return Path(path)


def _check_hub_fingerprint(
    snapshot: Path,
    expected: str,
    *,
    preprocess_name: str,
) -> None:
    meta_path = snapshot / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{preprocess_name}: Hub 成品缺少 meta.json（{snapshot}）"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    got = str(meta.get("fingerprint") or "")
    if got != expected:
        raise RuntimeError(
            f"{preprocess_name}: Hub fingerprint={got!r} 与本地配置 "
            f"fingerprint={expected!r} 不一致。请改 cache_source: raw "
            "从原文重切，或检查 hf_preprocessed_repo / revision。"
        )


def build_cache_from_hf(
    config: FL_PreprocessConfig,
    source,
    cache_dir: Path,
    *,
    snapshot_dir: Path | None = None,
) -> Dict[str, int]:
    """下载 Hub parquet 并写成与原文切分相同的 memmap 目录。"""
    from preprocess.preprocess import resolved_hf_repo, resolved_hf_revision

    fingerprint = _fingerprint(config, source)
    repo_id = resolved_hf_repo(config)
    if not repo_id:
        raise RuntimeError(
            f"{config.name}: cache_source=hf 但未配置 hf_preprocessed_repo"
        )
    revision = resolved_hf_revision(config)
    pad_id = int(get_token_layout(config.tokenizer).pad_token_id)

    snapshot = snapshot_dir or download_hf_preprocessed(repo_id, revision=revision)
    _check_hub_fingerprint(snapshot, fingerprint, preprocess_name=config.name)

    data_dir = snapshot / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{repo_id}: 缺少 data/ 目录（{snapshot}）")

    _cleanup_cache_dir(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_entries: Dict[str, dict] = {}
    split_counts: Dict[str, int] = {}
    local_splits = list(source.get_splits())
    for local_split in local_splits:
        hub_split = _hub_split_for_local(local_split)
        files = _parquet_files(data_dir, hub_split)
        if not files and local_split == "dev":
            files = _parquet_files(data_dir, "dev")
        _log_preprocess(
            f"Materializing split={local_split!r} from {len(files)} parquet files"
        )
        meta = materialize_split_from_parquet(
            files,
            cache_dir=cache_dir,
            split=local_split,
            config=config,
            pad_id=pad_id,
        )
        split_entries[local_split] = {
            "status": "complete",
            **_split_meta_to_manifest(meta),
        }
        split_counts[local_split] = meta.count

    train_count = int(split_counts.get("train", 0))
    if train_count <= 0:
        raise RuntimeError(
            "HF 成品 train split 为 0 chunks，拒绝写入 complete manifest"
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
