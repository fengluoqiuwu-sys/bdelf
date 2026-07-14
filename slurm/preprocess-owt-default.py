#!/usr/bin/env python3
"""Build OWT + default preprocess cache (single process, Slurm-safe).

Must be a real file on disk: preprocess uses multiprocessing spawn, which
cannot re-exec ``python - <<'PY'`` / ``<stdin>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]


def validate_raw_dataset(dataset: str) -> None:
    root = PROJECT / "cache" / "datasets" / dataset
    if not root.exists():
        raise SystemExit(
            f"missing {root}; download on login node first:\n"
            f"  python download_dataset.py {dataset}"
        )

    parquet = sorted(root.rglob("*.parquet"))
    if not parquet:
        raise SystemExit(f"no parquet under {root}")

    total = sum(p.stat().st_size for p in parquet)
    print(f"parquet_files={len(parquet)} total_bytes={total}")
    if total < 1 * 1024**3:
        raise SystemExit(f"parquet too small: {total / 1024**3:.2f} GiB")
    print("raw dataset OK")


def build_cache(dataset: str, preprocess: str) -> None:
    sys.path.insert(0, str(PROJECT))
    from preprocess import get_preprocessed

    print(f"[preprocess] building cache: dataset={dataset!r} preprocess={preprocess!r}")
    ds = get_preprocessed(preprocess, dataset)
    splits = ds.get_splits()
    print(f"[preprocess] splits={splits}")
    for split in splits:
        loaded = ds.load_split(split)
        print(f"[preprocess] split={split!r} samples={len(loaded):,}")
    print("[preprocess] done")


def validate_manifest(dataset: str, preprocess: str) -> Path:
    cache_root = PROJECT / "cache" / "preprocessed_datasets"
    pattern = f"{dataset}_{preprocess}_*"
    dirs = sorted(cache_root.glob(pattern))
    if not dirs:
        raise SystemExit(f"no {pattern} cache directory found")

    cache_dir = dirs[-1]
    manifest_path = cache_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    status = manifest.get("status")
    split_counts = manifest.get("split_counts", {})
    if status != "complete":
        raise SystemExit(f"manifest status={status!r}, expected 'complete'")

    print(f"cache_dir={cache_dir}")
    print(f"status={status}")
    print(f"split_counts={split_counts}")
    print("manifest OK")
    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preprocess cache for Slurm jobs.")
    parser.add_argument("--dataset", default="owt")
    parser.add_argument("--preprocess", default="default")
    args = parser.parse_args()

    print("=== validate raw dataset ===")
    validate_raw_dataset(args.dataset)

    print("=== build preprocess cache (single process) ===")
    build_cache(args.dataset, args.preprocess)

    print("=== validate manifest ===")
    validate_manifest(args.dataset, args.preprocess)


if __name__ == "__main__":
    main()
