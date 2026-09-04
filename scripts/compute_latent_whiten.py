#!/usr/bin/env python3
"""对已导出的 latent artifact 离线写入逐维 μ 白化统计。

不跑完全集：默认 ``DEFAULT_WHITEN_TOKENS``（2^20 有效 token）。
须已有预处理缓存（默认 owt + owt-seg512），本脚本不从头切分。

    .venv/bin/python scripts/compute_latent_whiten.py \\
      --latent-model latent_vae --tag 100m-b32-d16

    .venv/bin/python scripts/compute_latent_whiten.py \\
      --latent-model latent_vae --tag 100m-b32-d16 --force
"""

from __future__ import annotations

import argparse
import json
import sys

import repo_env

repo_env.ensure_repo_root()

import hf_config  # noqa: F401

from models.latent.whiten_stats import (
    DEFAULT_WHITEN_BATCH,
    DEFAULT_WHITEN_DATASET,
    DEFAULT_WHITEN_PREPROCESS,
    DEFAULT_WHITEN_SEED,
    DEFAULT_WHITEN_SPLIT,
    DEFAULT_WHITEN_TOKENS,
    write_artifact_whiten,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-dim μ mean/std on an artifacts/latent tag "
            f"(default {DEFAULT_WHITEN_TOKENS} valid tokens, not the full corpus)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/compute_latent_whiten.py "
            "--latent-model latent_vae --tag 100m-b32-d16\n"
            "  python scripts/compute_latent_whiten.py "
            "--latent-model latent_vae --tag 100m-b32-d16 --force\n"
        ),
    )
    parser.add_argument("--latent-model", required=True, help="artifacts 下的 latent 模型名")
    parser.add_argument("--tag", required=True, help="artifacts/latent/{model}/{tag}")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_WHITEN_DATASET,
        help=f"数据集名（默认 {DEFAULT_WHITEN_DATASET}）",
    )
    parser.add_argument(
        "--preprocess",
        default=DEFAULT_WHITEN_PREPROCESS,
        help=f"预处理名（默认 {DEFAULT_WHITEN_PREPROCESS}；须已有完整缓存）",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_WHITEN_SPLIT,
        help=f"split（默认 {DEFAULT_WHITEN_SPLIT}）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_WHITEN_TOKENS,
        help=f"有效 token 上限（默认 {DEFAULT_WHITEN_TOKENS}，不跑全集）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_WHITEN_BATCH,
        help=f"encode 批大小（默认 {DEFAULT_WHITEN_BATCH}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_WHITEN_SEED,
        help=f"抽样种子（默认 {DEFAULT_WHITEN_SEED}）",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / cpu；默认有 CUDA 则用 cuda",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖 artifact 里已有的 whitening_mean/std",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = write_artifact_whiten(
        args.latent_model,
        args.tag,
        dataset=args.dataset,
        preprocess=args.preprocess,
        split=args.split,
        max_tokens=int(args.max_tokens),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=args.device,
        force=bool(args.force),
    )
    meta = stats.as_meta()
    summary = {
        "latent_model": args.latent_model,
        "tag": args.tag,
        "n_valid": meta["n_valid"],
        "latent_dim": meta["latent_dim"],
        "std_min": meta["std_min"],
        "std_max": meta["std_max"],
        "std_ratio": meta["std_ratio"],
        "dataset": meta["dataset"],
        "preprocess": meta["preprocess"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
