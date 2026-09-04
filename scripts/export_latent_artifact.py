#!/usr/bin/env python3
"""把训练 run 的 latest 导出到 artifacts（推理权重 + 随后离线 μ 白化）。

有 EMA 则熔进最终参数，不另存优化器 / 训练配置。工作目录为仓库根::

    .venv/bin/python scripts/export_latent_artifact.py \\
      --run full/latent/latent_vae/<hash>

    .venv/bin/python scripts/export_latent_artifact.py \\
      --run full/latent/latent_vae/<hash> --tag 100m-b32-d1 --force

白化可单独补算::

    .venv/bin/python scripts/compute_latent_whiten.py \\
      --latent-model latent_vae --tag 100m-b32-d1
"""

from __future__ import annotations

import argparse
import sys

import repo_env

repo_env.ensure_repo_root()

import hf_config  # noqa: F401

from models.latent.artifact_export import export_latent_artifact
from models.latent.artifact_loader import list_artifact_tags


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export latent checkpoint_latest.pt to "
            "cache/checkpoints/artifacts/latent/{model}/{tag}/ "
            "(EMA baked into weights; model config only)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/export_latent_artifact.py "
            "--run full/latent/latent_vae/<hash>\n"
            "  python scripts/export_latent_artifact.py "
            "--run full/latent/latent_vae/<hash> --tag 100m-b32-d1 --force\n"
            "  python scripts/export_latent_artifact.py "
            "--run full/latent/latent_vae/<hash> --skip-whiten\n"
        ),
    )
    parser.add_argument(
        "--run",
        help="训练 run：{fast|full}/latent/{model}/{hash}",
    )
    parser.add_argument(
        "--checkpoint",
        help="显式 checkpoint_latest.pt 路径",
    )
    parser.add_argument(
        "--tag",
        help="选用 tag；默认 {size}-b{latent_dim}-d{block_size}，如 100m-b32-d1；kl_entropy 开时后缀 -sigma",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已有 artifacts 目录",
    )
    parser.add_argument(
        "--skip-whiten",
        action="store_true",
        help="只写权重，不离线估计逐维 μ 白化（随后可跑 compute_latent_whiten.py）",
    )
    parser.add_argument(
        "--list",
        metavar="MODEL",
        help="列出该 latent 模型已有 artifacts tag 后退出",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.list:
        tags = list_artifact_tags(args.list)
        if not tags:
            print(f"(no tags under artifacts/latent/{args.list}/)", file=sys.stderr)
            return
        for tag in tags:
            print(tag)
        return
    if not args.run and not args.checkpoint:
        print("须指定 --run 或 --checkpoint（或 --list MODEL）", file=sys.stderr)
        sys.exit(2)
    dest = export_latent_artifact(
        run=args.run,
        checkpoint=args.checkpoint,
        tag=args.tag,
        force=args.force,
        skip_whiten=args.skip_whiten,
    )
    print(f"exported {dest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
