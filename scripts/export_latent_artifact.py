#!/usr/bin/env python3
"""把 latent 训练 run 的 checkpoint_latest 导出到 artifacts（推理权重）。

有 EMA 则熔进最终参数，不另存优化器 / 训练配置。工作目录为仓库根::

    .venv/bin/python scripts/export_latent_artifact.py \\
      --run full/latent/latent_vae/<hash>

    .venv/bin/python scripts/export_latent_artifact.py \\
      --run full/latent/latent_vae/<hash> --tag 100m-b32-d1-sigma --force
"""

from __future__ import annotations

import argparse
import sys

import repo_env

repo_env.ensure_repo_root()

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
            "--run full/latent/latent_vae/<hash> --tag 100m-b32-d1-sigma --force\n"
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
        help="选用 tag；默认 {size}-b{latent_dim}-d{block_size}[-sigma]，如 100m-b32-d1-sigma",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已有 artifacts 目录",
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
    )
    print(f"exported {dest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
