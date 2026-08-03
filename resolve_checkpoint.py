#!/usr/bin/env python3
"""由与 ``train.py`` 相同的入参解析 checkpoint 配置哈希与目录。

用法::

    .venv/bin/python resolve_checkpoint.py \\
      --model ar --config 100m-fast \\
      --dataset owt --preprocess default --generate eval

    .venv/bin/python resolve_checkpoint.py ... --hash-only
    .venv/bin/python resolve_checkpoint.py ... --json
"""

from __future__ import annotations

import argparse
import json
import sys

from dataset import list_datasets
from models import list_model_configs, list_models, resolve_model_config_path
from preprocess import list_preprocess
from train import (
    get_train_config,
    list_generate,
    list_train_configs,
    list_train_models,
    parse_train_overrides,
)
from train.run_path import checkpoint_run_dir_from_cfg


def build_arg_parser() -> argparse.ArgumentParser:
    models = list_models() or ["<none>"]
    datasets = list_datasets() or ["<none>"]
    preprocess_names = list_preprocess() or ["<none>"]
    parser = argparse.ArgumentParser(
        description=(
            "Resolve train launch args to "
            "cache/checkpoints/{fast|full}/{model}/{config-hash}/ "
            "(same CLI rules as train.py; no aliases)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python resolve_checkpoint.py --model ar --config 100m-fast "
            "--dataset owt --preprocess default --generate eval\n"
            "  python resolve_checkpoint.py --model elf --config 100m-full "
            "--dataset owt --preprocess elf --generate eval --hash-only\n"
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model family name; options: {', '.join(models)}",
    )
    parser.add_argument(
        "--config",
        required=True,
        dest="train_config",
        metavar="CONFIG",
        help="Train config name, e.g. 100m-fast / 100m-full",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name (config/datasets/); options: {', '.join(datasets)}",
    )
    parser.add_argument(
        "--preprocess",
        required=True,
        help=(
            f"Preprocess config name (config/preprocess/); "
            f"options: {', '.join(preprocess_names)}"
        ),
    )
    parser.add_argument(
        "--generate",
        required=True,
        help="Generate config under config/generate/<model>/ (e.g. eval)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="SECTION.KEY=VALUE",
        help="Same as train.py --set (repeatable)",
    )
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="Print only the config hash",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print JSON with hash / relpath / absolute path",
    )
    return parser


def _validate_and_load(args: argparse.Namespace):
    models = list_models()
    if args.model not in models:
        raise SystemExit(
            f"Unknown model {args.model!r}. Available: {', '.join(models) or '<none>'}"
        )
    train_models = list_train_models()
    if args.model not in train_models:
        raise SystemExit(
            f"Model {args.model!r} has no train recipe. "
            f"Available: {', '.join(train_models)}"
        )
    configs = list_train_configs(args.model)
    if args.train_config not in configs:
        raise SystemExit(
            f"Unknown train config {args.train_config!r}. "
            f"{args.model} available: {', '.join(configs)}"
        )
    size = args.train_config.rsplit("-", 1)[0]
    try:
        resolve_model_config_path(args.model, size)
    except FileNotFoundError as exc:
        available = ", ".join(list_model_configs(args.model)) or "<none>"
        raise SystemExit(
            f"Model architecture config not found for {args.model}/{size}. "
            f"Available: {available}\n{exc}"
        ) from exc

    datasets = list_datasets()
    if args.dataset not in datasets:
        raise SystemExit(
            f"Unknown dataset {args.dataset!r}. "
            f"Available: {', '.join(datasets) or '<none>'}"
        )
    preprocess_names = list_preprocess()
    if args.preprocess not in preprocess_names:
        raise SystemExit(
            f"Unknown preprocess {args.preprocess!r}. "
            f"Available: {', '.join(preprocess_names) or '<none>'}"
        )
    generate_names = list_generate(args.model)
    if args.generate not in generate_names:
        raise SystemExit(
            f"Unknown generate config {args.generate!r}. "
            f"{args.model} available: {', '.join(generate_names) or '<none>'}"
        )

    try:
        overrides = parse_train_overrides(args.overrides)
    except ValueError as exc:
        raise SystemExit(f"Invalid --set override: {exc}") from exc

    # world_size 不进哈希；用 1 仅满足 compose 的整除约束
    try:
        cfg = get_train_config(
            args.model,
            args.train_config,
            dataset=args.dataset,
            preprocess=args.preprocess,
            generate=args.generate,
            world_size=1,
            overrides=overrides,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Failed to load train config: {exc}") from exc
    return cfg


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = _validate_and_load(args)
    run_dir = checkpoint_run_dir_from_cfg(cfg)
    rel = cfg.extra.get("run_relpath") or f"{cfg.variant}/{cfg.model}/{cfg.name}"

    if args.hash_only:
        print(cfg.name)
        return
    if args.as_json:
        print(
            json.dumps(
                {
                    "config_hash": cfg.name,
                    "run_relpath": rel,
                    "run_dir": str(run_dir),
                    "exists": run_dir.is_dir(),
                    "checkpoint_latest": str(run_dir / "checkpoint_latest.pt"),
                    "checkpoint_latest_exists": (run_dir / "checkpoint_latest.pt").is_file(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"config_hash: {cfg.name}")
    print(f"run_relpath: {rel}")
    print(f"run_dir:     {run_dir}")
    print(f"exists:      {run_dir.is_dir()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupt received; exiting.", file=sys.stderr)
        raise SystemExit(130) from None
