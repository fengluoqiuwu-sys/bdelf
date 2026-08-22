#!/usr/bin/env python3
"""Copy checkpoint runs into ``{variant}/{kind}/{model}/{hash}/`` (keep legacy tree)."""

from __future__ import annotations

import shutil
from pathlib import Path

import repo_env

repo_env.ensure_repo_root()

from models import kind_of, list_models

CHECKPOINT_ROOT = Path("cache/checkpoints")
VARIANTS = ("fast", "full")
SKIP_TOP = frozenset({"artifacts", "lm", "latent", "hash_guide.csv"})


def _base_family(model_dir: str) -> str:
    return model_dir[:-4] if model_dir.endswith("-old") else model_dir


def _kind_for_model_dir(model_dir: str) -> str | None:
    base = _base_family(model_dir)
    try:
        return kind_of(base)
    except KeyError:
        return None


def copy_checkpoints(*, dry_run: bool = False) -> int:
    if not CHECKPOINT_ROOT.is_dir():
        print("no checkpoint root")
        return 0
    copied = 0
    registered = set(list_models())
    for variant in VARIANTS:
        variant_dir = CHECKPOINT_ROOT / variant
        if not variant_dir.is_dir():
            continue
        for model_dir in sorted(variant_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name in SKIP_TOP:
                continue
            kind = _kind_for_model_dir(model_dir.name)
            if kind is None:
                print(f"skip unknown model dir: {model_dir}")
                continue
            base = _base_family(model_dir.name)
            if base not in registered and not model_dir.name.endswith("-old"):
                print(f"skip unregistered: {model_dir.name}")
                continue
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                dst = variant_dir / kind / model_dir.name / run_dir.name
                if dst.exists():
                    continue
                if dry_run:
                    print(f"would copy {run_dir} -> {dst}")
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(run_dir, dst, symlinks=True)
                    print(f"copied {run_dir.relative_to(CHECKPOINT_ROOT)} -> {dst.relative_to(CHECKPOINT_ROOT)}")
                copied += 1
    return copied


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    n = copy_checkpoints(dry_run=args.dry_run)
    print(f"done: {n} run(s)")
