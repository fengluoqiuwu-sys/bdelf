"""Checkpoint run identity: config hash → ``cache/checkpoints/{variant}/{model}/{hash}/``.

不允许别名 / 软链；路径唯一由训练入参与所解析 YAML 内容决定。
``world_size`` / 派生 accum / 微步间隔 / GPU 硬件规格不进指纹
（硬件另行写入 run 目录 ``hardware.json`` 并在续跑时校验）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from models import resolve_model_config_path

CHECKPOINT_ROOT = "cache/checkpoints"
CONFIG_HASH_LEN = 16


def _strip_meta(obj: Any) -> Any:
    """去掉 ``_`` 前缀键（如 ``_doc``），并递归规范化。"""
    if isinstance(obj, Mapping):
        return {
            str(k): _strip_meta(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
            if not str(k).startswith("_")
        }
    if isinstance(obj, list):
        return [_strip_meta(v) for v in obj]
    if isinstance(obj, tuple):
        return [_strip_meta(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def canonical_json(obj: Any) -> str:
    """稳定 JSON：排序键、无多余空白，供哈希。"""
    return json.dumps(
        _strip_meta(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    return raw


def dataclass_fingerprint(obj: Any) -> dict[str, Any]:
    """``asdict`` 后合并 ``extra``（去掉 ``_`` 键与冗余 ``name``）。"""
    from dataclasses import asdict, is_dataclass

    if not is_dataclass(obj):
        raise TypeError(f"expected dataclass, got {type(obj)!r}")
    raw = asdict(obj)
    extra = raw.pop("extra", {}) or {}
    merged = dict(raw)
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            if str(key).startswith("_"):
                continue
            merged[key] = value
    merged.pop("name", None)
    return _strip_meta(merged)


def build_train_fingerprint(
    *,
    model: str,
    model_config: str,
    variant: str,
    dataset: str,
    preprocess: str,
    generate: str,
    optimizer: Any,
    batch: Any,
    schedule: Any,
    eval_cfg: Any,
    generate_cfg: Any,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造进入哈希的指纹（不含 world_size / 硬件派生量）。"""
    repo = Path(__file__).resolve().parents[1]
    model_arch = _load_yaml_mapping(
        resolve_model_config_path(model, model_config),
    )
    # ``--set model.*`` 并入架构指纹（与 train.py 加载时一致）
    if overrides and overrides.get("model"):
        model_arch = {**model_arch, **dict(overrides["model"])}
    preprocess_yaml = _load_yaml_mapping(
        repo / "config" / "preprocess" / f"{preprocess}.yaml",
    )
    dataset_yaml = _load_yaml_mapping(
        repo / "config" / "datasets" / f"{dataset}.yaml",
    )
    gen_piece: dict[str, Any]
    if hasattr(generate_cfg, "to_sampling_cfg"):
        gen_piece = {"profile": generate, **generate_cfg.to_sampling_cfg()}
    elif isinstance(generate_cfg, Mapping):
        gen_piece = dict(generate_cfg)
    else:
        gen_piece = dataclass_fingerprint(generate_cfg)

    return {
        "model": model,
        "model_config": model_config,
        "variant": variant,
        "dataset": dataset,
        "preprocess": preprocess,
        "generate": generate,
        "overrides": _strip_meta(dict(overrides or {})),
        "optimizer": dataclass_fingerprint(optimizer),
        "batch": dataclass_fingerprint(batch),
        "schedule": dataclass_fingerprint(schedule),
        "eval": dataclass_fingerprint(eval_cfg),
        "generate_cfg": _strip_meta(gen_piece),
        "model_arch": model_arch,
        "preprocess_cfg": preprocess_yaml,
        "dataset_cfg": dataset_yaml,
    }


def config_hash_from_fingerprint(fingerprint: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(fingerprint).encode("utf-8")).hexdigest()
    return digest[:CONFIG_HASH_LEN]


def run_dir_for(
    *,
    variant: str,
    model: str,
    config_hash: str,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> Path:
    """``{root}/{fast|full}/{model}/{hash}/``；禁止别名。"""
    if variant not in ("fast", "full"):
        raise ValueError(f"variant must be fast|full, got {variant!r}")
    if not config_hash or any(c in config_hash for c in "/\\"):
        raise ValueError(f"invalid config_hash: {config_hash!r}")
    return Path(checkpoint_root) / variant / model / config_hash


def run_relpath(*, variant: str, model: str, config_hash: str) -> str:
    """相对 ``checkpoint_root`` 的路径字符串（generate ``--run`` 用）。"""
    return f"{variant}/{model}/{config_hash}"


def checkpoint_run_dir(
    *,
    variant: str,
    model: str,
    config_hash: str,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> Path:
    return run_dir_for(
        variant=variant,
        model=model,
        config_hash=config_hash,
        checkpoint_root=checkpoint_root,
    )


def checkpoint_run_dir_from_cfg(cfg: Any) -> Path:
    """从 ``FL_TrainConfig`` 得到 run 目录（``cfg.name`` 即 config hash）。"""
    return run_dir_for(
        variant=cfg.variant,
        model=cfg.model,
        config_hash=cfg.name,
        checkpoint_root=cfg.checkpoint_root,
    )
