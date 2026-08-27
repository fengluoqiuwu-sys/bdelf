"""Checkpoint run identity: config hash → ``cache/checkpoints/{variant}/{kind}/{model}/{hash}/``.

不允许别名 / 软链；路径唯一由训练入参与所解析 YAML 内容决定。
``world_size`` / 派生 accum / 微步间隔 / GPU 硬件规格不进指纹
（硬件另行写入 run 目录 ``hardware.json`` 并在续跑时校验）。
旧布局 ``{variant}/{model}/{hash}/`` 仍可读（generate / 续训兼容）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from models import kind_of, resolve_model_config_path

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
    """``asdict`` 后合并 ``extra``（去掉 ``_`` 键、冗余 ``name`` 与 ``_HASH_EXCLUDE``）。"""
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
    for key in getattr(type(obj), "_HASH_EXCLUDE", ()):
        merged.pop(key, None)
    return _strip_meta(merged)


def _fingerprint_overrides(
    overrides: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """``--set`` 进指纹；``eval.skip`` 不进哈希，从 overrides 里剥掉。"""
    ov = _strip_meta(dict(overrides or {}))
    eval_ov = ov.get("eval")
    if isinstance(eval_ov, dict):
        eval_ov = {k: v for k, v in eval_ov.items() if k != "skip"}
        if eval_ov:
            ov["eval"] = eval_ov
        else:
            ov.pop("eval", None)
    return ov


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

    out = {
        "model": model,
        "model_config": model_config,
        "variant": variant,
        "dataset": dataset,
        "preprocess": preprocess,
        "generate": generate,
        "overrides": _fingerprint_overrides(overrides),
        "optimizer": dataclass_fingerprint(optimizer),
        "batch": dataclass_fingerprint(batch),
        "schedule": dataclass_fingerprint(schedule),
        "eval": dataclass_fingerprint(eval_cfg),
        "generate_cfg": _strip_meta(gen_piece),
        "model_arch": model_arch,
        "preprocess_cfg": preprocess_yaml,
        "dataset_cfg": dataset_yaml,
    }
    if model in ("belf", "relf"):
        cur_name = preprocess_yaml.get("curriculum")
        if cur_name:
            cur = _load_yaml_mapping(
                repo / "config" / "train" / "curriculum" / f"{cur_name}.yaml",
            )
            out["curriculum_cfg"] = cur
            # 课程只记数据源名字；真正切分在 owt-seg512 / owt-bucket 正文。
            data_cfg: dict[str, Any] = {}
            for key in ("seg512_preprocess", "bucket_preprocess"):
                src = str(cur.get(key) or "").strip()
                if not src:
                    continue
                data_cfg[key] = _load_yaml_mapping(
                    repo / "config" / "preprocess" / f"{src}.yaml",
                )
            if data_cfg:
                out["curriculum_data_cfg"] = data_cfg
    return out


def config_hash_from_fingerprint(fingerprint: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(fingerprint).encode("utf-8")).hexdigest()
    return digest[:CONFIG_HASH_LEN]


def run_dir_for(
    *,
    variant: str,
    model: str,
    config_hash: str,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
    kind: str | None = None,
) -> Path:
    """``{root}/{fast|full}/{kind}/{model}/{hash}/``；禁止别名。"""
    if variant not in ("fast", "full"):
        raise ValueError(f"variant must be fast|full, got {variant!r}")
    if not config_hash or any(c in config_hash for c in "/\\"):
        raise ValueError(f"invalid config_hash: {config_hash!r}")
    model_kind = kind or kind_of(model)
    return Path(checkpoint_root) / variant / model_kind / model / config_hash


def run_relpath(*, variant: str, model: str, config_hash: str) -> str:
    """相对 ``checkpoint_root`` 的路径字符串（generate ``--run`` 用）。"""
    return f"{variant}/{kind_of(model)}/{model}/{config_hash}"


def legacy_run_dir_for(
    *,
    variant: str,
    model: str,
    config_hash: str,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> Path:
    """旧布局 ``{variant}/{model}/{hash}/``（只读兼容）。"""
    return Path(checkpoint_root) / variant / model / config_hash


def resolve_run_dir(
    *,
    variant: str,
    model: str,
    config_hash: str,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> Path:
    """优先新布局；不存在时回退旧布局。"""
    root = Path(checkpoint_root)
    primary = run_dir_for(
        variant=variant,
        model=model,
        config_hash=config_hash,
        checkpoint_root=root,
    )
    if primary.is_dir():
        return primary
    legacy = legacy_run_dir_for(
        variant=variant,
        model=model,
        config_hash=config_hash,
        checkpoint_root=root,
    )
    if legacy.is_dir():
        return legacy
    return primary


def parse_checkpoint_run_relpath(run: str) -> tuple[str, str, str]:
    """Parse ``--run`` into ``(variant, model, config_hash)``.

    Accepts legacy ``{variant}/{model}/{hash}`` and
    ``{variant}/{kind}/{model}/{hash}``.
    """
    parts = Path(run).parts
    if len(parts) == 3 and parts[0] in ("fast", "full"):
        return parts[0], parts[1], parts[2]
    if len(parts) == 4 and parts[0] in ("fast", "full") and parts[1] in ("lm", "latent"):
        return parts[0], parts[2], parts[3]
    raise ValueError(
        f"invalid run relpath {run!r}; expected "
        "{{fast|full}}/{{model}}/{{hash}} or "
        "{{fast|full}}/{{lm|latent}}/{{model}}/{{hash}}"
    )


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
    return resolve_run_dir(
        variant=cfg.variant,
        model=cfg.model,
        config_hash=cfg.name,
        checkpoint_root=cfg.checkpoint_root,
    )
