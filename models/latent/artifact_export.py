"""把训练 run 的 latest 导出为 ``artifacts/latent/{model}/{tag}/`` 推理权重。

只写最终模型参数（有 EMA 则熔进 ``model``，不再另存一份）和架构 ``config.json``。
不含优化器 / RNG / 训练超参 / hardware。导出后不可再续训。
加载仍走 ``artifact_loader.load_latent_artifact``（只读）。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from models.kinds import kind_of
from models.latent.artifact_loader import (
    CHECKPOINT_ROOT,
    _LATEST,
    _check_segment,
    _require_latent_model,
    resolve_artifact_dir,
)

_SIZE_RE = re.compile(r"(\d+m)\b", re.IGNORECASE)


def default_artifact_tag(model_cfg: dict[str, Any]) -> str:
    """``100m-b32-d1``：档位 + 瓶颈 B + 块大小 D。"""
    name = str(model_cfg.get("name") or "")
    m = _SIZE_RE.search(name)
    size = m.group(1).lower() if m else "unk"
    b = model_cfg.get("latent_dim")
    d = model_cfg.get("block_size", 1)
    if b is None:
        raise ValueError("model config 缺少 latent_dim，无法生成 tag")
    return f"{size}-b{int(b)}-d{int(d)}"


def _read_sidecar_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _model_meta_from_ckpt_or_sidecar(
    ck: dict[str, Any], run_dir: Path
) -> dict[str, Any]:
    meta = ck.get("model_meta") or {}
    if meta.get("name") and meta.get("config"):
        return {"name": meta["name"], "config": dict(meta["config"])}
    saved = _read_sidecar_config(run_dir).get("model") or {}
    if saved.get("name") and saved.get("config"):
        return {"name": saved["name"], "config": dict(saved["config"])}
    raise ValueError(
        f"{run_dir}: checkpoint 与 config.json 均无可用 model 架构参数"
    )


def _bake_final_weights(ck: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """live 权重上覆盖 EMA；buffers 保持 live。只保留最终一份。"""
    src = ck.get("model")
    if not isinstance(src, dict) or not src:
        raise ValueError("checkpoint 无 model 权重")
    out: dict[str, Any] = {}
    for key, value in src.items():
        if torch.is_tensor(value):
            out[key] = value.detach().contiguous().cpu()
        else:
            out[key] = value
    ema = ck.get("ema")
    baked = False
    if isinstance(ema, dict) and ema:
        for key, value in ema.items():
            if key in out and torch.is_tensor(value):
                out[key] = value.detach().contiguous().cpu()
                baked = True
    return out, baked


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _clean_artifact_dir(dest: Path) -> None:
    """目录里只留 checkpoint_latest.pt 与 config.json。"""
    keep = {_LATEST, "config.json"}
    if not dest.is_dir():
        return
    for child in dest.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            import shutil

            shutil.rmtree(child)
        else:
            child.unlink()


def resolve_source_checkpoint(
    *,
    run: str | None = None,
    checkpoint: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
) -> Path:
    root = Path(checkpoint_root or CHECKPOINT_ROOT)
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        return path
    if not run:
        raise ValueError("须指定 --run 或 --checkpoint")
    rel = str(run).strip().strip("/")
    path = root / rel / _LATEST
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    return path


def export_latent_artifact(
    *,
    run: str | None = None,
    checkpoint: str | Path | None = None,
    tag: str | None = None,
    checkpoint_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """从训练 latest 导出推理权重到 ``artifacts/latent/{model}/{tag}/``。

    有 EMA 则熔进 ``model`` 后丢弃影子权重；不写优化器 / 训练配置 / hardware。
    """
    src = resolve_source_checkpoint(
        run=run, checkpoint=checkpoint, checkpoint_root=checkpoint_root
    )
    run_dir = src.parent
    ck = torch.load(src, map_location="cpu", weights_only=False)
    model_meta = _model_meta_from_ckpt_or_sidecar(ck, run_dir)
    latent_model = _require_latent_model(str(model_meta["name"]))
    try:
        kind = kind_of(latent_model)
    except KeyError as exc:
        raise ValueError(f"未知模型 {latent_model!r}") from exc
    if kind != "latent":
        raise ValueError(f"{latent_model!r} 不是 latent 模型")

    model_cfg = dict(model_meta["config"] or {})
    tag_name = _check_segment(tag, what="tag") if tag else default_artifact_tag(model_cfg)
    dest = resolve_artifact_dir(
        latent_model, tag_name, checkpoint_root=checkpoint_root
    )
    dest_ckpt = dest / _LATEST
    if dest_ckpt.is_file() and not force:
        raise FileExistsError(
            f"已存在 {dest_ckpt}；覆盖请加 --force"
        )

    weights, ema_baked = _bake_final_weights(ck)
    del ck
    payload = {
        "model": weights,
        "model_meta": {"name": latent_model, "config": model_cfg},
        "ema_baked": ema_baked,
    }
    dest.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, dest_ckpt)
    config_json = {
        "model": {
            "name": latent_model,
            "config": model_cfg,
        }
    }
    (dest / "config.json").write_text(
        json.dumps(config_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _clean_artifact_dir(dest)
    return dest
