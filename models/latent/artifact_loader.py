"""按 ``latent_model`` + ``tag`` 只读加载 ``artifacts`` 选用权重。

布局::

    cache/checkpoints/artifacts/latent/{latent_model}/{tag}/checkpoint_latest.pt

只加载架构参数（``config.json`` / ``model_meta``）与模型权重（及可选 EMA / 白化 buffer），
**不**恢复优化器 / RNG，也**禁止**经本模块写回 ``artifacts/latent/``。
训练产物仍在 ``fast|full/latent/{model}/{hash}/``；选用目录由人手拷贝。
白化 ``m,s`` 由导出后离线写入（``scripts/compute_latent_whiten.py``），不进 VAE 训练。
ACE / cola_vae 等其它 ``artifacts/<kind>/`` 不受影响。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.kinds import kind_of, list_models
from models.model import build_model

# 与 train.run_path.CHECKPOINT_ROOT 对齐（避免 models→train 循环导入）
CHECKPOINT_ROOT = "cache/checkpoints"

_ARTIFACTS_DIR = "artifacts"
_LATENT_DIR = "latent"
_LATEST = "checkpoint_latest.pt"
WHITEN_JSON = "whiten.json"
# 写入 artifact 的逐维 μ 仿射；不进 VAE 训练模块，加载时再挂到 backbone。
WHITEN_BUFFER_NAMES = frozenset(
    {
        "whitening_mean",
        "whitening_std",
        "whitening_mean_z",
        "whitening_std_z",
    }
)


def artifacts_latent_root(*, checkpoint_root: str | Path | None = None) -> Path:
    """``cache/checkpoints/artifacts/latent``。"""
    root = Path(checkpoint_root or CHECKPOINT_ROOT)
    return root / _ARTIFACTS_DIR / _LATENT_DIR


def is_artifacts_latent_path(path: str | Path) -> bool:
    """是否落在 ``.../artifacts/latent/...``（不含 ace / cola_vae 等其它 artifacts）。"""
    parts = Path(path).parts
    try:
        i = parts.index(_ARTIFACTS_DIR)
    except ValueError:
        return False
    return i + 1 < len(parts) and parts[i + 1] == _LATENT_DIR


def _check_segment(value: str, *, what: str) -> str:
    name = str(value).strip()
    if not name or any(c in name for c in "/\\") or name in (".", ".."):
        raise ValueError(f"{what} 须为单段目录名（不含斜杠），收到 {value!r}")
    return name


def _require_latent_model(latent_model: str) -> str:
    name = _check_segment(latent_model, what="latent_model")
    try:
        kind = kind_of(name)
    except KeyError as exc:
        known = ", ".join(list_models(kind="latent"))
        raise ValueError(
            f"未知 latent 模型 {name!r}；已注册: {known or '(无)'}"
        ) from exc
    if kind != "latent":
        raise ValueError(f"{name!r} 的 kind={kind!r}，artifacts 加载器只接受 latent 模型")
    return name


def resolve_artifact_dir(
    latent_model: str,
    tag: str,
    *,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """解析 ``artifacts/latent/{model}/{tag}/``，不创建目录。"""
    model = _require_latent_model(latent_model)
    tag_name = _check_segment(tag, what="tag")
    return artifacts_latent_root(checkpoint_root=checkpoint_root) / model / tag_name


def resolve_artifact_checkpoint(
    latent_model: str,
    tag: str,
    *,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """返回选用目录下的 ``checkpoint_latest.pt``。"""
    run_dir = resolve_artifact_dir(
        latent_model, tag, checkpoint_root=checkpoint_root
    )
    path = run_dir / _LATEST
    if not path.is_file():
        raise FileNotFoundError(
            f"artifacts checkpoint 不存在: {path}。"
            f"请从训练 run 拷贝到 artifacts/latent/{latent_model}/{tag}/"
            f"（不要经本加载器写入）。"
        )
    return path


def list_artifact_tags(
    latent_model: str,
    *,
    checkpoint_root: str | Path | None = None,
) -> list[str]:
    """列出某模型下含 ``checkpoint_latest.pt`` 的 tag。"""
    model = _require_latent_model(latent_model)
    parent = artifacts_latent_root(checkpoint_root=checkpoint_root) / model
    if not parent.is_dir():
        return []
    tags: list[str] = []
    for child in sorted(parent.iterdir()):
        if child.is_dir() and (child / _LATEST).is_file():
            tags.append(child.name)
    return tags


def pop_whiten_state(weights: dict[str, Any]) -> dict[str, torch.Tensor]:
    """从 state_dict 取出白化 buffer，避免 ``load_state_dict`` 碰到未知键。"""
    found: dict[str, torch.Tensor] = {}
    for key in list(weights.keys()):
        tail = str(key).rsplit(".", 1)[-1]
        if tail not in WHITEN_BUFFER_NAMES:
            continue
        val = weights.pop(key)
        if torch.is_tensor(val):
            found[tail] = val
    return found


def attach_whiten_buffers(
    model: torch.nn.Module,
    buffers: dict[str, torch.Tensor],
) -> None:
    """把离线 μ 白化向量挂到 backbone（fp32），供 BELF/RELF 读取。"""
    if not buffers:
        return
    target = getattr(model, "backbone", model)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    for name, val in buffers.items():
        if name not in WHITEN_BUFFER_NAMES or not torch.is_tensor(val):
            continue
        tensor = val.detach().float().reshape(-1).contiguous().to(device=device)
        target.register_buffer(name, tensor, persistent=True)


def _load_model_meta(ckpt_path: Path, ck: dict[str, Any]) -> dict[str, Any]:
    meta = ck.get("model_meta") or {}
    if meta.get("name") and meta.get("config"):
        return meta

    config_json = ckpt_path.parent / "config.json"
    if config_json.is_file():
        with open(config_json, encoding="utf-8") as f:
            saved = json.load(f)
        model_meta = saved.get("model") or {}
        if model_meta.get("name") and model_meta.get("config"):
            return model_meta

    raise ValueError(
        f"{ckpt_path} 缺少 model_meta，且旁路 config.json 不可用"
    )


@dataclass(frozen=True)
class LatentArtifactLoad:
    """只读加载结果：模型 + 架构参数，不含优化器。"""

    model: torch.nn.Module
    model_meta: dict[str, Any]
    step: int
    train_cfg: dict[str, Any] | None
    used_ema: bool
    ckpt_path: Path
    latent_model: str
    tag: str


def load_latent_artifact(
    latent_model: str,
    tag: str,
    *,
    device: torch.device | str | None = None,
    apply_ema: bool = True,
    checkpoint_root: str | Path | None = None,
) -> LatentArtifactLoad:
    """按模型名与 tag 加载权重和架构参数。

    只读 ``artifacts/latent/``：不写文件、不创建目录、不保存更新。
    由 ``export_latent_artifact`` 写出的权重若已熔 EMA，则 ``used_ema=True``。
    """
    model_name = _require_latent_model(latent_model)
    tag_name = _check_segment(tag, what="tag")
    ckpt_path = resolve_artifact_checkpoint(
        model_name, tag_name, checkpoint_root=checkpoint_root
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_meta = _load_model_meta(ckpt_path, ck)
    saved_name = model_meta.get("name")
    if saved_name and saved_name != model_name:
        raise ValueError(
            f"tag={tag_name!r} 的 config 模型为 {saved_name!r}，"
            f"与 latent_model={model_name!r} 不一致"
        )
    model_cfg = dict(model_meta["config"] or {})
    model = build_model(model_name, model_cfg)
    weights = ck.get("model")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{ckpt_path}: checkpoint 无 model 权重")
    whiten_buf = pop_whiten_state(weights)
    model.load_state_dict(weights)
    model.eval()

    train_cfg = ck.get("train_config")
    step = int(ck.get("step", 0))
    ema_raw = ck.get("ema")
    ema_baked = bool(ck.get("ema_baked"))
    del ck

    if device is None:
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device)
    dtype_name = (train_cfg or {}).get("dtype", "bf16") if torch_device.type == "cuda" else "fp32"
    if dtype_name == "fp16":
        dtype = torch.float16
    elif dtype_name == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    model = model.to(device=torch_device, dtype=dtype)
    attach_whiten_buffers(model, whiten_buf)

    used_ema = ema_baked
    if apply_ema and isinstance(ema_raw, dict) and ema_raw:
        from train.ema import apply_ema_weights

        used_ema = apply_ema_weights(model, ema_raw) or used_ema
    return LatentArtifactLoad(
        model=model,
        model_meta=model_meta,
        step=step,
        train_cfg=train_cfg if isinstance(train_cfg, dict) else None,
        used_ema=used_ema,
        ckpt_path=ckpt_path,
        latent_model=model_name,
        tag=tag_name,
    )


def save_latent_artifact(*_args: Any, **_kwargs: Any) -> None:
    """显式拒绝写回 artifacts/latent（本加载器无保存能力）。"""
    raise RuntimeError(
        "禁止经 artifacts 加载器保存或更新 "
        "cache/checkpoints/artifacts/latent/。该目录只读；"
        "请用 scripts/export_latent_artifact.py 从训练 run 导出，"
        "再用 scripts/compute_latent_whiten.py 写入白化统计。"
    )


def assert_not_artifacts_latent_path(path: str | Path) -> None:
    """训练 / 通用保存入口用：目标落在 artifacts/latent/ 则拒绝。"""
    if is_artifacts_latent_path(path):
        save_latent_artifact()
