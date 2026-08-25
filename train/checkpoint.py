"""Checkpoint save/load and model unwrap helpers."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from train import FL_TrainConfig


def unwrap_model(model: nn.Module) -> nn.Module:
    """剥离 DDP 与 torch.compile 包装,拿到原始模块。

    torch.compile 返回的 OptimizedModule 通过 _orig_mod 暴露原模块,DDP 通过 module 暴露。
    先 compile 再 DDP 会嵌套两层,故迭代解包直到没有包装为止。
    """
    m = model
    while True:
        if isinstance(m, DDP):
            m = m.module
        elif hasattr(m, "_orig_mod"):
            m = m._orig_mod
        else:
            return m


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: FL_TrainConfig,
    model_meta: dict[str, Any],
    *,
    ema_state: dict[str, torch.Tensor] | None = None,
    curriculum_state: dict[str, Any] | None = None,
) -> None:
    """Atomically write a checkpoint (tmp file + ``os.replace``).

    Direct ``torch.save`` to the final path can leave a truncated file if the
    process is killed mid-write; resume would then fail on a corrupt latest ckpt.
    """
    path = Path(path)
    from models.latent.artifact_loader import assert_not_artifacts_latent_path

    assert_not_artifacts_latent_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = unwrap_model(model)
    grads = [
        p.grad.detach().cpu() if p.grad is not None else None
        for p in raw.parameters()
    ]
    payload: dict[str, Any] = {
        "step": step,
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "grads": grads,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        },
        "train_config": asdict(cfg),
        "model_meta": model_meta,
    }
    if ema_state:
        payload["ema"] = {k: v.detach().cpu() for k, v in ema_state.items()}
    if curriculum_state:
        payload["curriculum"] = dict(curriculum_state)
    if torch.cuda.is_available():
        payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer | Any,
    device: torch.device,
) -> None:
    if hasattr(optimizer, "muon") and hasattr(optimizer, "adamw"):
        _move_optimizer_state_to_device(optimizer.muon, device)
        _move_optimizer_state_to_device(optimizer.adamw, device)
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    cfg: FL_TrainConfig,
    model_meta: dict[str, Any],
    restore_rng: bool = True,
) -> tuple[int, dict[str, torch.Tensor] | None, dict[str, Any] | None]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    saved_cfg = ck.get("train_config") or {}
    if saved_cfg.get("name") and saved_cfg["name"] != cfg.name:
        raise ValueError(
            f"checkpoint run={saved_cfg['name']!r} does not match current {cfg.name!r}"
        )
    saved_use_muon = bool(saved_cfg.get("use_muon", False))
    if saved_use_muon != cfg.use_muon:
        raise ValueError(
            f"checkpoint use_muon={saved_use_muon} does not match current {cfg.use_muon}"
        )
    saved_meta = ck.get("model_meta") or {}
    if saved_meta.get("name") and saved_meta["name"] != model_meta.get("name"):
        raise ValueError(
            f"checkpoint model {saved_meta.get('name')!r} "
            f"does not match current {model_meta.get('name')!r}"
        )

    raw = unwrap_model(model)
    raw.load_state_dict(ck["model"])
    opt_state = ck["optimizer"]
    if cfg.use_muon:
        if not isinstance(opt_state, dict) or opt_state.get("kind") != "muon_hybrid":
            raise ValueError(
                "checkpoint optimizer is not hybrid Muon state; "
                "cannot resume with use_muon=True"
            )
    elif isinstance(opt_state, dict) and opt_state.get("kind") == "muon_hybrid":
        raise ValueError(
            "checkpoint optimizer is hybrid Muon state; "
            "cannot resume with use_muon=False"
        )
    optimizer.load_state_dict(opt_state)
    _move_optimizer_state_to_device(optimizer, device)

    grads = ck.get("grads")
    if grads is not None:
        for p, g in zip(raw.parameters(), grads):
            p.grad = g.to(device) if g is not None else None

    ema_raw = ck.get("ema")
    ema_state: dict[str, torch.Tensor] | None = None
    if isinstance(ema_raw, dict) and ema_raw:
        ema_state = {
            k: v.to(device=device) for k, v in ema_raw.items()
        }

    # The RNG snapshot comes from rank 0; non-zero ranks keep their
    # set_seed(seed + rank) state so per-rank noise stays decorrelated.
    rng = ck.get("rng")
    if restore_rng and rng is not None:
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return int(ck["step"]), ema_state, ck.get("curriculum")


def load_init_weights(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    """从另一 run 的 ckpt **只加载权重**（及可选 EMA），不恢复优化器 / step / RNG。

    允许源 ``model`` 名不同（如 ELF → TrACE）；``strict=False`` 以容纳
    新增 buffer（``attr_d`` 等）。训练从 step 0 开始，写入新 hash 目录。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"init-ckpt 不存在: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    raw = unwrap_model(model)
    src = ck.get("model")
    if not isinstance(src, dict) or not src:
        raise ValueError(f"{path}: checkpoint 无 model 权重")
    dst_keys = set(raw.state_dict().keys())
    n_overlap = sum(1 for k in src if k in dst_keys)
    if n_overlap < 1:
        raise ValueError(
            f"{path}: 与当前模型没有任何同名权重 "
            f"(src={len(src)} keys, dst={len(dst_keys)} keys)"
        )
    incompatible = raw.load_state_dict(src, strict=False)
    saved_cfg = ck.get("train_config") or {}
    saved_meta = ck.get("model_meta") or {}
    info: dict[str, Any] = {
        "path": str(path),
        "src_step": int(ck.get("step", -1)),
        "src_run": saved_cfg.get("name"),
        "src_model": saved_meta.get("name"),
        "n_src": len(src),
        "n_overlap": n_overlap,
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }
    ema_raw = ck.get("ema")
    ema_loaded: dict[str, torch.Tensor] | None = None
    if isinstance(ema_raw, dict) and ema_raw:
        ema_loaded = {k: v.to(device=device) for k, v in ema_raw.items()}
    return ema_loaded, info
