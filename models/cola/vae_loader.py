"""Resolve and load Stage-1 Cola VAE checkpoints for Stage-2 training.

选用权重放在 ``cache/checkpoints/artifacts/cola_vae/<tag>/``（人手从训练
run 拷贝）；训练产物仍在 ``fast|full/cola_vae/<hash>/``，默认**不再**按
mtime 扫训练目录。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from models.cola.state_dict import remap_cola_mlp_keys
from models.cola_vae import FL_ColaVAEConfig, build_model_from_config as build_vae
from models.model import ensure_token_layout
from train import CHECKPOINT_ROOT


def artifacts_vae_root(
    *,
    vae_model: str = "cola_vae",
    checkpoint_root: str | Path | None = None,
) -> Path:
    """``cache/checkpoints/artifacts/<vae_model>``。"""
    root = Path(checkpoint_root or CHECKPOINT_ROOT)
    return root / "artifacts" / vae_model


def _checkpoint_in_dir(dir_path: Path) -> Path | None:
    ckpt = dir_path / "checkpoint_latest.pt"
    return ckpt if ckpt.is_file() else None


def _resolve_artifacts_tag(
    tag: str,
    *,
    vae_model: str,
    root: Path,
) -> Path:
    if not tag or any(c in tag for c in "/\\"):
        raise ValueError(
            f"VAE tag must be a single path segment (no slashes), got {tag!r}"
        )
    path = _checkpoint_in_dir(artifacts_vae_root(vae_model=vae_model, checkpoint_root=root) / tag)
    if path is None:
        raise FileNotFoundError(
            f"VAE artifact not found: artifacts/{vae_model}/{tag}/checkpoint_latest.pt. "
            f"Copy a trained run, e.g. "
            f"cp -a {root}/full/{vae_model}/<hash> "
            f"{root}/artifacts/{vae_model}/{tag}"
        )
    return path


def resolve_vae_checkpoint(
    *,
    vae_model: str,
    vae_size: str,
    variant: str | None = None,
    vae_run: str | None = None,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Resolve VAE ``checkpoint_latest.pt``.

    Order:
    1. ``COLA_VAE_CHECKPOINT`` env（显式文件）
    2. ``COLA_VAE_TAG`` env → ``artifacts/<vae_model>/<tag>/``
    3. ``vae_run``：若含 ``/`` 或 ``artifacts|fast|full`` 前缀则相对
       ``checkpoint_root``；否则当作 artifacts tag
    4. ``artifacts/<vae_model>/`` 下恰好一个带 ``checkpoint_latest.pt`` 的 tag
    不再默认扫 ``fast|full/<vae_model>/<hash>/``（训练目录需显式 env / 路径）。
    """
    _ = vae_size, variant  # 保留签名兼容；分辨率不参与路径
    env = os.environ.get("COLA_VAE_CHECKPOINT")
    if env:
        path = Path(env)
        if not path.is_file():
            raise FileNotFoundError(f"COLA_VAE_CHECKPOINT not found: {path}")
        return path

    root = Path(checkpoint_root or CHECKPOINT_ROOT)
    tag_env = os.environ.get("COLA_VAE_TAG")
    if tag_env:
        return _resolve_artifacts_tag(tag_env.strip(), vae_model=vae_model, root=root)

    if vae_run:
        spec = str(vae_run).strip().strip("/")
        if "/" in spec or spec.startswith(("artifacts", "fast", "full")):
            path = root / spec / "checkpoint_latest.pt"
            if not path.is_file():
                raise FileNotFoundError(f"VAE checkpoint not found: {path}")
            return path
        return _resolve_artifacts_tag(spec, vae_model=vae_model, root=root)

    art = artifacts_vae_root(vae_model=vae_model, checkpoint_root=root)
    candidates: list[Path] = []
    if art.is_dir():
        for child in sorted(art.iterdir()):
            if not child.is_dir():
                continue
            ckpt = _checkpoint_in_dir(child)
            if ckpt is not None:
                candidates.append(ckpt)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        tags = ", ".join(p.parent.name for p in candidates)
        raise FileNotFoundError(
            f"Multiple VAE artifacts under {art}/ ({tags}). "
            "Set COLA_VAE_TAG=<tag> or config vae_run=<tag> "
            "(or COLA_VAE_CHECKPOINT=/path/to/checkpoint_latest.pt)."
        )
    raise FileNotFoundError(
        f"No VAE under {art}/<tag>/checkpoint_latest.pt. "
        f"Train {vae_model} under fast|full/, then copy, e.g.\n"
        f"  mkdir -p {art}/my-tag && "
        f"cp -a {root}/full/{vae_model}/<hash>/checkpoint_latest.pt "
        f"{art}/my-tag/\n"
        "Or set COLA_VAE_CHECKPOINT / COLA_VAE_TAG / vae_run."
    )


def load_vae_backbone(
    *,
    vae_model: str,
    vae_size: str,
    variant: str | None = None,
    vae_run: str | None = None,
    device: torch.device | str | None = None,
    checkpoint_root: str | Path | None = None,
):
    """Build VAE from its YAML and load weights from a resolved checkpoint."""
    from models.model import config_from_yaml, resolve_model_config_path

    ckpt_path = resolve_vae_checkpoint(
        vae_model=vae_model,
        vae_size=vae_size,
        variant=variant,
        vae_run=vae_run,
        checkpoint_root=checkpoint_root,
    )
    cfg_path = resolve_model_config_path(vae_model, vae_size)
    config = config_from_yaml(FL_ColaVAEConfig, cfg_path)
    ensure_token_layout(config)
    model = build_vae(config)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model") or payload.get("state_dict") or payload
    # Checkpoints wrap FL_PreTrainedModel → keys may be ``backbone.*``.
    if any(k.startswith("backbone.") for k in state):
        state = {
            (k[len("backbone.") :] if k.startswith("backbone.") else k): v
            for k, v in state.items()
        }
    state = remap_cola_mlp_keys(state)
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"VAE load missing keys from {ckpt_path}: {missing[:8]}..."
        )
    if device is not None:
        model.backbone.to(device)
    return model.backbone, ckpt_path
