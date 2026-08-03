"""Resolve and load Stage-1 Cola VAE checkpoints for Stage-2 training."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from models.cola.state_dict import remap_cola_mlp_keys
from models.cola_vae import FL_ColaVAEConfig, build_model_from_config as build_vae
from models.model import ensure_token_layout
from train import CHECKPOINT_ROOT


def resolve_vae_checkpoint(
    *,
    vae_model: str,
    vae_size: str,
    variant: str | None = None,
    vae_run: str | None = None,
    checkpoint_root: str | Path | None = None,
) -> Path:
    """Resolve VAE ``checkpoint_latest.pt``.

    Order: ``COLA_VAE_CHECKPOINT`` env → ``vae_run``（相对 root 的
    ``{fast|full}/cola_vae/<hash>``）→ 在新布局下按 mtime 选最新
    ``{variant?}/cola_vae/*/checkpoint_latest.pt``。
    """
    env = os.environ.get("COLA_VAE_CHECKPOINT")
    if env:
        path = Path(env)
        if not path.is_file():
            raise FileNotFoundError(f"COLA_VAE_CHECKPOINT not found: {path}")
        return path

    root = Path(checkpoint_root or CHECKPOINT_ROOT)
    if vae_run:
        path = root / vae_run / "checkpoint_latest.pt"
        if not path.is_file():
            raise FileNotFoundError(f"VAE checkpoint not found: {path}")
        return path

    variants = [variant] if variant in ("fast", "full") else ["fast", "full"]
    candidates: list[Path] = []
    for var in variants:
        model_root = root / var / vae_model
        if not model_root.is_dir():
            continue
        for hash_dir in model_root.iterdir():
            ckpt = hash_dir / "checkpoint_latest.pt"
            if ckpt.is_file():
                candidates.append(ckpt)
    if not candidates:
        raise FileNotFoundError(
            f"No VAE checkpoint under {root}/{{fast,full}}/{vae_model}/<hash>/. "
            "Train cola_vae first or set vae_run / COLA_VAE_CHECKPOINT "
            "(use scripts/resolve_checkpoint.py for the hash path)."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
