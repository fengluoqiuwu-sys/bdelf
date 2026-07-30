"""EMA shadow weights (aligned with official ELF ``ema_decay1=0.9999``)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import torch
import torch.nn as nn

from train.checkpoint import unwrap_model


def init_ema(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone trainable parameters into a detached EMA shadow dict."""
    raw = unwrap_model(model)
    return {
        name: param.detach().clone()
        for name, param in raw.named_parameters()
        if param.requires_grad
    }


def ema_update(
    ema_state: Dict[str, torch.Tensor],
    model: nn.Module,
    decay: float,
) -> None:
    """In-place ``ema = decay * ema + (1 - decay) * param`` over trainable params."""
    if decay <= 0.0 or decay >= 1.0:
        raise ValueError(f"ema decay must be in (0, 1), got {decay}")
    raw = unwrap_model(model)
    for name, param in raw.named_parameters():
        if not param.requires_grad or name not in ema_state:
            continue
        ema_state[name].lerp_(param.detach(), 1.0 - decay)


@contextmanager
def swap_ema_weights(
    model: nn.Module,
    ema_state: Dict[str, torch.Tensor] | None,
) -> Iterator[None]:
    """Temporarily copy EMA weights into ``model`` for eval / generation."""
    if not ema_state:
        yield
        return
    raw = unwrap_model(model)
    backup: Dict[str, torch.Tensor] = {}
    try:
        for name, param in raw.named_parameters():
            if name not in ema_state:
                continue
            backup[name] = param.detach().clone()
            param.data.copy_(
                ema_state[name].to(device=param.device, dtype=param.dtype)
            )
        yield
    finally:
        for name, param in raw.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
