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


def ema_absorb_new(
    ema_state: Dict[str, torch.Tensor],
    model: nn.Module,
) -> int:
    """把解冻后新出现的可训参数收进 EMA（初值=当前 live）。"""
    raw = unwrap_model(model)
    n = 0
    for name, param in raw.named_parameters():
        if not param.requires_grad or name in ema_state:
            continue
        ema_state[name] = param.detach().clone()
        n += 1
    return n


def ema_merge_loaded(
    ema_state: Dict[str, torch.Tensor],
    loaded_ema: Dict[str, torch.Tensor],
) -> int:
    """把 checkpoint 里的 EMA 键写入影子表，含启动时尚未可训、解冻后才有的键。"""
    n = 0
    for name, val in loaded_ema.items():
        if name in ema_state:
            ema_state[name].copy_(val.to(device=ema_state[name].device))
        else:
            ema_state[name] = val.detach().clone()
        n += 1
    return n


def ema_update(
    ema_state: Dict[str, torch.Tensor],
    model: nn.Module,
    decay: float,
) -> None:
    """In-place ``ema = decay * ema + (1 - decay) * param`` over trainable params."""
    if decay <= 0.0 or decay >= 1.0:
        raise ValueError(f"ema decay must be in (0, 1), got {decay}")
    ema_absorb_new(ema_state, model)
    raw = unwrap_model(model)
    for name, param in raw.named_parameters():
        if not param.requires_grad or name not in ema_state:
            continue
        ema_state[name].lerp_(param.detach(), 1.0 - decay)


def ema_weight_map(
    model: nn.Module,
    ema_state: Dict[str, torch.Tensor] | None,
) -> Dict[str, torch.Tensor] | None:
    """组装 EMA 权重表，**不** ``copy_`` 进已编译模块的 Parameter。

    给 ``using_ema_weights``（换模块字典、不 bump version）用。
    """
    if not ema_state:
        return None
    raw = unwrap_model(model)
    out: Dict[str, torch.Tensor] = {}
    for name, param in raw.named_parameters():
        src = ema_state.get(name)
        if src is None:
            out[name] = param
            continue
        if src.device != param.device or src.dtype != param.dtype:
            src = src.to(device=param.device, dtype=param.dtype, non_blocking=True)
        out[name] = src
    return out


def apply_ema_weights(
    model: nn.Module,
    ema_state: Dict[str, torch.Tensor] | None,
) -> bool:
    """把 EMA 影子权重拷进 ``model``（生成 / 推理用，不还原）。

    无 ``ema`` 或键对不上时返回 False，保留 live 权重。
    """
    if not ema_state:
        return False
    raw = unwrap_model(model)
    n = 0
    for name, param in raw.named_parameters():
        if name not in ema_state:
            continue
        param.data.copy_(
            ema_state[name].to(device=param.device, dtype=param.dtype)
        )
        n += 1
    return n > 0


@contextmanager
def using_ema_weights(
    model: nn.Module,
    ema_state: Dict[str, torch.Tensor] | None,
) -> Iterator[None]:
    """把 EMA 张量挂进模块字典，**不** ``copy_`` 进已捕获的 Parameter。

    给仍须 ``eval()`` 的 eager 路径（ELF PPL / gen / TrACE 估 d）：
    评测打在 unwrap 后的原模块上，训练 compile 图继续握着原来的
    Parameter 对象。结束后 ``_reparametrize_module`` 把字典换回去。
    """
    if not ema_state:
        yield
        return
    weight_map = ema_weight_map(model, ema_state)
    if not weight_map:
        yield
        return
    from torch.nn.utils.stateless import _reparametrize_module

    raw = unwrap_model(model)
    with _reparametrize_module(raw, weight_map, tie_weights=True, strict=False):
        yield


@contextmanager
def swap_ema_weights(
    model: nn.Module,
    ema_state: Dict[str, torch.Tensor] | None,
) -> Iterator[None]:
    """把 EMA 权重 ``copy_`` 进同一套 Parameter（编译图复用同一 storage）。

    2.13 在 dict-tag 快路径下 ``copy_`` 通常不重编。latent 在线评测要打
    已编译训练图时走这里；ELF / TrACE 请用 ``using_ema_weights``。
    """
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
