"""self-cond CFG：``w_sc`` 采样、速度靶混合、训练期丢掉左段。"""

from __future__ import annotations

import torch
import torch.nn as nn


def keep_params_in_graph(module: nn.Module, like: torch.Tensor) -> torch.Tensor:
    """把 ``module`` 参数留在图里（空 CE 支路 / 单卡也安全），不扫全部元素。"""
    acc = like.new_zeros(())
    for p in module.parameters():
        acc = acc + p.reshape(-1)[0] * 0.0
    return acc


def sample_w_sc(
    batch: int,
    p_mean: float,
    p_std: float,
    w_min: float,
    w_max: float,
    device: torch.device | str,
) -> torch.Tensor:
    """logit-normal 再映到 ``[w_min, w_max]``（ELF ``sample_cfg_scale`` 的仿射）。

    ``z~N(p_mean, p_std^2)``，``u=sigmoid(z)``，
    ``a=1+w_min``，``b=1+w_max``，``w=a*(b/a)^u-1``。
    """
    z = torch.randn(int(batch), device=device, dtype=torch.float32)
    z = z * float(p_std) + float(p_mean)
    u = torch.sigmoid(z)
    a = 1.0 + float(w_min)
    b = 1.0 + float(w_max)
    return a * torch.pow(torch.as_tensor(b / a, device=device, dtype=u.dtype), u) - 1.0


def _bcast_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    while x.ndim < ref.ndim:
        x = x.unsqueeze(-1)
    return x.to(dtype=ref.dtype)


def blend_v_tgt(
    v_z: torch.Tensor,
    v_u: torch.Tensor,
    v_c: torch.Tensor,
    w: torch.Tensor,
    guided: torch.Tensor | bool,
) -> torch.Tensor:
    """guided 时 ``v_z+(1-1/w)*(v_c-v_u)``，否则 ``v_z``。"""
    if isinstance(guided, bool):
        if not guided:
            return v_z
        scale = 1.0 - 1.0 / w
        return v_z + _bcast_like(scale, v_z) * (v_c - v_u)
    g = guided.to(dtype=v_z.dtype)
    scale = (1.0 - 1.0 / w) * g
    return v_z + _bcast_like(scale, v_z) * (v_c - v_u)


def maybe_drop_left(h_left: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """以 ``drop_prob`` 按样本把左段置 0（训练期无条件上下文）。"""
    p = float(drop_prob)
    if p <= 0.0:
        return h_left
    bsz = h_left.size(0)
    drop = torch.rand(bsz, device=h_left.device, dtype=torch.float32) < p
    keep = (~drop).to(dtype=h_left.dtype).reshape(bsz, *([1] * (h_left.ndim - 1)))
    return h_left * keep
