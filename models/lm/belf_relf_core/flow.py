"""ELF 时间轴的 rectified flow：``t=1`` 干净、``t=0`` 纯噪。"""

from __future__ import annotations

import torch


def _expand_t(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 ``t`` 扩成与 ``ref`` 可广播：``(B,)`` / ``(B,L)`` / ``(B,L,1)``。"""
    if t.ndim == ref.ndim:
        return t
    if t.ndim == 1:
        return t.reshape(-1, *([1] * (ref.ndim - 1)))
    if t.ndim == 2 and ref.ndim == 3:
        return t.unsqueeze(-1)
    if t.ndim + 1 == ref.ndim:
        return t.unsqueeze(-1)
    raise ValueError(f"无法把 t 形状 {tuple(t.shape)} 扩到 {tuple(ref.shape)}")


def interpolate(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """线性插值 ``z = t*x0 + (1-t)*noise``。"""
    t_exp = _expand_t(t, x0)
    return t_exp * x0 + (1.0 - t_exp) * noise


def x_to_v(
    x_hat: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    vel_eps: float,
) -> torch.Tensor:
    """x-pred 换速度：``v = (x_hat-z) / max(1-t, vel_eps)``。"""
    t_exp = _expand_t(t, z)
    denom = (1.0 - t_exp).clamp(min=float(vel_eps))
    return (x_hat - z) / denom


def v_star(
    x0: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    vel_eps: float,
) -> torch.Tensor:
    """目标速度：与 ``x_to_v`` 相同公式，用干净端 ``x0``。"""
    return x_to_v(x0, z, t, vel_eps)
