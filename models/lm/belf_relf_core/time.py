"""logit-normal 分位梯子：训练与推理共用确定档位 ``{L_i}``。"""

from __future__ import annotations

import math

import torch


def check_time_step(T: int) -> None:
    """``time_step`` 须 ``T>=4``（``T`` 次流；无 decode 跳）。"""
    if int(T) < 4:
        raise ValueError(f"time_step T 须 >= 4，收到 {T}")


def _logit_normal_quantile(
    p: torch.Tensor,
    p_mean: float,
    p_std: float,
) -> torch.Tensor:
    """开区间 ``Q(p)=σ(P_m + P_s Φ^{-1}(p))``，``p∈(0,1)``。"""
    # Φ^{-1}(p) = √2 erfinv(2p-1)
    inv_cdf = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
    return torch.sigmoid(p_mean + p_std * inv_cdf)


def ladder_levels(
    T: int,
    p_mean: float,
    p_std: float,
    eps: float,
) -> torch.Tensor:
    """``L_i = Q(i/T)``，长度 ``T+1``（``i=0,…,T``）。

    ``Q(0)=0``，``Q(1)=1-eps``；开区间分位再夹到 ``[0, 1-eps]``。
    训练 / 推理的 ``G`` 只落在 ``L_0…L_{T-1}``；``L_T`` 仅作末次 Euler 终点。
    """
    check_time_step(T)
    if not (0.0 < float(eps) < 1.0):
        raise ValueError(f"t_clean_eps 须在 (0, 1)，收到 {eps}")
    if float(p_std) <= 0.0:
        raise ValueError(f"denoiser p_std 须 > 0，收到 {p_std}")
    t_int = int(T)
    cap = 1.0 - float(eps)
    out = torch.empty(t_int + 1, dtype=torch.float64)
    out[0] = 0.0
    out[-1] = cap
    if t_int > 1:
        i = torch.arange(1, t_int, dtype=torch.float64)
        p = i / float(t_int)
        q = _logit_normal_quantile(p, float(p_mean), float(p_std))
        out[1:-1] = q.clamp(0.0, cap)
    return out
