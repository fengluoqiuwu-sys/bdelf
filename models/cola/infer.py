"""块级 ODE 采样：对齐官方 ``generate_task_repaint_inference`` 的积分与 CFG。"""

from __future__ import annotations

from typing import Callable

import torch


@torch.no_grad()
def sample_block_ode(
    *,
    predict_v: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z_init: torch.Tensor,
    clean_prefix: torch.Tensor | None,
    num_steps: int,
    cfg_scale: float,
    ode_T: float = 1000.0,
) -> torch.Tensor:
    """从噪声积分一块 latent 到干净。

    时间轴与官方一致：``linspace(T, 0, steps+1)``，
    ``z ← z - v * (t_curr - t_next) / T``。
    CFG 无条件支路 = **仅当前块**（空前缀），不是把前缀置零。
    ``predict_v(z_full, t)`` 返回当前块速度；``t`` 为 ``(B,)`` 或 ``(B, L)``。
    """
    z = z_init
    block_len = z.size(1)
    bsz = z.size(0)
    device = z.device
    dtype = z.dtype
    t_grid = torch.linspace(float(ode_T), 0.0, num_steps + 1, device=device, dtype=torch.float32)
    has_prefix = clean_prefix is not None and clean_prefix.numel() > 0
    prefix_len = int(clean_prefix.size(1)) if has_prefix else 0
    use_cfg = float(cfg_scale) != 1.0 and has_prefix

    for i in range(len(t_grid) - 1):
        t_curr = t_grid[i]
        t_next = t_grid[i + 1]
        dt = (float(t_curr) - float(t_next)) / max(float(ode_T), 1.0)

        if has_prefix:
            z_full = torch.cat([clean_prefix, z], dim=1)
            t_full = torch.zeros(bsz, prefix_len + block_len, device=device, dtype=dtype)
            t_full[:, prefix_len:] = t_curr.to(dtype=dtype)
            v_cond = predict_v(z_full, t_full)
        else:
            t_batch = t_curr.to(dtype=dtype).expand(bsz)
            v_cond = predict_v(z, t_batch)

        if use_cfg:
            t_uncond = t_curr.to(dtype=dtype).expand(bsz)
            v_uncond = predict_v(z, t_uncond)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        z = z - v * dt
    return z
