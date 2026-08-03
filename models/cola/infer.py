"""Block-wise ODE sampling for Cola latent prior (CFG + clean condition)."""

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
    t_eps: float,
    time_schedule: str,
    p_mean: float,
    p_std: float,
) -> torch.Tensor:
    """Integrate one latent block from noise to clean.

    ``predict_v(z_full, t_batch)`` returns velocity for the **current block**
    slice (last ``block_len`` positions of the conditioned sequence).
    ``z_full`` is ``cat(clean_prefix, z_block)`` with clean prefix held fixed
    (clean condition repaint).
    """
    device = z_init.device
    z = z_init
    block_len = z.size(1)
    steps = _time_grid(
        num_steps, device, time_schedule, p_mean, p_std,
    )
    for i in range(len(steps) - 1):
        t = steps[i]
        t_next = steps[i + 1]
        t_batch = t.expand(z.size(0))
        if clean_prefix is not None and clean_prefix.numel() > 0:
            z_full = torch.cat([clean_prefix, z], dim=1)
        else:
            z_full = z

        v_cond = predict_v(z_full, t_batch)
        if cfg_scale != 1.0 and clean_prefix is not None and clean_prefix.numel() > 0:
            # Uncond: zero prefix (drop condition) for CFG.
            z_uncond = torch.cat([torch.zeros_like(clean_prefix), z], dim=1)
            v_uncond = predict_v(z_uncond, t_batch)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        dt = t_next - t
        z = z + dt * v
        # Keep numerical stability near t→1
        if float(t_next) >= 1.0 - t_eps:
            break
    return z


def _time_grid(
    num_steps: int,
    device: torch.device,
    schedule: str,
    p_mean: float,
    p_std: float,
) -> torch.Tensor:
    if schedule == "logit_normal" and num_steps > 1:
        z = torch.randn(num_steps - 1, device=device) * p_std + p_mean
        interior = torch.sigmoid(z).sort().values
        return torch.cat(
            [torch.zeros(1, device=device), interior, torch.ones(1, device=device)],
        )
    return torch.linspace(0.0, 1.0, num_steps + 1, device=device)
