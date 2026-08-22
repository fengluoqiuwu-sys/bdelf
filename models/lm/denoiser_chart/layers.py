"""DenoiserChart layers: shared ELF primitives + local ``DenoiserChartWarp``."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.lm.elf_core.layers import *  # noqa: F403
from models.lm.elf_core.layers import TimestepEmbedder, make_linear


class DenoiserChartWarp(nn.Module):
    """Bottleneck 上仅 denoise 可见的对角仿射 W_t(h)=exp(alpha(t))*h+beta(t)。

    输出层零初始化：开训 gamma=1、beta=0，前向等于未 warp。
    """

    def __init__(self, bottleneck_dim: int, freq_size: int = 256) -> None:
        super().__init__()
        self.freq_size = freq_size
        self.to_ab = make_linear(freq_size, 2 * bottleneck_dim, bias=True)
        nn.init.zeros_(self.to_ab.weight)
        nn.init.zeros_(self.to_ab.bias)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # h: (B, S, d_b)；t: (B,)
        t_freq = TimestepEmbedder.timestep_embedding(t, self.freq_size)
        alpha, beta = self.to_ab(t_freq.to(dtype=h.dtype)).chunk(2, dim=-1)
        gamma = torch.exp(alpha).unsqueeze(1)
        return gamma * h + beta.unsqueeze(1)
