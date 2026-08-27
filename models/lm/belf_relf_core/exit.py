"""出口：一层线性。``linear`` 映到 logits（ELF）；``decoder`` 映到 latent X（再走 VAE-dec，Cola）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ExitMap(nn.Module):
    """规格 Exit：只有一层 ``Linear(D → out)``。"""

    def __init__(
        self,
        *,
        kind: str,
        n_embd: int,
        out_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        kind = str(kind).strip().lower()
        if kind not in ("decoder", "linear"):
            raise ValueError(f"exit 须为 decoder|linear，收到 {kind!r}")
        self.kind = kind
        self.proj = nn.Linear(int(n_embd), int(out_dim), bias=bool(bias))
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# 旧名，避免外部引用断裂。
CausalExit = ExitMap
