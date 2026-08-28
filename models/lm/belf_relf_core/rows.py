"""按 batch 维切行 / 写回，供 BELF / RELF 子批 G。"""

from __future__ import annotations

from typing import Any

import torch

from models.lm.belf_relf_core.layers import LeftKVCache


def row_idx(flag: torch.Tensor) -> torch.Tensor:
    """``(B,)`` 布尔 → 选中行下标。"""
    return flag.nonzero(as_tuple=True)[0]


def take_rows(idx: torch.Tensor, *xs: Any) -> tuple[Any, ...]:
    """按 ``idx`` 切 batch 维；``None`` 原样，``LeftKVCache`` 切每层 K/V。"""
    out: list[Any] = []
    for x in xs:
        if x is None:
            out.append(None)
        elif isinstance(x, LeftKVCache):
            out.append(x.index_select(idx))
        else:
            out.append(x.index_select(0, idx))
    return tuple(out)
