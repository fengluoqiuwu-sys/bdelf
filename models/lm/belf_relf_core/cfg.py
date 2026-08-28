"""self-cond CFG：``w_sc`` 采样、速度靶混合、训练期丢掉左段。"""

from __future__ import annotations

import torch
import torch.nn as nn


def self_left_p(*, tokens_seen: int, thaw_tokens: int, prob: float) -> float:
    """在 compile 图外根据累计 token 给出 self-left 概率。

    Dynamo 把 ``nn.Module`` 上的 int 当静态量；若在 ``forward`` 里读每步更新的
    ``_tokens_seen`` 会每步重编译，DDP 下打到 ``recompile_limit`` 即死锁。
    """
    if int(thaw_tokens) <= 0 or int(tokens_seen) >= int(thaw_tokens):
        return float(prob)
    return 0.0


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
    if float(p_std) < 0.0:
        raise ValueError(f"w_sc p_std 须 >= 0，收到 {p_std}")
    if float(w_min) <= 0.0 or float(w_max) <= float(w_min):
        raise ValueError(
            f"w_sc 须 0 < w_min < w_max，收到 w_min={w_min}, w_max={w_max}"
        )
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


def maybe_drop_left(
    h_left: torch.Tensor,
    drop_prob: float,
    *,
    return_drop: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """以 ``drop_prob`` 按样本把左段置 0（训练期无条件上下文）。

    ``return_drop=True`` 时另返回 ``(B,)`` 布尔：该样本是否丢掉左段，供切断注意力。
    """
    p = float(drop_prob)
    bsz = h_left.size(0)
    if p <= 0.0:
        drop = torch.zeros(bsz, device=h_left.device, dtype=torch.bool)
        return (h_left, drop) if return_drop else h_left
    drop = torch.rand(bsz, device=h_left.device, dtype=torch.float32) < p
    keep = (~drop).to(dtype=h_left.dtype).reshape(bsz, *([1] * (h_left.ndim - 1)))
    out = h_left * keep
    return (out, drop) if return_drop else out


def pad_after_first_eos(
    tokens: torch.Tensor,
    eos_id: int,
    pad_id: int,
) -> torch.Tensor:
    """保留首个 EOS，其后写成 PAD，避免停机后仍留下采样词或凑长 PAD。"""
    is_eos = tokens == int(eos_id)
    seen = is_eos.to(torch.int32).cumsum(dim=1)
    keep = (seen == 0) | (is_eos & (seen == 1))
    return torch.where(keep, tokens, tokens.new_full((), int(pad_id)))


def hide_left_keys(
    attn_mask: torch.Tensor,
    drop: torch.Tensor,
    left_len: int,
) -> torch.Tensor:
    """``drop`` 为 ``(B,)``：丢掉左段的样本，右段 query 看不见左段 key。"""
    if left_len <= 0 or drop.numel() == 0:
        return attn_mask
    bsz = int(drop.size(0))
    mask = attn_mask
    if mask.size(0) == 1 and bsz > 1:
        mask = mask.expand(bsz, *([-1] * (mask.ndim - 1))).clone()
    else:
        mask = mask.clone()
    # 不用 drop.any()：避免热路径 GPU 同步。全假时这步是空写。
    mask[drop, :, left_len:, :left_len] = float("-inf")
    return mask
