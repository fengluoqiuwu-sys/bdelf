"""DMA-H：日程化 embedding round-trip + straight-through（修 B / 假词轴）。

当 ``t >= dma_t0`` 时：用**同一前向**的 decode-head logits → argmax → 查 T5
token embedding 表回连续态；前向用硬化结果，反向对原预测做 ST（训练期 SC
反馈本身已 detach，ST 主要为将来带梯度路径预留）。

量化必须用 decode CE 训练的读出头 logits，禁止 ``x_pred @ unembed``（``x_pred``
来自另一头 ``final_layer``，与 unembed 无对齐）。

``dma_mode=gumbel`` 预留（T2b）；当前仅实现 ``round_trip``。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dma_gate(t: torch.Tensor, t0: float) -> torch.Tensor:
    """``(B,)`` 或标量 t → ``(B, 1, 1)`` 硬门控（``t >= t0``）。"""
    t_flat = t.reshape(-1).to(dtype=torch.float32)
    g = (t_flat >= float(t0)).to(dtype=t.dtype)
    return g.view(-1, 1, 1)


def embed_tokens_table(
    token_ids: torch.Tensor,
    *,
    embed_weight: torch.Tensor,
    latent_mean: float,
    latent_std: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """查冻结 T5 input embedding 表并按 ELF/ODAR latent 尺度归一化。"""
    emb = F.embedding(token_ids, embed_weight)
    scale = max(float(latent_std), 1e-8)
    return ((emb - float(latent_mean)) / scale).to(dtype=dtype)


def commit_mse(
    x: torch.Tensor,
    *,
    logits: torch.Tensor,
    embed_weight: torch.Tensor,
    latent_mean: float,
    latent_std: float,
    t: torch.Tensor,
    t0: float,
    loss_mask: torch.Tensor,
    extra_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """DMA-commit：相对 ‖x-sg(embed(argmax))‖² / ‖sg(embed)‖²，仅 ``t≥t0``。

    用相对尺度，使 λ≈0.1 与早期 mse 同量级；梯度只流向 ``x``。无触发时返回 0。
    """
    gate = dma_gate(t, t0)
    if extra_gate is not None:
        gate = gate * extra_gate.to(dtype=gate.dtype)
    # 避免 .item() 触发 dynamo graph break；无触发时乘 0 即可
    token_ids = logits.argmax(dim=-1)
    x_hard = embed_tokens_table(
        token_ids,
        embed_weight=embed_weight,
        latent_mean=latent_mean,
        latent_std=latent_std,
        dtype=x.dtype,
    ).detach()
    per_token = ((x - x_hard) ** 2).mean(dim=-1)
    hard_norm = (x_hard ** 2).mean(dim=-1).clamp(min=1e-4)
    per_token = per_token / hard_norm
    mask = loss_mask.to(dtype=per_token.dtype) * gate.squeeze(-1)
    denom = torch.clamp(mask.sum(), min=1.0)
    # gate 全 0 时仍返回有限 0（mask 和为 0 → denom=1 但分子为 0）
    return (per_token * mask).sum() / denom


def round_trip_st(
    x: torch.Tensor,
    *,
    logits: torch.Tensor,
    embed_weight: torch.Tensor,
    latent_mean: float,
    latent_std: float,
    t: torch.Tensor,
    t0: float,
    mode: str = "round_trip",
    tau: float = 1.0,
    extra_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """对 ``x`` 做 DMA-H；``t < t0`` 的样本行原样返回。

    Args:
        x: ``(B, S, C)`` 预测连续态。
        logits: ``(B, S, V)`` 与 ``x`` **同一次** ``net_forward`` 的 decode-head 输出。
        t: ``(B,)`` 或可广播到 batch 的时间。
        t0: 硬化起始时间。
        mode: ``round_trip``（argmax）或 ``gumbel``（未实现）。
        tau: Gumbel 温度（预留）。
        extra_gate: 可选 ``(B,1,1)`` 额外掩码（如非 decode 行）。
    """
    del tau  # gumbel 预留
    mode_l = str(mode).lower().strip()
    if mode_l != "round_trip":
        raise NotImplementedError(
            f"dma_mode={mode!r} 尚未实现；当前仅支持 'round_trip'（T2b 再补 gumbel）"
        )

    gate = dma_gate(t, t0)
    if extra_gate is not None:
        gate = gate * extra_gate.to(dtype=gate.dtype)

    # 全 batch 未触发时跳过，避免无谓 embedding 查表
    if float(gate.max().item()) <= 0.0:
        return x

    token_ids = logits.argmax(dim=-1)
    x_hard = embed_tokens_table(
        token_ids,
        embed_weight=embed_weight,
        latent_mean=latent_mean,
        latent_std=latent_std,
        dtype=x.dtype,
    )
    # ST：前向 = x_hard，反向梯度流向 x（反馈路径通常已 detach）
    x_st = x + (x_hard - x).detach()
    return x + gate * (x_st - x)
