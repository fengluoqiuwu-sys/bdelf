"""TrACE 长训：每 1k opt-step 用 ACE Alg.1（N=128）估 ``d``，再 EMA 更新停梯度缓冲。

不在 ``train_step`` 里用训练 batch 重算 ``d``（无 seq-rep 标签）。
fast 变体默认跳过估 ``d``，避免本机冒烟被采样拖死。
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist

from models.lm.trace.ace import (
    collect_ace_sc_rep_pairs,
    direction_from_sc_rep_pairs,
)
from train.checkpoint import unwrap_model
from train.ema import swap_ema_weights


def _backbone(model: Any) -> Any:
    raw = unwrap_model(model)
    return getattr(raw, "backbone", raw)


def _shard(n: int, world_size: int, rank: int) -> tuple[int, int]:
    base, rem = divmod(int(n), int(world_size))
    extra = 1 if rank < rem else 0
    start = rank * base + min(rank, rem)
    return start, base + extra


def _all_gather_pairs(
    feats: torch.Tensor | None,
    reps: torch.Tensor | None,
    *,
    dim: int,
    device: torch.device,
    is_distributed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """各 rank 的 ``(n_i, e)`` / ``(n_i,)`` 对齐后 all_gather，再去掉 pad。"""
    if not is_distributed:
        assert feats is not None and reps is not None
        return feats, reps

    local_n = 0 if feats is None else int(feats.size(0))
    n_t = torch.tensor([local_n], device=device, dtype=torch.long)
    n_list = [torch.zeros_like(n_t) for _ in range(dist.get_world_size())]
    dist.all_gather(n_list, n_t)
    counts = [int(t.item()) for t in n_list]
    max_n = max(counts) if counts else 0
    if max_n < 1:
        raise RuntimeError("TrACE 估 d：所有 rank 都没有轨迹")

    if feats is None:
        feats_pad = torch.zeros(max_n, dim, device=device, dtype=torch.float32)
        reps_pad = torch.zeros(max_n, device=device, dtype=torch.float32)
    else:
        feats_pad = torch.zeros(max_n, dim, device=device, dtype=torch.float32)
        reps_pad = torch.zeros(max_n, device=device, dtype=torch.float32)
        feats_pad[:local_n].copy_(feats.to(device=device, dtype=torch.float32))
        reps_pad[:local_n].copy_(reps.to(device=device, dtype=torch.float32))

    world = dist.get_world_size()
    feat_buf = [torch.empty_like(feats_pad) for _ in range(world)]
    rep_buf = [torch.empty_like(reps_pad) for _ in range(world)]
    dist.all_gather(feat_buf, feats_pad)
    dist.all_gather(rep_buf, reps_pad)
    chunks_f = [feat_buf[i][: counts[i]] for i in range(world) if counts[i] > 0]
    chunks_r = [rep_buf[i][: counts[i]] for i in range(world) if counts[i] > 0]
    return torch.cat(chunks_f, dim=0).cpu(), torch.cat(chunks_r, dim=0).cpu()


def maybe_refresh_attr_d(
    model: Any,
    *,
    ema_state: dict[str, torch.Tensor] | None,
    opt_step: int,
    variant: str,
    generate_sampling: dict[str, Any],
    rank: int,
    world_size: int,
    is_distributed: bool,
    device: torch.device,
    log: Callable[[str], None] | None = None,
) -> bool:
    """若到点则用 EMA 权重重跑 Alg.1 并 EMA 更新 ``attr_d``。返回是否更新。"""
    bb = _backbone(model)
    if float(getattr(bb, "attr_weight", 0.0) or 0.0) <= 0.0:
        return False
    if str(variant) == "fast":
        return False
    freeze = bool(getattr(bb, "attr_freeze_d", False))
    valid = float(bb.attr_d_valid.item()) > 0.5
    if freeze and valid:
        return False

    warmup = int(getattr(bb, "attr_warmup_steps", 1000))
    every = max(1, int(getattr(bb, "attr_d_every", 1000)))
    last = int(bb.attr_d_last_opt.item())
    if opt_step < warmup:
        return False
    # 冷却只看上次尝试（成功或因 gap 跳过都拨 last），与 d 是否已有效无关。
    # 否则 gap 不够时 last 不前进，每个 opt-step 都会重跑 Alg.1。
    if last > 0 and (opt_step - last) < every:
        return False

    n = int(getattr(bb, "attr_estimate_n", 128))
    if n < 6:
        bb.attr_d_last_opt.fill_(int(opt_step))
        if rank == 0 and log is not None:
            log(f"TrACE skip 估 d：attr_estimate_n={n} < 6")
        return False

    sampling_cfg = dict(generate_sampling)
    sampling_cfg["ace"] = False
    sampling_cfg.pop("ace_direction", None)

    tok_name = str(getattr(getattr(unwrap_model(model), "config", None), "tokenizer", None) or "t5-small")
    bs = int(getattr(bb, "attr_estimate_bs", 8))
    seed = int(getattr(bb, "attr_estimate_seed", 0))
    beta = float(getattr(bb, "attr_d_ema_beta", 0.9))
    min_gap = float(getattr(bb, "attr_min_rep_gap", 0.01))
    dim = int(bb.text_encoder_dim)

    start, n_local = _shard(n, world_size if is_distributed else 1, rank)
    was_training = model.training
    model.eval()
    if rank == 0 and log is not None:
        log(
            f"TrACE 估 d：opt_step={opt_step} n={n} "
            f"shard={n_local}@{start} freeze={freeze} ema={ema_state is not None}"
        )

    feats: torch.Tensor | None = None
    reps: torch.Tensor | None = None
    with torch.no_grad():
        with swap_ema_weights(model, ema_state):
            if n_local > 0:
                feats, reps = collect_ace_sc_rep_pairs(
                    bb,
                    sampling_cfg=sampling_cfg,
                    tokenizer_name=tok_name,
                    n=n_local,
                    batch_size=bs,
                    seed=seed + start * 100003,
                    log_progress=(rank == 0),
                )

    feats, reps = _all_gather_pairs(
        feats, reps, dim=dim, device=device, is_distributed=is_distributed,
    )
    d_hat, meta = direction_from_sc_rep_pairs(feats, reps)
    gap = float(meta["rep_gap"])
    if gap < min_gap:
        bb.attr_d_last_opt.fill_(int(opt_step))
        nxt = int(opt_step) + every
        if rank == 0 and log is not None:
            log(
                f"TrACE 跳过更新 d：rep_gap={gap:.4f} < {min_gap:g} "
                f"(rep_lo={meta['rep_lo']:.4f} rep_hi={meta['rep_hi']:.4f})；"
                f"下次不早于 opt_step={nxt}"
            )
        if was_training:
            model.train()
        return False

    d_hat = d_hat.to(device=bb.attr_d.device, dtype=bb.attr_d.dtype)
    old = bb.attr_d.detach()
    if valid:
        mixed = beta * old + (1.0 - beta) * d_hat
        mixed = mixed / (mixed.norm() + 1e-8)
        rho = float((d_hat * old).sum().abs().item())
    else:
        mixed = d_hat
        rho = float("nan")
    bb.attr_d.copy_(mixed)
    bb.attr_d_valid.fill_(1.0)
    bb.attr_d_last_opt.fill_(int(opt_step))
    bb.last_attr_rho = rho
    if was_training:
        model.train()
    if rank == 0 and log is not None:
        rho_s = "n/a" if rho != rho else f"{rho:.3f}"
        log(
            f"TrACE 更新 d：rep_lo={meta['rep_lo']:.4f} rep_hi={meta['rep_hi']:.4f} "
            f"gap={gap:.4f} cos={rho_s} beta={beta:g}"
        )
    return True
