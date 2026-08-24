"""latent 课程训练在线评测：seg512 held-out + owt-bucket 分桶指标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from preprocess.preprocess import _PreprocessedSplitDataset, get_preprocessed
from train.batching import collate_input_ids
from train.checkpoint import unwrap_model
from train.eval import forward_loss
from train.latent_curriculum import (
    LatentCurriculumSampler,
    LatentCurriculumSpec,
    _bucket_indices,
    batch_graph_l,
)
from train.metrics import _TRAIN_LOG, _train_log


def _as_optional_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _round_metric(val: float | None, *, acc: bool = False) -> str | float:
    if val is None:
        return ""
    if acc:
        return round(val, 4)
    return round(val, 6)


@dataclass(frozen=True)
class LatentCurriculumEvalContext:
    """课程评测上下文：seg512 held-out + bucket held-out 分桶索引。"""

    seg512_loader: DataLoader
    seg512_probe_pool: Dataset
    eval_split: str
    bucket_split: _PreprocessedSplitDataset
    bucket_indices: dict[int, np.ndarray]
    pad_token_id: int
    per_bucket_samples: int

    @classmethod
    def build(
        cls,
        spec: LatentCurriculumSpec,
        *,
        dataset: str,
        pad_token_id: int,
        batch_size: int,
        eval_sample_count: int | None,
        eval_sample_seed: int,
        eval_split: str,
        rank: int,
        world_size: int,
    ) -> LatentCurriculumEvalContext:
        from train.batching import TokenChunkDataset, build_eval_subset, shard_eval_dataset

        seg512 = get_preprocessed(spec.seg512_preprocess, dataset)
        bucket = get_preprocessed(spec.bucket_preprocess, dataset)
        seg512_eval = TokenChunkDataset(seg512.load_split(eval_split))
        eval_ds, _ = build_eval_subset(
            seg512_eval, eval_sample_count, eval_sample_seed,
        )
        eval_local = shard_eval_dataset(
            eval_ds, rank=rank, world_size=world_size,
        )
        seg512_loader = DataLoader(
            eval_local,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_input_ids,
        )
        bucket_split = bucket.load_split(eval_split)
        bucket_indices = _bucket_indices(bucket_split)
        total = int(eval_sample_count or 1024)
        per_bucket = max(32, total // max(1, len(bucket_indices)))
        return cls(
            seg512_loader=seg512_loader,
            seg512_probe_pool=eval_ds,
            eval_split=eval_split,
            bucket_split=bucket_split,
            bucket_indices=bucket_indices,
            pad_token_id=pad_token_id,
            per_bucket_samples=per_bucket,
        )


def curriculum_in_observation_window(sampler: LatentCurriculumSampler) -> bool:
    stage = sampler.current_stage
    if stage.name not in ("s3", "s4"):
        return False
    stage_start = sum(
        s.effective_budget for s in sampler.spec.stages[: sampler._stage_idx]
    )
    elapsed = sampler.effective_tokens_global - stage_start
    return elapsed < sampler.spec.observation_window_tokens


def _pad_row(row: torch.Tensor, graph_l: int, pad_id: int, eff_len: int | None = None) -> torch.Tensor:
    if eff_len is not None:
        row = row[:eff_len]
    if row.numel() > graph_l:
        return row[:graph_l]
    if row.numel() < graph_l:
        pad = torch.full((graph_l - row.numel(),), pad_id, dtype=row.dtype)
        return torch.cat([row, pad])
    return row


def _collate_bucket(items: list[dict[str, torch.Tensor]], graph_l: int, pad_id: int) -> torch.Tensor:
    rows = [
        _pad_row(
            item["input_ids"],
            graph_l,
            pad_id,
            eff_len=int(item["length"]) if "length" in item else None,
        )
        for item in items
    ]
    return torch.stack(rows, dim=0)


def _pad_batch_to(batch: torch.Tensor, pad_to: int, pad_id: int) -> torch.Tensor:
    """只 pad、不截断 held-out token（长度已超过目标时原样返回）。"""
    t = int(batch.size(1))
    if t >= pad_to:
        return batch
    pad = torch.full(
        (batch.size(0), pad_to - t),
        pad_id,
        dtype=batch.dtype,
        device=batch.device,
    )
    return torch.cat([batch, pad], dim=1)


@torch.no_grad()
def _latent_batch_metrics(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    pad_to_len: int | None = None,
    pad_token_id: int | None = None,
) -> dict[str, float]:
    """在线 latent 指标：打 DDP/compile 包装（与训练同图），保持 train()。

    EMA 由调用方 ``swap_ema_weights`` ``copy_`` 进同一 Parameter。输入 pad
    到当前阶段 ``batch_graph_l``，命中训练已编译的 shape。不切 eval()。
    """
    use_amp = device.type == "cuda"
    batch = batch.to(device, non_blocking=True)
    if pad_to_len is not None and pad_token_id is not None:
        batch = _pad_batch_to(batch, pad_to_len, pad_token_id)
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        forward_loss(model, batch)
    return dict(unwrap_model(model).train_metrics())


def _aggregate_metrics(
    totals: dict[str, float],
    counts: dict[str, int],
    batch_metrics: dict[str, float],
) -> None:
    for key in ("recon_ce", "kl", "mask", "token_acc", "mask_acc"):
        val = _as_optional_float(batch_metrics.get(key))
        if val is None:
            continue
        totals[key] = totals.get(key, 0.0) + val
        counts[key] = counts.get(key, 0) + 1


def _finalize_metrics(totals: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, total in totals.items():
        n = counts.get(key, 0)
        if n > 0:
            out[key] = total / n
    return out


@torch.no_grad()
def eval_latent_loader_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    is_distributed: bool,
    pbar_parent: tqdm | None = None,
    log: bool = True,
    desc: str = "eval",
    pad_to_len: int | None = None,
    pad_token_id: int | None = None,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    batch_iter: Iterable = loader
    show_pbar = log and pbar_parent is not None and len(loader) > 0
    if show_pbar:
        pbar_parent.clear()
        batch_iter = tqdm(
            loader,
            desc=desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            total=len(loader),
        )
    try:
        for batch in batch_iter:
            if not isinstance(batch, torch.Tensor):
                raise TypeError("loader must yield input_ids tensors")
            _aggregate_metrics(
                totals, counts,
                _latent_batch_metrics(
                    model, batch, device, amp_dtype,
                    pad_to_len=pad_to_len, pad_token_id=pad_token_id,
                ),
            )
    finally:
        if isinstance(batch_iter, tqdm):
            batch_iter.close()
        if show_pbar and pbar_parent is not None:
            pbar_parent.refresh()

    if is_distributed:
        keys = sorted(set(totals) | set(counts))
        if keys:
            vec = torch.tensor(
                [totals.get(k, 0.0) for k in keys] + [float(counts.get(k, 0)) for k in keys],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            half = len(keys)
            totals = {k: float(vec[i].item()) for i, k in enumerate(keys)}
            counts = {k: int(vec[half + i].item()) for i, k in enumerate(keys)}

    return _finalize_metrics(totals, counts)


def _subsample_indices(pool: np.ndarray, count: int, seed: int) -> np.ndarray:
    if pool.size <= count:
        return pool
    rng = np.random.default_rng(seed)
    return rng.choice(pool, size=count, replace=False)


@torch.no_grad()
def eval_bucket_metrics(
    model: nn.Module,
    ctx: LatentCurriculumEvalContext,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    eval_sample_seed: int,
    rank: int,
    world_size: int,
    is_distributed: bool,
    pbar_parent: tqdm | None = None,
    log: bool = True,
    stage_graph_l: int | None = None,
    max_bucket_len: int | None = None,
) -> dict[str, dict[str, float]]:
    """各 pad 桶 held-out 指标；rank 分片后 allreduce。

    只评 ``bucket <= max_bucket_len``（缺省不截断），避免 S2 去跑 1024/2048
    图长。桶序列 pad 到 ``batch_graph_l(阶段, bucket)``，与训练同 shape。
    """
    out: dict[str, dict[str, float]] = {}
    for bucket in sorted(ctx.bucket_indices):
        if max_bucket_len is not None and int(bucket) > int(max_bucket_len):
            continue
        pool = ctx.bucket_indices[bucket]
        if pool.size == 0:
            continue
        chosen = _subsample_indices(
            pool, ctx.per_bucket_samples, eval_sample_seed + bucket * 17,
        )
        if is_distributed:
            indices = list(range(rank, len(chosen), world_size))
            local_ids = chosen[indices]
        else:
            local_ids = chosen
        if local_ids.size == 0:
            metrics: dict[str, float] = {}
        else:
            batches: list[torch.Tensor] = []
            bs = 16
            items = [ctx.bucket_split[int(i)] for i in local_ids]
            pad_l = (
                batch_graph_l(int(stage_graph_l), int(bucket))
                if stage_graph_l is not None
                else int(bucket)
            )
            for start in range(0, len(items), bs):
                chunk = items[start : start + bs]
                batches.append(
                    _collate_bucket(chunk, pad_l, ctx.pad_token_id),
                )
            totals: dict[str, float] = {}
            counts: dict[str, int] = {}
            for batch in batches:
                _aggregate_metrics(
                    totals,
                    counts,
                    _latent_batch_metrics(
                        model, batch, device, amp_dtype,
                    ),
                )
            metrics = _finalize_metrics(totals, counts)
            if is_distributed:
                keys = sorted(set(totals) | set(counts))
                if keys:
                    vec = torch.tensor(
                        [totals.get(k, 0.0) for k in keys]
                        + [float(counts.get(k, 0)) for k in keys],
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
                    half = len(keys)
                    totals = {k: float(vec[i].item()) for i, k in enumerate(keys)}
                    counts = {k: int(vec[half + i].item()) for i, k in enumerate(keys)}
                    metrics = _finalize_metrics(totals, counts)
        out[str(bucket)] = metrics
        if log and rank == 0 and metrics:
            recon = metrics.get("recon_ce", float("nan"))
            kl = metrics.get("kl", float("nan"))
            mask_acc = metrics.get("mask_acc", float("nan"))
            summary = (
                f"eval bucket-{bucket}: recon_ce {recon:.4f}"
                if recon == recon
                else f"eval bucket-{bucket}:"
            )
            if kl == kl:
                summary += f" kl {kl:.4f}"
            if mask_acc == mask_acc:
                summary += f" mask_acc {mask_acc:.3f}"
            if pbar_parent is not None:
                tqdm.write(f"{_TRAIN_LOG} {summary}")
            else:
                _train_log(summary)
    return out


def latent_curriculum_eval_fields() -> list[str]:
    base = [
        "curriculum_stage",
        "observation_window",
        "seg512_recon_ce",
        "seg512_kl",
        "seg512_mask_acc",
    ]
    for bucket in (256, 512, 1024, 2048):
        prefix = f"b{bucket}"
        base.extend([
            f"{prefix}_recon_ce",
            f"{prefix}_kl",
            f"{prefix}_mask_acc",
        ])
    return base


def build_latent_curriculum_eval_row(
    *,
    step: int,
    tokens: int,
    lr: float,
    seg512_metrics: dict[str, float],
    bucket_metrics: dict[str, dict[str, float]] | None,
    curriculum_stage: str,
    observation_window: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": step,
        "tokens": tokens,
        "lr": lr,
        "eval_loss": _round_metric(_as_optional_float(seg512_metrics.get("recon_ce"))),
        "curriculum_stage": curriculum_stage,
        "observation_window": int(observation_window),
        "seg512_recon_ce": _round_metric(_as_optional_float(seg512_metrics.get("recon_ce"))),
        "seg512_kl": _round_metric(_as_optional_float(seg512_metrics.get("kl"))),
        "seg512_mask_acc": _round_metric(
            _as_optional_float(seg512_metrics.get("mask_acc")), acc=True,
        ),
    }
    for bucket in (256, 512, 1024, 2048):
        metrics = (bucket_metrics or {}).get(str(bucket), {})
        prefix = f"b{bucket}"
        row[f"{prefix}_recon_ce"] = _round_metric(
            _as_optional_float(metrics.get("recon_ce")),
        )
        row[f"{prefix}_kl"] = _round_metric(_as_optional_float(metrics.get("kl")))
        row[f"{prefix}_mask_acc"] = _round_metric(
            _as_optional_float(metrics.get("mask_acc")), acc=True,
        )
    return row


def run_latent_curriculum_eval(
    model: nn.Module,
    *,
    ctx: LatentCurriculumEvalContext,
    sampler: LatentCurriculumSampler,
    step: int,
    tokens: int,
    lr: float,
    device: torch.device,
    amp_dtype: torch.dtype,
    eval_sample_seed: int,
    rank: int,
    world_size: int,
    is_distributed: bool,
    pbar_parent: tqdm | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """seg512 held-out 全量指标；S2 起追加 bucket 分桶指标。

    保持 ``train()``，前向打编译模块。EMA 由调用方 ``swap_ema_weights``
    写入同一 Parameter。分桶只评 ``bucket <= 当前阶段 graph_l``，并 pad
    到 ``batch_graph_l``。
    """
    stage_l = sampler.current_stage.graph_l
    seg_pad = batch_graph_l(stage_l, min(512, stage_l))
    seg512 = eval_latent_loader_metrics(
        model,
        ctx.seg512_loader,
        device,
        amp_dtype,
        is_distributed=is_distributed,
        pbar_parent=pbar_parent,
        log=log,
        desc="eval seg512",
        pad_to_len=seg_pad,
        pad_token_id=ctx.pad_token_id,
    )
    bucket_out: dict[str, dict[str, float]] | None = None
    if sampler._stage_idx >= 1:
        bucket_out = eval_bucket_metrics(
            model,
            ctx,
            device=device,
            amp_dtype=amp_dtype,
            eval_sample_seed=eval_sample_seed,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
            pbar_parent=pbar_parent,
            log=log,
            stage_graph_l=stage_l,
            max_bucket_len=stage_l,
        )
    return build_latent_curriculum_eval_row(
        step=step,
        tokens=tokens,
        lr=lr,
        seg512_metrics=seg512,
        bucket_metrics=bucket_out,
        curriculum_stage=sampler.current_stage.name,
        observation_window=curriculum_in_observation_window(sampler),
    )
