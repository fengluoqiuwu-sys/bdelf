#!/usr/bin/env python3
"""LLM pretraining entry point for bdelf."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import hf_config  # noqa: F401
from models import build_model, get_hf_model, list_model_configs, list_models, resolve_model_config_path
from models.tokens import FL_TokenLayout, token_layout_from_cfg
from dataset import list_datasets
from preprocess import get_preprocessed, list_preprocess
from train import FL_TrainConfig, get_train_config, list_train_configs, list_train_models

TRAIN_CSV_FIELDS = [
    "step",
    "train_loss",
    "train_ppl",
    "loss_branch",
    "denoise_mse",
    "decode_ce",
    "lr",
    "tokens_per_sec",
]
EVAL_CSV_FIELDS = ["step", "gpt2_loss", "gpt2_ppl", "lr"]

_TRAIN_LOG = "[train]"


def _train_log(msg: str, *, file: Any = None) -> None:
    if file is None:
        file = sys.stdout
    print(f"{_TRAIN_LOG} {msg}", file=file, flush=True)


class TokenChunkDataset(Dataset):
    """Wraps a preprocessed split; yields ``input_ids`` tensors."""

    def __init__(self, split_ds: Dataset) -> None:
        self.split_ds = split_ds

    def __len__(self) -> int:
        return len(self.split_ds)

    def __getitem__(self, index: int) -> torch.Tensor:
        item = self.split_ds[index]
        if isinstance(item, dict):
            return item["input_ids"]
        return item


def collate_input_ids(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch)


def build_eval_subset(
    dataset: Dataset,
    sample_count: int | None,
    seed: int,
) -> tuple[Dataset, int]:
    """Return eval dataset, optionally subsampled without mutating the source split."""
    total = len(dataset)
    if sample_count is None or sample_count >= total:
        return dataset, total
    gen = torch.Generator()
    gen.manual_seed(seed)
    indices = torch.randperm(total, generator=gen)[:sample_count].tolist()
    return Subset(dataset, indices), sample_count


_PERM_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
_PERM_CACHE_MAX = 3


def _get_epoch_perm(dataset_len: int, seed: int, epoch: int) -> np.ndarray:
    key = (dataset_len, seed, epoch)
    perm = _PERM_CACHE.get(key)
    if perm is None:
        gen = torch.Generator()
        gen.manual_seed(seed + epoch)
        perm = torch.randperm(dataset_len, generator=gen).numpy()
        if len(_PERM_CACHE) >= _PERM_CACHE_MAX:
            _PERM_CACHE.pop(next(iter(_PERM_CACHE)))
        _PERM_CACHE[key] = perm
    return perm


def get_train_batch_indices(
    step: int,
    dataset_len: int,
    batch_size: int,
    world_size: int,
    seed: int,
) -> np.ndarray:
    global_batch = batch_size * world_size
    if dataset_len < global_batch:
        raise ValueError(
            f"Dataset size {dataset_len} is less than global batch {global_batch} "
            f"(batch_size={batch_size} x world_size={world_size})"
        )

    global_pos = step * global_batch
    epoch = global_pos // dataset_len
    offset = global_pos % dataset_len

    parts: list[np.ndarray] = []
    filled = 0
    while filled < global_batch:
        perm = _get_epoch_perm(dataset_len, seed, epoch)
        take = min(global_batch - filled, dataset_len - offset)
        parts.append(perm[offset : offset + take])
        filled += take
        offset += take
        if offset >= dataset_len:
            epoch += 1
            offset = 0
    indices = parts[0] if len(parts) == 1 else np.concatenate(parts)
    return np.ascontiguousarray(indices, dtype=np.int64)


def fetch_train_batch(
    dataset: TokenChunkDataset,
    step: int,
    batch_size: int,
    world_size: int,
    rank: int,
    seed: int,
) -> torch.Tensor:
    indices = get_train_batch_indices(step, len(dataset), batch_size, world_size, seed)
    start = rank * batch_size
    rank_indices = indices[start : start + batch_size]
    rows = [dataset[int(i)] for i in rank_indices]
    return collate_input_ids(rows)


def setup_distributed(cfg: FL_TrainConfig) -> tuple[int, int, torch.device, bool]:
    if cfg.world_size <= 1:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device found. Single-GPU training requires a GPU.")
        return 0, 1, torch.device("cuda"), False

    if "RANK" not in os.environ:
        raise RuntimeError("Distributed worker missing RANK environment variable")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != cfg.world_size:
        raise RuntimeError(
            f"Configured world_size={cfg.world_size}, but {world_size} processes were launched."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device(f"cuda:{local_rank}"), True


def _spawn_worker(
    local_rank: int,
    model_name: str,
    train_config: str,
    run_name: str | None,
    world_size: int,
    dataset: str,
    preprocess: str,
) -> None:
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    cfg = get_train_config(
        model_name, train_config, dataset=dataset, preprocess=preprocess,
    )
    if run_name:
        cfg.name = run_name
    size = train_config.rsplit("-", 1)[0]
    run_training(model_name, size, cfg)


def get_amp_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def get_lr(step: int, cfg: FL_TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.learning_rate * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.learning_rate * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def uses_full_sequence(model: nn.Module) -> bool:
    return getattr(unwrap_model(model), "full_sequence_training", False)


def uses_dual_branch_logging(model: nn.Module) -> bool:
    return getattr(unwrap_model(model), "dual_branch_logging", False)


def forward_loss(
    model: nn.Module,
    batch: torch.Tensor,
    *,
    branch: str | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {}
    if branch is not None:
        if not uses_dual_branch_logging(model):
            raise ValueError(f"Model does not support branch={branch!r}")
        kwargs["branch"] = branch
    if uses_full_sequence(model):
        _, loss = model(batch, None, **kwargs)
    else:
        _, loss = model(batch[:, :-1], batch[:, 1:], **kwargs)
    return loss


def loss_to_ppl(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def eval_sampling_cfg(cfg: FL_TrainConfig) -> dict[str, bool]:
    """BDELF eval sampling config: defaults to legacy (full AdaLN, consistent with training)."""
    return {"use_fast_infer": cfg.eval_use_fast_infer}


def prepare_gpt2_eval_batch(
    batch: torch.Tensor,
    layout: FL_TokenLayout,
    *,
    gpt2_vocab_size: int,
    fill_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map extended-vocab token ids to GPT-2 baseline range for eval PPL."""
    input_ids = batch.clone()
    labels = batch.clone()
    oov = input_ids >= gpt2_vocab_size
    input_ids[oov] = fill_token_id
    for token_id in (layout.bos_token_id, layout.eos_token_id, layout.pad_token_id):
        labels[labels == token_id] = -100
    return input_ids, labels


@torch.no_grad()
def eval_gpt2_ppl(
    gpt2_model: nn.Module,
    loader: DataLoader,
    amp_dtype: torch.dtype,
    eval_token_layout: FL_TokenLayout,
    *,
    pbar_parent: tqdm | None = None,
) -> tuple[float, float]:
    """Eval PPL from gpt2-large (standard causal LM CE on eval split)."""
    gpt2_model.eval()
    gpt2_device = next(gpt2_model.parameters()).device
    gpt2_vocab_size = int(getattr(gpt2_model.config, "vocab_size", 50257))
    fill_token_id = int(
        getattr(gpt2_model.config, "eos_token_id", None) or 50256,
    )
    use_amp = gpt2_device.type == "cuda"
    total_loss = 0.0
    batches = 0
    if len(loader) == 0:
        return float("nan"), float("nan")
    batch_iter: DataLoader | tqdm = loader
    if pbar_parent is not None:
        batch_iter = tqdm(
            loader,
            desc="eval/gpt2-large",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            parent=pbar_parent,
            total=len(loader),
        )
    try:
        for batch in batch_iter:
            batch = batch.to(gpt2_device, non_blocking=True)
            input_ids, labels = prepare_gpt2_eval_batch(
                batch,
                eval_token_layout,
                gpt2_vocab_size=gpt2_vocab_size,
                fill_token_id=fill_token_id,
            )
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = gpt2_model(input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                total_loss += float(loss.item())
            batches += 1
    finally:
        if pbar_parent is not None and isinstance(batch_iter, tqdm):
            batch_iter.close()
    avg_loss = total_loss / max(1, batches)
    return avg_loss, loss_to_ppl(avg_loss)


def append_csv_row(csv_path: Path, fields: list[str], row: dict[str, Any]) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def truncate_csv_for_resume(csv_path: Path, start_step: int) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return 0
        rows_by_step: dict[int, dict[str, str]] = {}
        for row in reader:
            step = int(row["step"])
            if step < start_step:
                rows_by_step[step] = row
    rows = [rows_by_step[s] for s in sorted(rows_by_step)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    return float(raw)


def update_ppl_plots(
    train_csv: Path,
    eval_csv: Path,
    out_dir: Path,
) -> None:
    train_rows = _read_csv_rows(train_csv)
    eval_rows = _read_csv_rows(eval_csv)
    if not train_rows:
        return

    train_steps = [int(r["step"]) for r in train_rows]
    train_ppl = [_parse_float(r.get("train_ppl")) for r in train_rows]
    train_lr = [float(r["lr"]) for r in train_rows]

    eval_steps = [int(r["step"]) for r in eval_rows]
    eval_ppl = [_parse_float(r.get("gpt2_ppl")) for r in eval_rows]

    for cap, filename in ((1000.0, "ppl_under_1000.png"), (100.0, "ppl_under_100.png")):
        t_steps, t_ppls = zip(
            *[(s, p) for s, p in zip(train_steps, train_ppl) if p is not None and p <= cap]
        ) if any(p is not None and p <= cap for p in train_ppl) else ([], [])

        e_steps, e_ppls = zip(
            *[(s, p) for s, p in zip(eval_steps, eval_ppl) if p is not None and p <= cap]
        ) if any(p is not None and p <= cap for p in eval_ppl) else ([], [])

        if not t_steps and not e_steps:
            continue

        fig, ax_ppl = plt.subplots(figsize=(10, 4.5))

        if t_steps:
            ax_ppl.plot(
                t_steps, t_ppls, color="#4C72B0", alpha=0.55, linewidth=1.2,
                label="train ppl (exp loss)", zorder=1,
            )
        if e_steps:
            ax_ppl.plot(
                e_steps, e_ppls, color="#D62728", linewidth=2.8, marker="o",
                markersize=4, label="eval ppl (gpt2-large)", zorder=5,
            )

        ax_lr = ax_ppl.twinx()
        lr_steps, lr_vals = zip(
            *[(s, lr) for s, lr in zip(train_steps, train_lr) if lr > 0]
        ) if train_lr else ([], [])
        if lr_steps:
            ax_lr.plot(
                lr_steps, lr_vals, color="#7F7F7F", linestyle="--",
                linewidth=1.0, alpha=0.9, label="lr", zorder=2,
            )
            ax_lr.set_ylabel("learning rate")
            ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        ax_ppl.set_xlabel("step")
        ax_ppl.set_ylabel("perplexity")
        ax_ppl.set_title(f"PPL & LR (ppl ≤ {cap:g})")
        ax_ppl.grid(True, alpha=0.25)

        handles, labels = ax_ppl.get_legend_handles_labels()
        h2, l2 = ax_lr.get_legend_handles_labels()
        ax_ppl.legend(handles + h2, labels + l2, loc="upper right")

        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)


def build_train_row(
    step: int,
    train_loss: float,
    lr: float,
    tokens_per_sec: float,
    *,
    dual_branch: bool,
    loss_branch: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": step,
        "train_loss": round(train_loss, 6),
        "train_ppl": "",
        "loss_branch": "",
        "denoise_mse": "",
        "decode_ce": "",
        "lr": lr,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
    if dual_branch:
        row["loss_branch"] = loss_branch
        if loss_branch == "denoise":
            row["denoise_mse"] = round(train_loss, 6)
        else:
            ppl = loss_to_ppl(train_loss)
            row["decode_ce"] = round(train_loss, 6)
            row["train_ppl"] = round(ppl, 4)
    else:
        row["train_ppl"] = round(loss_to_ppl(train_loss), 4)
    return row


def _train_metrics_text(row: dict[str, Any]) -> str:
    branch = row.get("loss_branch") or ""
    if branch == "denoise":
        return (
            f"[denoise] mse {row['train_loss']:.4f} | "
            f"lr {row['lr']:.2e} | {row['tokens_per_sec']:.0f} tok/s"
        )
    if branch == "decode":
        return (
            f"[decode] ce {row['train_loss']:.4f} ppl {row['train_ppl']} | "
            f"lr {row['lr']:.2e} | {row['tokens_per_sec']:.0f} tok/s"
        )
    return (
        f"loss {row['train_loss']:.4f} ppl {row['train_ppl']} | "
        f"lr {row['lr']:.2e} | {row['tokens_per_sec']:.0f} tok/s"
    )


def format_interval_summary(
    step: int,
    max_steps: int,
    row: dict[str, Any],
    *,
    gpt2_loss: float | None = None,
    gpt2_ppl: float | None = None,
) -> list[str]:
    pct = 100.0 * (step + 1) / max_steps
    lines = [f"[{step + 1}/{max_steps} ({pct:.1f}%)] {_train_metrics_text(row)}"]
    if gpt2_loss is not None and gpt2_ppl is not None:
        lines.append(f"  eval/gpt2-large: loss {gpt2_loss:.4f} ppl {gpt2_ppl:.2f}")
    return lines


def _rank0_log(msg: str, pbar: tqdm | None) -> None:
    line = f"{_TRAIN_LOG} {msg}"
    if pbar is not None:
        tqdm.write(line)
    else:
        print(line, flush=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: FL_TrainConfig,
    model_meta: dict[str, Any],
) -> None:
    raw = unwrap_model(model)
    grads = [
        p.grad.detach().cpu() if p.grad is not None else None
        for p in raw.parameters()
    ]
    payload: dict[str, Any] = {
        "step": step,
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "grads": grads,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        },
        "train_config": asdict(cfg),
        "model_meta": model_meta,
    }
    if torch.cuda.is_available():
        payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    cfg: FL_TrainConfig,
    model_meta: dict[str, Any],
) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    saved_cfg = ck.get("train_config") or {}
    if saved_cfg.get("name") and saved_cfg["name"] != cfg.name:
        raise ValueError(
            f"checkpoint run={saved_cfg['name']!r} does not match current {cfg.name!r}"
        )
    saved_meta = ck.get("model_meta") or {}
    if saved_meta.get("name") and saved_meta["name"] != model_meta.get("name"):
        raise ValueError(
            f"checkpoint model {saved_meta.get('name')!r} "
            f"does not match current {model_meta.get('name')!r}"
        )

    raw = unwrap_model(model)
    raw.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    _move_optimizer_state_to_device(optimizer, device)

    grads = ck.get("grads")
    if grads is not None:
        for p, g in zip(raw.parameters(), grads):
            p.grad = g.to(device) if g is not None else None

    rng = ck.get("rng")
    if rng is not None:
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return int(ck["step"])


def set_seed(seed: int, rank: int) -> None:
    s = seed + rank
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_eval_baseline(cfg: FL_TrainConfig) -> nn.Module:
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[cfg.eval_model_dtype]
    device = cfg.eval_model_device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("eval_model_device=cuda but no CUDA device was found")
    model = get_hf_model(cfg.eval_model, torch_dtype=torch_dtype, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def train_loop(
    model: nn.Module,
    cfg: FL_TrainConfig,
    model_meta: dict[str, Any],
    train_ds: TokenChunkDataset,
    eval_loader: DataLoader | None,
    gpt2_model: nn.Module | None,
    eval_token_layout: FL_TokenLayout | None,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    is_distributed: bool,
) -> None:
    amp_dtype = get_amp_dtype(cfg.dtype)
    run_dir = Path(cfg.checkpoint_root) / cfg.name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {"train": asdict(cfg), "model": model_meta},
                f,
                indent=2,
                ensure_ascii=False,
            )

    if is_distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    raw = unwrap_model(model)
    decay_params = [p for p in raw.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in raw.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )

    train_csv = run_dir / "train_log.csv"
    eval_csv = run_dir / "eval_log.csv"
    latest_ckpt = run_dir / "checkpoint_latest.pt"

    step = 0
    optimizer.zero_grad(set_to_none=True)

    if rank == 0 and cfg.resume and latest_ckpt.is_file():
        step = load_checkpoint(
            latest_ckpt, model, optimizer, device, cfg=cfg, model_meta=model_meta,
        )
        kept_train = truncate_csv_for_resume(train_csv, step)
        kept_eval = truncate_csv_for_resume(eval_csv, step)
        update_ppl_plots(train_csv, eval_csv, run_dir)
        _train_log(
            f"Resuming from checkpoint: step {step} "
            f"(train_log {kept_train} rows, eval_log {kept_eval} rows)",
        )
        if step >= cfg.max_steps:
            _train_log(f"Reached max_steps={cfg.max_steps}; training is already complete")
            return

    if is_distributed:
        dist.barrier()

    dual_branch = uses_dual_branch_logging(model)
    if rank == 0 and dual_branch:
        _train_log(
            "BDELF dual-branch: train ppl is decode branch only; eval ppl from gpt2-large",
        )

    model.train()
    t0 = time.time()
    step_backward_done = False
    pbar: tqdm | None = None
    if rank == 0:
        pbar = tqdm(
            total=cfg.max_steps,
            initial=step,
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )

    try:
        while step < cfg.max_steps:
            batch = fetch_train_batch(
                train_ds, step, cfg.batch_size, world_size, rank, cfg.seed,
            )
            batch = batch.to(device, non_blocking=True)

            lr = get_lr(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = forward_loss(model, batch)
                loss = loss / cfg.grad_accum_steps

            loss.backward()
            step_backward_done = True

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss = loss.item() * cfg.grad_accum_steps
            loss_branch = getattr(raw, "last_loss_branch", "") if dual_branch else ""
            elapsed = time.time() - t0
            seq_tokens = batch.size(0) * (
                batch.size(1) if uses_full_sequence(model) else batch.size(1) - 1
            )
            tokens_per_sec = seq_tokens / max(elapsed, 1e-6)

            row = build_train_row(
                step, train_loss, lr, tokens_per_sec,
                dual_branch=dual_branch, loss_branch=loss_branch,
            )

            if rank == 0:
                pbar.set_postfix(
                    loss=f"{train_loss:.3f}",
                    lr=f"{lr:.2e}",
                    tok_s=f"{tokens_per_sec:.0f}",
                )
                pbar.update(1)
                append_csv_row(train_csv, TRAIN_CSV_FIELDS, row)

                interval_done = (
                    (step + 1) % cfg.eval_every == 0 or (step + 1) >= cfg.max_steps
                )
                if interval_done:
                    gpt2_loss: float | None = None
                    gpt2_ppl: float | None = None
                    if (
                        (step + 1) % cfg.eval_every == 0
                        and eval_loader is not None
                        and gpt2_model is not None
                        and eval_token_layout is not None
                    ):
                        gpt2_loss, gpt2_ppl = eval_gpt2_ppl(
                            gpt2_model,
                            eval_loader,
                            amp_dtype,
                            eval_token_layout,
                            pbar_parent=pbar,
                        )
                        pbar.refresh()
                        eval_row = {
                            "step": step,
                            "gpt2_loss": round(gpt2_loss, 6),
                            "gpt2_ppl": round(gpt2_ppl, 4),
                            "lr": lr,
                        }
                        append_csv_row(eval_csv, EVAL_CSV_FIELDS, eval_row)

                    for line in format_interval_summary(
                        step, cfg.max_steps, row,
                        gpt2_loss=gpt2_loss, gpt2_ppl=gpt2_ppl,
                    ):
                        _rank0_log(line, pbar)

                if (step + 1) % cfg.log_plot_every == 0:
                    update_ppl_plots(train_csv, eval_csv, run_dir)

                if (step + 1) % cfg.save_every == 0:
                    save_checkpoint(latest_ckpt, model, optimizer, step + 1, cfg, model_meta)
                    if (step + 1) % cfg.snapshot_every == 0:
                        save_checkpoint(
                            run_dir / f"checkpoint_step_{step + 1:07d}.pt",
                            model, optimizer, step + 1, cfg, model_meta,
                        )
                    _rank0_log(f"  [ckpt] saved at step {step + 1}", pbar)

            if is_distributed:
                dist.barrier()

            step += 1
            t0 = time.time()

    except KeyboardInterrupt:
        if rank == 0:
            if pbar is not None:
                pbar.close()
                pbar = None
            next_step = step + 1 if step_backward_done else step
            _train_log(f"Interrupted at step {step}; saving checkpoint ...")
            save_checkpoint(latest_ckpt, model, optimizer, next_step, cfg, model_meta)
            update_ppl_plots(train_csv, eval_csv, run_dir)
            _train_log(f"Saved; resume from step {next_step} on next run")
        if is_distributed:
            dist.barrier()
        return

    if rank == 0:
        if pbar is not None:
            pbar.close()
        save_checkpoint(latest_ckpt, model, optimizer, step, cfg, model_meta)
        update_ppl_plots(train_csv, eval_csv, run_dir)
        _train_log(f"Training finished after {step} steps; results in {run_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    models = list_models() or ["<none>"]
    datasets = list_datasets() or ["<none>"]
    preprocess_names = list_preprocess() or ["<none>"]
    parser = argparse.ArgumentParser(
        description="bdelf pretraining entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --model ar --config 100m-fast "
            "--dataset owt --preprocess default\n"
            "  python train.py --model bdelf --config 900m-full "
            "--dataset owt --preprocess default\n\n"
            f"Available models: {', '.join(models)}\n"
            f"Available datasets: {', '.join(datasets)}\n"
            f"Available preprocess configs: {', '.join(preprocess_names)}\n"
            "Train configs: {{100m,300m,900m}}-{{fast,full}}"
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model family name; options: {', '.join(models)}",
    )
    parser.add_argument(
        "--config",
        required=True,
        dest="train_config",
        metavar="CONFIG",
        help="Train config name, e.g. 100m-fast / 900m-full",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name (config/datasets/); options: {', '.join(datasets)}",
    )
    parser.add_argument(
        "--preprocess",
        required=True,
        help=f"Preprocess config name (config/preprocess/); options: {', '.join(preprocess_names)}",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Checkpoint directory name; defaults to train config name field",
    )
    return parser


def validate_args(args: argparse.Namespace) -> tuple[str, str, FL_TrainConfig]:
    models = list_models()
    if args.model not in models:
        raise SystemExit(
            f"Unknown model {args.model!r}. Available: {', '.join(models) or '<none>'}\n"
            f"Model config directory: config/models/<model>/"
        )

    train_models = list_train_models()
    if args.model not in train_models:
        raise SystemExit(
            f"Model {args.model!r} has no train batch config. Available: {', '.join(train_models)}"
        )

    configs = list_train_configs(args.model)
    if args.train_config not in configs:
        raise SystemExit(
            f"Unknown train config {args.train_config!r}. {args.model} available: "
            f"{', '.join(configs)}\n"
            f"Naming format: {{100m,300m,900m}}-{{fast,full}}"
        )

    size = args.train_config.rsplit("-", 1)[0]
    try:
        resolve_model_config_path(args.model, size)
    except FileNotFoundError as exc:
        available = ", ".join(list_model_configs(args.model)) or "<none>"
        raise SystemExit(
            f"Model architecture config not found: config/models/{args.model}/{size}.yaml\n"
            f"Available: {available}\n{exc}"
        ) from exc

    datasets = list_datasets()
    if args.dataset not in datasets:
        raise SystemExit(
            f"Unknown dataset {args.dataset!r}. Available: {', '.join(datasets) or '<none>'}\n"
            f"Config directory: config/datasets/"
        )

    preprocess_names = list_preprocess()
    if args.preprocess not in preprocess_names:
        raise SystemExit(
            f"Unknown preprocess config {args.preprocess!r}. Available: "
            f"{', '.join(preprocess_names) or '<none>'}\n"
            f"Config directory: config/preprocess/"
        )

    try:
        cfg = get_train_config(
            args.model,
            args.train_config,
            dataset=args.dataset,
            preprocess=args.preprocess,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Failed to load train config: {exc}") from exc

    if args.run_name:
        cfg.name = args.run_name
    return args.model, size, cfg


def run_training(model_name: str, model_size: str, cfg: FL_TrainConfig) -> None:
    rank, world_size, device, is_distributed = setup_distributed(cfg)
    set_seed(cfg.seed, rank)

    if rank == 0:
        _train_log(f"Model: {model_name}/{model_size}")
        _train_log(f"Train config: {cfg.name} ({cfg.variant})")
        _train_log(f"Data: dataset={cfg.dataset}, preprocess={cfg.preprocess}")
        _train_log(f"Device: {device}, world_size={world_size}")

    try:
        preprocessed = get_preprocessed(cfg.preprocess, cfg.dataset)
    except FileNotFoundError as exc:
        msg = (
            f"Preprocessed data unavailable: {exc}\n"
            f"Check that dataset={cfg.dataset}, preprocess={cfg.preprocess} "
            f"are configured correctly (first run will download the dataset "
            f"and build the preprocess cache automatically)"
        )
        if rank == 0:
            _train_log(msg, file=sys.stderr)
        raise SystemExit(msg) from exc

    splits = preprocessed.get_splits()
    if "train" not in splits or "eval" not in splits:
        raise SystemExit(f"Dataset is missing train/eval splits; current splits: {splits}")

    train_ds = TokenChunkDataset(preprocessed.load_split("train"))
    eval_ds_full = TokenChunkDataset(preprocessed.load_split("eval"))
    eval_ds, eval_run_size = build_eval_subset(
        eval_ds_full,
        cfg.eval_sample_count,
        cfg.eval_sample_seed,
    )

    eval_loader: DataLoader | None = None
    gpt2_model: nn.Module | None = None
    if rank == 0:
        if len(eval_ds) == 0:
            _train_log("WARNING: eval dataset is empty; gpt2-large eval will be skipped")
        else:
            eval_loader = DataLoader(
                eval_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_input_ids,
            )
            gpt2_model = load_eval_baseline(cfg)
            _train_log(
                f"Loaded eval baseline {cfg.eval_model} "
                f"({cfg.eval_model_dtype}, {cfg.eval_model_device})",
            )

    model_cfg_path = resolve_model_config_path(model_name, model_size)
    import yaml

    with open(model_cfg_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}

    eval_token_layout = token_layout_from_cfg(model_cfg)

    model = build_model(model_name, model_cfg).to(device)
    model_meta = {
        "name": model_name,
        "config_file": str(model_cfg_path),
        "config": model_cfg,
    }

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _train_log(f"Train model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
        _train_log(
            f"train split: {len(train_ds):,} samples, "
            f"eval split: {len(eval_ds_full):,} samples",
        )
        if eval_run_size < len(eval_ds_full):
            _train_log(
                f"eval subsample: {eval_run_size:,} / {len(eval_ds_full):,} "
                f"(seed={cfg.eval_sample_seed})",
            )

    train_loop(
        model,
        cfg,
        model_meta,
        train_ds,
        eval_loader,
        gpt2_model,
        eval_token_layout,
        rank=rank,
        world_size=world_size,
        device=device,
        is_distributed=is_distributed,
    )

    if is_distributed:
        dist.destroy_process_group()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    model_name, model_size, cfg = validate_args(args)

    if cfg.world_size > 1 and "RANK" not in os.environ:
        if not torch.cuda.is_available():
            raise SystemExit("Multi-GPU training requires CUDA.")
        n_gpu = torch.cuda.device_count()
        if n_gpu < cfg.world_size:
            raise SystemExit(
                f"Configured world_size={cfg.world_size}, "
                f"but this machine has only {n_gpu} GPU(s)."
            )
        import torch.multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        _train_log(
            f"Auto-spawning {cfg.world_size} processes (spawn), "
            f"MASTER_PORT={os.environ.get('MASTER_PORT', '29500')}",
        )
        mp.spawn(
            _spawn_worker,
            args=(
                model_name,
                args.train_config,
                args.run_name,
                cfg.world_size,
                args.dataset,
                args.preprocess,
            ),
            nprocs=cfg.world_size,
            join=True,
        )
        return

    run_training(model_name, model_size, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _train_log("Interrupt received; exiting.")
    except Exception as exc:
        _train_log(f"Error: {exc}", file=sys.stderr)
        raise
