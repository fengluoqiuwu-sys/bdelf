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
from models import (
    build_model,
    get_hf_model,
    list_model_configs,
    list_models,
    resolve_model_config_path,
)
from models.tokens import FL_TokenLayout, token_layout_from_cfg
from dataset import list_datasets
from preprocess import get_preprocess, get_preprocessed, list_preprocess
from train import FL_TrainConfig, get_train_config, list_train_configs, list_train_models
from train.muon import build_optimizer, scaled_lr, schedule_optimizer_lrs


# =============================================================================
# Inductor workaround
# =============================================================================


def _patch_inductor_bool_eq() -> None:
    """Work around a torch.compile/Inductor bug on boolean value ranges.

    ``SymPyValueRangeAnalysis.eq`` runs ``a.lower > b.upper`` to test for
    disjoint ranges, but when the operands are boolean (e.g. the ``mode_*``
    embeddings and timestep embedding feed a bool-typed indexing expr), sympy
    forbids ordered comparison on Booleans and raises "A Boolean argument can
    only be used in Eq and Ne". Sibling ops (lt/gt/mul/...) already special-case
    ``is_bool``; only ``eq`` was missed (fixed upstream, not in this torch).
    Patch ``eq`` to mirror that handling; ``ne`` delegates to ``eq`` and is
    fixed for free. See https://github.com/pytorch/pytorch/issues/188231.
    """
    try:
        import sympy
        from torch.utils._sympy.value_ranges import (
            SymPyValueRangeAnalysis,
            ValueRanges,
        )
    except Exception:
        return

    @staticmethod
    def _eq(a, b):  # type: ignore[no-untyped-def]
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton() and a.lower == b.lower:
            return ValueRanges.wrap(sympy.true)
        if a.is_bool or b.is_bool:
            # Booleans are unorderable; two unequal singletons are disjoint,
            # otherwise the result is an unknown bool.
            if a.is_singleton() and b.is_singleton():
                return ValueRanges.wrap(sympy.false)
            return ValueRanges(sympy.false, sympy.true)
        if a.lower > b.upper or b.lower > a.upper:  # ranges disjoint
            return ValueRanges.wrap(sympy.false)
        return ValueRanges(sympy.false, sympy.true)

    SymPyValueRangeAnalysis.eq = _eq


_patch_inductor_bool_eq()


# =============================================================================
# Logging constants
# =============================================================================

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
EVAL_CSV_FIELDS = ["step", "eval_loss", "eval_ppl", "gen_loss", "gen_ppl", "lr"]

_TRAIN_LOG = "[train]"


def _train_log(msg: str, *, file: Any = None) -> None:
    if file is None:
        file = sys.stdout
    print(f"{_TRAIN_LOG} {msg}", file=file, flush=True)


# =============================================================================
# Dataset and batching
# =============================================================================


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


# =============================================================================
# Distributed setup and process launch
# =============================================================================


def _local_compile_root() -> str | None:
    """Pick a writable node-local directory for Triton/Inductor caches.

    Triton and Inductor rely on temp-file + rename; that protocol breaks on
    BeeGFS/NFS (missing ``*.cubin`` / cache files under concurrent compile
    workers). Prefer Slurm node scratch, then TMPDIR, then /tmp.
    """
    job = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    for candidate in (
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("TMPDIR"),
        "/tmp",
    ):
        if not candidate:
            continue
        try:
            if not os.path.isdir(candidate) or not os.access(candidate, os.W_OK):
                continue
            root = os.path.join(candidate, f"bdelf-compile-{job}")
            os.makedirs(root, exist_ok=True)
            return root
        except OSError:
            continue
    return None


def _isolate_compile_cache(local_rank: int) -> None:
    """Point each local rank at its own Triton/Inductor cache on local disk.

    Always prefer node-local scratch over any shared-FS path set by Slurm
    (BeeGFS). Per-rank subdirs additionally avoid same-node contention.
    """
    local_root = _local_compile_root()
    for var, subdir in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        if local_root is not None:
            base = os.path.join(local_root, subdir)
        else:
            base = os.environ.get(var)
            if not base:
                continue
        per_rank = os.path.join(base, f"rank{local_rank}")
        os.makedirs(per_rank, exist_ok=True)
        os.environ[var] = per_rank


def setup_distributed(cfg: FL_TrainConfig) -> tuple[int, int, torch.device, bool]:
    if cfg.world_size <= 1:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device found. Single-GPU training requires a GPU.")
        _isolate_compile_cache(0)
        return 0, 1, torch.device("cuda"), False

    if "RANK" not in os.environ:
        raise RuntimeError("Distributed worker missing RANK environment variable")

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    torch.cuda.set_device(local_rank)
    # Isolate compile caches before any torch.compile happens in train_loop.
    _isolate_compile_cache(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != cfg.world_size:
        raise RuntimeError(
            f"Configured world_size={cfg.world_size}, but {world_size} processes were launched."
        )

    return rank, world_size, torch.device(f"cuda:{local_rank}"), True


ALLOWED_FULL_ULTRA_WORLD_SIZES = frozenset({1, 2, 4, 8})


def _resolve_launch_world_size(train_config: str) -> int | None:
    """Auto-detect GPU count for full/ultra; ``None`` means keep hardware default."""
    variant = train_config.rsplit("-", 1)[-1]
    if variant not in ("full", "ultra"):
        return None
    if not torch.cuda.is_available():
        raise SystemExit(
            f"{variant} training requires CUDA; no GPU detected "
            f"(torch.cuda.is_available() is False)."
        )
    n = torch.cuda.device_count()
    if n not in ALLOWED_FULL_ULTRA_WORLD_SIZES:
        raise SystemExit(
            f"full/ultra 需要 1/2/4/8 张可见 GPU，当前 device_count={n}"
        )
    return n


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
        model_name,
        train_config,
        dataset=dataset,
        preprocess=preprocess,
        world_size=world_size,
    )
    if run_name:
        cfg.name = run_name
    size = train_config.rsplit("-", 1)[0]
    run_training(model_name, size, cfg)


# =============================================================================
# Training helpers
# =============================================================================


def get_amp_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def get_lr(step: int, cfg: FL_TrainConfig) -> float:
    return scaled_lr(step, cfg, cfg.learning_rate)


def unwrap_model(model: nn.Module) -> nn.Module:
    """剥离 DDP 与 torch.compile 包装,拿到原始模块。

    torch.compile 返回的 OptimizedModule 通过 _orig_mod 暴露原模块,DDP 通过 module 暴露。
    先 compile 再 DDP 会嵌套两层,故迭代解包直到没有包装为止。
    """
    m = model
    while True:
        if isinstance(m, DDP):
            m = m.module
        elif hasattr(m, "_orig_mod"):
            m = m._orig_mod
        else:
            return m


def _all_ranks_true(local_ok: bool, device: torch.device, is_distributed: bool) -> bool:
    """Return True only if every rank reports ``local_ok``."""
    if not is_distributed:
        return local_ok
    flag = torch.tensor([int(local_ok)], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return flag.item() > 0


def _grads_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        grad = param.grad
        if grad is not None and not torch.isfinite(grad).all():
            return False
    return True


def _params_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return False
    return True


def _sync_after_rank0_work(
    *,
    is_distributed: bool,
    device: torch.device,
    rank0_work: bool,
) -> None:
    """Barrier only when rank 0 ran eval/save/plot (non-collective) work."""
    if not is_distributed:
        return
    flag = torch.tensor([int(rank0_work)], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        dist.barrier()


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


def _eval_loss_branch(model: nn.Module) -> str | None:
    """BDELF eval uses decode CE; AR/BD3LM use the default training loss."""
    if uses_dual_branch_logging(model):
        return "decode"
    return None


def _sample_synced_train_branch(
    model: nn.Module,
    device: torch.device,
    *,
    is_distributed: bool,
) -> str:
    """Pick denoise/decode once per step; broadcast so every DDP rank matches."""
    raw = unwrap_model(model)
    p = float(raw.backbone.decoder_prob)
    if is_distributed:
        pick_decode = torch.zeros(1, device=device, dtype=torch.float32)
        if dist.get_rank() == 0:
            pick_decode[0] = float(torch.rand((), device=device) < p)
        dist.broadcast(pick_decode, src=0)
        return "decode" if pick_decode.item() > 0.5 else "denoise"
    return "decode" if torch.rand((), device=device) < p else "denoise"


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def eval_model_ppl(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    pbar_parent: tqdm | None = None,
) -> tuple[float, float]:
    """Eval split loss and exp(loss) PPL from the training model."""
    was_training = model.training
    model.eval()
    branch = _eval_loss_branch(model)
    use_amp = device.type == "cuda"
    total_loss = 0.0
    batches = 0
    if len(loader) == 0:
        return float("nan"), float("nan")

    batch_iter: DataLoader | tqdm = loader
    if pbar_parent is not None:
        pbar_parent.clear()
        batch_iter = tqdm(
            loader,
            desc="eval",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            total=len(loader),
        )
    try:
        for eval_batch in batch_iter:
            eval_batch = eval_batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = forward_loss(model, eval_batch, branch=branch)
            total_loss += float(loss.item())
            batches += 1
    finally:
        if isinstance(batch_iter, tqdm):
            batch_iter.close()
        if pbar_parent is not None:
            pbar_parent.refresh()
        if was_training:
            model.train()

    avg_loss = total_loss / max(1, batches)
    avg_ppl = loss_to_ppl(avg_loss)
    if batches > 0:
        label = "decode ce" if branch == "decode" else "loss"
        summary = f"eval: {label} {avg_loss:.4f} ppl {avg_ppl:.2f}"
        if pbar_parent is not None:
            tqdm.write(f"{_TRAIN_LOG} {summary}")
        else:
            _train_log(summary)
    return avg_loss, avg_ppl


def prepare_gpt2_eval_batch(
    batch: torch.Tensor,
    layout: FL_TokenLayout,
    *,
    gpt2_vocab_size: int,
    fill_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map extended-vocab token ids into the GPT-2 baseline range for CE/PPL."""
    input_ids = batch.clone()
    labels = batch.clone()
    oov = input_ids >= gpt2_vocab_size
    input_ids[oov] = fill_token_id
    for token_id in (layout.bos_token_id, layout.eos_token_id, layout.pad_token_id):
        labels[labels == token_id] = -100
    return input_ids, labels


def prepare_gpt2_eval_batch_retokenize(
    batch: torch.Tensor,
    *,
    src_tokenizer_name: str,
    gpt2_vocab_size: int,
    fill_token_id: int,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode with the train tokenizer, then re-encode with GPT-2 for Gen. PPL.

    Required when the train model uses a non-GPT-2 vocabulary (e.g. ELF / T5).
    """
    from transformers import AutoTokenizer

    from tokenizer import get_tokenizer

    src_tok = get_tokenizer(src_tokenizer_name)
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    if gpt2_tok.pad_token_id is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token

    texts = [
        src_tok.decode(row.tolist(), skip_special_tokens=True)
        for row in batch.detach().cpu()
    ]
    encoded = gpt2_tok(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    labels = input_ids.clone()
    pad_id = int(gpt2_tok.pad_token_id)
    labels[labels == pad_id] = -100
    oov = input_ids >= gpt2_vocab_size
    input_ids[oov] = fill_token_id
    return input_ids, labels


def _gen_eval_sampling_cfg(cfg: FL_TrainConfig) -> dict[str, Any]:
    sampling_cfg: dict[str, Any] = {"use_fast_infer": cfg.eval_use_fast_infer}
    if cfg.model == "bd3lm":
        sampling_cfg["num_steps"] = cfg.eval_gen_steps
    elif cfg.model == "elf":
        # Keep eval sampling lighter than the default 32–64-step SDE.
        sampling_cfg["num_sampling_steps"] = min(16, cfg.eval_gen_steps)
        sampling_cfg["sampling_method"] = "ode"
        sampling_cfg["temperature"] = 0.0  # paper decode: argmax
    return sampling_cfg


def load_gen_eval_baseline(cfg: FL_TrainConfig) -> nn.Module:
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[cfg.gen_eval_model_dtype]
    device = cfg.gen_eval_model_device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gen_eval_model_device=cuda but no CUDA device was found")
    model = get_hf_model(cfg.gen_eval_model, torch_dtype=torch_dtype, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def eval_one_batch_gen_ppl(
    train_model: nn.Module,
    gpt2_model: nn.Module,
    *,
    cfg: FL_TrainConfig,
    train_device: torch.device,
    train_amp_dtype: torch.dtype,
    token_layout: FL_TokenLayout,
    seed: int,
    pbar_parent: tqdm | None = None,
) -> tuple[float, float]:
    """Unconditional one-batch gen. PPL: sample with train model, score via gpt2-large."""
    was_training = train_model.training
    train_model.eval()
    gpt2_model.eval()
    gpt2_device = next(gpt2_model.parameters()).device
    gpt2_vocab_size = int(getattr(gpt2_model.config, "vocab_size", 50257))
    fill_token_id = int(
        getattr(gpt2_model.config, "eos_token_id", None) or 50256,
    )
    seqlen = int(cfg.extra.get("chunk_length", 1024))
    use_train_amp = train_device.type == "cuda"
    use_gpt2_amp = gpt2_device.type == "cuda"
    gpt2_amp_dtype = get_amp_dtype(cfg.gen_eval_model_dtype)

    if pbar_parent is not None:
        pbar_parent.clear()
        tqdm.write(
            f"{_TRAIN_LOG} eval/gen: sampling {cfg.batch_size} x {seqlen} "
            f"(seed={seed}) ...",
        )

    # Isolate sampling RNG from the training loop.
    devices = [train_device] if train_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if train_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        gen_model = unwrap_model(train_model)
        with torch.amp.autocast(
            "cuda", dtype=train_amp_dtype, enabled=use_train_amp,
        ):
            generated, _nfe = gen_model.generate(
                num_samples=cfg.batch_size,
                seqlen=seqlen,
                for_eval=True,
                sampling_cfg=_gen_eval_sampling_cfg(cfg),
            )

    if cfg.model == "elf":
        src_tok_name = get_preprocess(cfg.preprocess).tokenizer
        input_ids, labels = prepare_gpt2_eval_batch_retokenize(
            generated,
            src_tokenizer_name=src_tok_name,
            gpt2_vocab_size=gpt2_vocab_size,
            fill_token_id=fill_token_id,
            device=gpt2_device,
            max_length=seqlen,
        )
    else:
        generated = generated.to(gpt2_device, non_blocking=True)
        input_ids, labels = prepare_gpt2_eval_batch(
            generated,
            token_layout,
            gpt2_vocab_size=gpt2_vocab_size,
            fill_token_id=fill_token_id,
        )
    with torch.amp.autocast("cuda", dtype=gpt2_amp_dtype, enabled=use_gpt2_amp):
        outputs = gpt2_model(input_ids, labels=labels)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        gen_loss = float(loss.item())
    gen_ppl = loss_to_ppl(gen_loss)

    if was_training:
        train_model.train()
    if pbar_parent is not None:
        pbar_parent.refresh()

    summary = (
        f"eval/gen ({cfg.gen_eval_model}): loss {gen_loss:.4f} ppl {gen_ppl:.2f}"
    )
    if pbar_parent is not None:
        tqdm.write(f"{_TRAIN_LOG} {summary}")
    else:
        _train_log(summary)
    return gen_loss, gen_ppl


# =============================================================================
# Metrics logging and plots
# =============================================================================


def append_csv_row(csv_path: Path, fields: list[str], row: dict[str, Any]) -> None:
    if csv_path.exists():
        ensure_csv_schema(csv_path, fields)
        write_header = False
    else:
        write_header = True
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def ensure_csv_schema(csv_path: Path, fields: list[str]) -> None:
    """Rewrite CSV if the on-disk header is missing newly added columns."""
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        if old_fields == fields:
            return
        rows = list(reader)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def truncate_csv_for_resume(csv_path: Path, start_step: int) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        if not old_fields:
            return 0
        # Prefer the canonical schema so resume can introduce new columns.
        fieldnames = EVAL_CSV_FIELDS if csv_path.name == "eval_log.csv" else old_fields
        if csv_path.name == "train_log.csv":
            fieldnames = TRAIN_CSV_FIELDS
        rows_by_step: dict[int, dict[str, str]] = {}
        for row in reader:
            step = int(row["step"])
            if step < start_step:
                rows_by_step[step] = row
    rows = [rows_by_step[s] for s in sorted(rows_by_step)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
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


def _decode_ce_train_series(
    train_rows: list[dict[str, str]],
) -> tuple[list[int], list[float], list[float]]:
    """Train decode-CE points for plotting (BDELF dual-branch)."""
    steps: list[int] = []
    ppls: list[float] = []
    lrs: list[float] = []
    for row in train_rows:
        if row.get("loss_branch") != "decode":
            continue
        ppl = _parse_float(row.get("train_ppl"))
        if ppl is None:
            ce = _parse_float(row.get("decode_ce"))
            if ce is not None:
                ppl = loss_to_ppl(ce)
        if ppl is None:
            continue
        steps.append(int(row["step"]))
        ppls.append(ppl)
        lrs.append(float(row["lr"]))
    return steps, ppls, lrs


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
    train_lr = [float(r["lr"]) for r in train_rows]

    dual_branch = any(r.get("loss_branch") in ("denoise", "decode") for r in train_rows)
    if dual_branch:
        train_plot_steps, train_ppl, _ = _decode_ce_train_series(train_rows)
    else:
        train_plot_steps = train_steps
        train_ppl = [_parse_float(r.get("train_ppl")) for r in train_rows]

    eval_steps = [int(r["step"]) for r in eval_rows]
    eval_ppl = [
        _parse_float(r.get("eval_ppl") or r.get("gpt2_ppl")) for r in eval_rows
    ]
    gen_ppl = [_parse_float(r.get("gen_ppl")) for r in eval_rows]

    for cap, filename in ((1000.0, "ppl_under_1000.png"), (100.0, "ppl_under_100.png")):
        t_steps, t_ppls = zip(
            *[
                (s, p)
                for s, p in zip(train_plot_steps, train_ppl)
                if p is not None and p <= cap
            ]
        ) if any(p is not None and p <= cap for p in train_ppl) else ([], [])

        e_steps, e_ppls = zip(
            *[(s, p) for s, p in zip(eval_steps, eval_ppl) if p is not None and p <= cap]
        ) if any(p is not None and p <= cap for p in eval_ppl) else ([], [])

        g_steps, g_ppls = zip(
            *[(s, p) for s, p in zip(eval_steps, gen_ppl) if p is not None and p <= cap]
        ) if any(p is not None and p <= cap for p in gen_ppl) else ([], [])

        if not t_steps and not e_steps and not g_steps:
            continue

        fig, ax_ppl = plt.subplots(figsize=(10, 4.5))

        if t_steps:
            train_label = (
                "train decode ppl (exp ce)"
                if dual_branch
                else "train ppl (exp loss)"
            )
            ax_ppl.plot(
                t_steps, t_ppls, color="#4C72B0", alpha=0.55, linewidth=1.2,
                label=train_label, zorder=1,
            )
        if e_steps:
            ax_ppl.plot(
                e_steps, e_ppls, color="#D62728", linewidth=2.8, marker="o",
                markersize=4, label="eval ppl (exp loss)", zorder=5,
            )
        if g_steps:
            ax_ppl.plot(
                g_steps, g_ppls, color="#2CA02C", linewidth=2.4, marker="s",
                markersize=4, label="gen ppl (gpt2-large)", zorder=6,
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
        elif loss_branch == "decode":
            ppl = loss_to_ppl(train_loss)
            row["decode_ce"] = round(train_loss, 6)
            row["train_ppl"] = round(ppl, 4)
            row["train_loss"] = row["decode_ce"]
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
) -> list[str]:
    pct = 100.0 * (step + 1) / max_steps
    return [f"[{step + 1}/{max_steps} ({pct:.1f}%)] {_train_metrics_text(row)}"]


def _rank0_log(msg: str, pbar: tqdm | None) -> None:
    line = f"{_TRAIN_LOG} {msg}"
    if pbar is not None:
        tqdm.write(line)
    else:
        print(line, flush=True)


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: FL_TrainConfig,
    model_meta: dict[str, Any],
) -> None:
    """Atomically write a checkpoint (tmp file + ``os.replace``).

    Direct ``torch.save`` to the final path can leave a truncated file if the
    process is killed mid-write; resume would then fail on a corrupt latest ckpt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer | Any,
    device: torch.device,
) -> None:
    if hasattr(optimizer, "muon") and hasattr(optimizer, "adamw"):
        _move_optimizer_state_to_device(optimizer.muon, device)
        _move_optimizer_state_to_device(optimizer.adamw, device)
        return
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
    restore_rng: bool = True,
) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    saved_cfg = ck.get("train_config") or {}
    if saved_cfg.get("name") and saved_cfg["name"] != cfg.name:
        raise ValueError(
            f"checkpoint run={saved_cfg['name']!r} does not match current {cfg.name!r}"
        )
    saved_use_muon = bool(saved_cfg.get("use_muon", False))
    if saved_use_muon != cfg.use_muon:
        raise ValueError(
            f"checkpoint use_muon={saved_use_muon} does not match current {cfg.use_muon}"
        )
    saved_meta = ck.get("model_meta") or {}
    if saved_meta.get("name") and saved_meta["name"] != model_meta.get("name"):
        raise ValueError(
            f"checkpoint model {saved_meta.get('name')!r} "
            f"does not match current {model_meta.get('name')!r}"
        )

    raw = unwrap_model(model)
    raw.load_state_dict(ck["model"])
    opt_state = ck["optimizer"]
    if cfg.use_muon:
        if not isinstance(opt_state, dict) or opt_state.get("kind") != "muon_hybrid":
            raise ValueError(
                "checkpoint optimizer is not hybrid Muon state; "
                "cannot resume with use_muon=True"
            )
    elif isinstance(opt_state, dict) and opt_state.get("kind") == "muon_hybrid":
        raise ValueError(
            "checkpoint optimizer is hybrid Muon state; "
            "cannot resume with use_muon=False"
        )
    optimizer.load_state_dict(opt_state)
    _move_optimizer_state_to_device(optimizer, device)

    grads = ck.get("grads")
    if grads is not None:
        for p, g in zip(raw.parameters(), grads):
            p.grad = g.to(device) if g is not None else None

    # The RNG snapshot comes from rank 0; non-zero ranks keep their
    # set_seed(seed + rank) state so per-rank noise stays decorrelated.
    rng = ck.get("rng")
    if restore_rng and rng is not None:
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return int(ck["step"])


# =============================================================================
# Train loop
# =============================================================================


def set_seed(seed: int, rank: int) -> None:
    s = seed + rank
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def train_loop(
    model: nn.Module,
    cfg: FL_TrainConfig,
    model_meta: dict[str, Any],
    train_ds: TokenChunkDataset,
    eval_loader: DataLoader | None,
    gpt2_model: nn.Module | None,
    gen_token_layout: FL_TokenLayout | None,
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

    # 在 DDP 之前 compile:backbone 是固定的 2*chunk_length 序列 + 静态 branch kwarg,
    # Inductor 能拿到稳定 shape、每个分支一张图。所有参数每步都被使用(mode embedding 在
    # forward 里被触碰),故 DDP 可不带 find_unused_parameters。
    compile_model = bool(cfg.extra.get("compile", False)) and device.type == "cuda"
    if compile_model:
        if rank == 0:
            _train_log(
                "torch.compile enabled; the first denoise/decode steps are slow "
                "while Inductor compiles kernels "
                f"(triton={os.environ.get('TRITON_CACHE_DIR')}, "
                f"inductor={os.environ.get('TORCHINDUCTOR_CACHE_DIR')})",
            )
        model = torch.compile(model)

    if is_distributed:
        ddp_kwargs: dict[str, Any] = {
            "device_ids": [device.index],
            "output_device": device.index,
        }
        model = DDP(model, **ddp_kwargs)

    raw = unwrap_model(model)
    optimizer = build_optimizer(raw, cfg)

    train_csv = run_dir / "train_log.csv"
    eval_csv = run_dir / "eval_log.csv"
    latest_ckpt = run_dir / "checkpoint_latest.pt"

    step = 0
    optimizer.zero_grad(set_to_none=True)

    resume_from_ckpt = cfg.resume and latest_ckpt.is_file()
    if is_distributed:
        # All ranks must agree on resuming; trust rank 0's view of the file.
        flag = torch.tensor([int(resume_from_ckpt)], device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        resume_from_ckpt = bool(flag.item())

    if resume_from_ckpt:
        # Every rank must load weights/optimizer state: DDP only broadcasts
        # parameters at construction time (above), so a rank-0-only load would
        # leave the other ranks on their random init.
        step = load_checkpoint(
            latest_ckpt, model, optimizer, device,
            cfg=cfg, model_meta=model_meta, restore_rng=(rank == 0),
        )
        if rank == 0:
            kept_train = truncate_csv_for_resume(train_csv, step)
            kept_eval = truncate_csv_for_resume(eval_csv, step)
            update_ppl_plots(train_csv, eval_csv, run_dir)
            _train_log(
                f"Resuming from checkpoint: step {step} "
                f"(train_log {kept_train} rows, eval_log {kept_eval} rows)",
            )
        if step >= cfg.max_steps:
            if rank == 0:
                _train_log(
                    f"Reached max_steps={cfg.max_steps}; training is already complete"
                )
            return

    if is_distributed:
        dist.barrier()

    dual_branch = uses_dual_branch_logging(model)
    if rank == 0 and dual_branch:
        decoder_prob = float(cfg.extra.get("decoder_prob", 0.2))
        _train_log(
            f"{cfg.model.upper()} dual-branch: random denoise/decode sampling "
            f"(decode prob={decoder_prob:g}); "
            "metrics/plots use decode CE; only decode steps count toward token budget",
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

            lr = scaled_lr(step, cfg, cfg.learning_rate)
            if cfg.use_muon:
                schedule_optimizer_lrs(
                    optimizer,
                    adam_lr=lr,
                    muon_lr=scaled_lr(step, cfg, cfg.muon_learning_rate),
                )
            else:
                for group in optimizer.param_groups:
                    group["lr"] = lr

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                train_branch = (
                    _sample_synced_train_branch(
                        model, device, is_distributed=is_distributed,
                    )
                    if dual_branch
                    else None
                )
                micro_loss = forward_loss(model, batch, branch=train_branch)

            loss_ok = _all_ranks_true(
                bool(torch.isfinite(micro_loss).item()),
                device,
                is_distributed,
            )
            if loss_ok:
                (micro_loss / cfg.grad_accum_steps).backward()
                step_backward_done = True
            elif rank == 0:
                if not _params_are_finite(raw):
                    _train_log(
                        f"Non-finite loss at step {step} with corrupted weights; "
                        "stop and resume from an earlier checkpoint",
                    )
                else:
                    _train_log(f"Skipping backward at step {step}: non-finite loss")
            if not loss_ok and not _params_are_finite(raw):
                raise RuntimeError(
                    f"Non-finite model weights at step {step}; "
                    "resume from an earlier checkpoint",
                )

            if (step + 1) % cfg.grad_accum_steps == 0:
                grads_ok = _all_ranks_true(
                    _grads_are_finite(model),
                    device,
                    is_distributed,
                )
                if grads_ok:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                elif rank == 0:
                    _train_log(
                        f"Skipping optimizer step at step {step}: non-finite gradients",
                    )
                optimizer.zero_grad(set_to_none=True)

            train_loss = micro_loss.item() if loss_ok else float("nan")
            loss_branch = getattr(raw, "last_loss_branch", "") if dual_branch else ""
            elapsed = time.time() - t0
            seq_tokens = batch.size(0) * (
                batch.size(1) if uses_full_sequence(model) else batch.size(1) - 1
            )
            if dual_branch:
                effective_tokens = seq_tokens if loss_branch == "decode" else 0
            else:
                effective_tokens = seq_tokens
            tokens_per_sec = effective_tokens / max(elapsed, 1e-6)

            row = build_train_row(
                step, train_loss, lr, tokens_per_sec,
                dual_branch=dual_branch, loss_branch=loss_branch,
            )

            rank0_sync = False
            if rank == 0:
                if dual_branch and loss_branch == "decode":
                    postfix = {
                        "ce": f"{train_loss:.3f}",
                        "ppl": f"{loss_to_ppl(train_loss):.1f}",
                        "lr": f"{lr:.2e}",
                        "tok_s": f"{tokens_per_sec:.0f}",
                    }
                elif dual_branch:
                    postfix = {
                        "branch": "denoise",
                        "mse": f"{train_loss:.3f}",
                        "lr": f"{lr:.2e}",
                    }
                else:
                    postfix = {
                        "loss": f"{train_loss:.3f}",
                        "lr": f"{lr:.2e}",
                        "tok_s": f"{tokens_per_sec:.0f}",
                    }
                pbar.set_postfix(**postfix)
                pbar.update(1)
                append_csv_row(train_csv, TRAIN_CSV_FIELDS, row)

                interval_done = (
                    (step + 1) % cfg.eval_step == 0 or (step + 1) >= cfg.max_steps
                )
                if interval_done:
                    if (
                        (step + 1) % cfg.eval_step == 0
                        and eval_loader is not None
                    ):
                        eval_loss, eval_ppl = eval_model_ppl(
                            unwrap_model(model),
                            eval_loader,
                            device,
                            amp_dtype,
                            pbar_parent=pbar,
                        )
                        gen_loss: float | None = None
                        gen_ppl: float | None = None
                        if gpt2_model is not None and gen_token_layout is not None:
                            gen_loss, gen_ppl = eval_one_batch_gen_ppl(
                                model,
                                gpt2_model,
                                cfg=cfg,
                                train_device=device,
                                train_amp_dtype=amp_dtype,
                                token_layout=gen_token_layout,
                                seed=cfg.seed + step,
                                pbar_parent=pbar,
                            )
                        eval_row = {
                            "step": step,
                            "eval_loss": round(eval_loss, 6),
                            "eval_ppl": round(eval_ppl, 4),
                            "gen_loss": (
                                round(gen_loss, 6) if gen_loss is not None else ""
                            ),
                            "gen_ppl": (
                                round(gen_ppl, 4) if gen_ppl is not None else ""
                            ),
                            "lr": lr,
                        }
                        append_csv_row(eval_csv, EVAL_CSV_FIELDS, eval_row)
                    rank0_sync = True

                    for line in format_interval_summary(step, cfg.max_steps, row):
                        _rank0_log(line, pbar)

                if (step + 1) % cfg.log_plot_step == 0:
                    update_ppl_plots(train_csv, eval_csv, run_dir)
                    rank0_sync = True

                # save_step / snapshot_step are independent intervals; do not nest
                # snapshot under save (snapshot_step need not divide save_step).
                next_step = step + 1
                do_save = next_step % cfg.save_step == 0
                do_snapshot = next_step % cfg.snapshot_step == 0
                if do_save or do_snapshot:
                    # Always refresh latest when writing any durable checkpoint.
                    save_checkpoint(
                        latest_ckpt, model, optimizer, next_step, cfg, model_meta,
                    )
                    if do_snapshot:
                        save_checkpoint(
                            run_dir / f"checkpoint_step_{next_step:07d}.pt",
                            model, optimizer, next_step, cfg, model_meta,
                        )
                    _rank0_log(f"  [ckpt] saved at step {next_step}", pbar)
                    rank0_sync = True

            _sync_after_rank0_work(
                is_distributed=is_distributed,
                device=device,
                rank0_work=rank0_sync,
            )

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

    # Always persist the finished run. Periodic saves only fire when
    # max_steps is a multiple of save_step/snapshot_step; the final write
    # covers the common case where it is not.
    if rank == 0:
        if pbar is not None:
            pbar.close()
        save_checkpoint(latest_ckpt, model, optimizer, step, cfg, model_meta)
        final_snapshot = run_dir / f"checkpoint_step_{step:07d}.pt"
        save_checkpoint(final_snapshot, model, optimizer, step, cfg, model_meta)
        update_ppl_plots(train_csv, eval_csv, run_dir)
        _train_log(
            f"Training finished after {step} steps; "
            f"saved {latest_ckpt.name} and {final_snapshot.name} in {run_dir}"
        )
    if is_distributed:
        # Keep peers alive until rank 0 finishes the (often multi-GB) write;
        # otherwise destroy_process_group can race with the final save.
        dist.barrier()


# =============================================================================
# CLI
# =============================================================================


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
            "--dataset owt --preprocess default\n"
            "  python train.py --model elf --config 100m-fast "
            "--dataset owt --preprocess elf\n"
            "  python train.py --model elf --config 100m-full "
            "--dataset owt --preprocess elf\n\n"
            f"Available models: {', '.join(models)}\n"
            f"Available datasets: {', '.join(datasets)}\n"
            f"Available preprocess configs: {', '.join(preprocess_names)}\n"
            "Train configs: {{100m,300m,900m}}-{{fast,full,ultra}}"
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
        help="Train config name, e.g. 100m-fast / 900m-full / 900m-ultra",
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
            f"Naming format: {{100m,300m,900m}}-{{fast,full,ultra}}"
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
        launch_world_size = _resolve_launch_world_size(args.train_config)
        cfg = get_train_config(
            args.model,
            args.train_config,
            dataset=args.dataset,
            preprocess=args.preprocess,
            world_size=launch_world_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Failed to load train config: {exc}") from exc

    if args.run_name:
        cfg.name = args.run_name
    return args.model, size, cfg


# =============================================================================
# Entrypoint
# =============================================================================


def run_training(model_name: str, model_size: str, cfg: FL_TrainConfig) -> None:
    rank, world_size, device, is_distributed = setup_distributed(cfg)
    set_seed(cfg.seed, rank)

    if rank == 0:
        _train_log(f"Model: {model_name}/{model_size}")
        _train_log(f"Train config: {cfg.name} ({cfg.variant})")
        if cfg.use_muon:
            _train_log(
                f"Optimizer: Muon+AdamW hybrid "
                f"(muon_lr={cfg.muon_learning_rate}, adam_lr={cfg.learning_rate})"
            )
        _train_log(f"Data: dataset={cfg.dataset}, preprocess={cfg.preprocess}")
        _train_log(f"Device: {device}, world_size={world_size}")
        if cfg.target_tokens is not None:
            opt_steps = int(cfg.extra.get("max_optimizer_steps", 0)) or (
                cfg.max_steps // max(1, cfg.grad_accum_steps)
            )
            if cfg.model in ("bdelf", "elf"):
                _train_log(
                    f"Token budget: {cfg.target_tokens:,} decode-equivalent tokens "
                    f"({cfg.effective_tokens_per_optimizer_step:,}/opt-step decode, "
                    f"{cfg.tokens_per_optimizer_step:,}/opt-step raw) → "
                    f"{opt_steps:,} optimizer steps "
                    f"({cfg.max_steps:,} micro-steps, accum={cfg.grad_accum_steps})",
                )
            else:
                _train_log(
                    f"Token budget: {cfg.target_tokens:,} tokens "
                    f"({cfg.tokens_per_optimizer_step:,}/opt-step) → "
                    f"{opt_steps:,} optimizer steps "
                    f"({cfg.max_steps:,} micro-steps, accum={cfg.grad_accum_steps})",
                )

    try:
        # On a cache miss only rank 0 downloads/builds; the other ranks wait
        # and then attach to the finished cache. Concurrent builds would write
        # the same shard/manifest files and corrupt the cache.
        if is_distributed and rank != 0:
            dist.barrier()
        preprocessed = get_preprocessed(cfg.preprocess, cfg.dataset)
        if is_distributed and rank == 0:
            dist.barrier()
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
    if rank == 0:
        if len(eval_ds) == 0:
            _train_log("WARNING: eval dataset is empty; eval will be skipped")
        else:
            eval_loader = DataLoader(
                eval_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_input_ids,
            )

    model_cfg_path = resolve_model_config_path(model_name, model_size)
    import yaml

    with open(model_cfg_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}

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
        _train_log(
            f"gen. ppl: 1 batch / eval via {cfg.gen_eval_model} "
            f"({cfg.gen_eval_model_dtype} on {cfg.gen_eval_model_device})",
        )

    gpt2_model: nn.Module | None = None
    gen_token_layout: FL_TokenLayout | None = None
    if rank == 0:
        gen_token_layout = token_layout_from_cfg(model_cfg)
        gpt2_model = load_gen_eval_baseline(cfg)
        _train_log(
            f"Loaded gen-eval baseline {cfg.gen_eval_model} "
            f"on {cfg.gen_eval_model_device}",
        )

    train_loop(
        model,
        cfg,
        model_meta,
        train_ds,
        eval_loader,
        gpt2_model,
        gen_token_layout,
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

        # Build the dataset/preprocess cache once in the parent so workers hit
        # a warm cache; a cold build inside a worker could exceed the NCCL
        # barrier timeout that the other ranks wait on.
        try:
            get_preprocessed(cfg.preprocess, cfg.dataset)
        except FileNotFoundError as exc:
            raise SystemExit(f"Preprocessed data unavailable: {exc}") from exc

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
