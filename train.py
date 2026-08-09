#!/usr/bin/env python3
"""LLM pretraining entry point for bdelf."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

# Do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here:
# on RTX 5080 + WSL2 + torch cu130 it segfaults at the first CUDA alloc.

import logging
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

import hf_config  # noqa: F401
from models import (
    build_model,
    list_model_configs,
    list_models,
    resolve_model_config_path,
)
from dataset import list_datasets
from preprocess import get_preprocessed, list_preprocess
from train import (
    FL_TrainConfig,
    get_train_config,
    list_generate,
    list_train_configs,
    list_train_models,
    parse_train_overrides,
)
from train.batching import (
    TokenChunkDataset,
    build_eval_subset,
    collate_input_ids,
    fetch_train_batch,
    shard_eval_dataset,
)
from train.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from train.ema import ema_update, init_ema, swap_ema_weights
from train.eval import (
    eval_model_ppl,
    eval_one_batch_gen_ppl,
    forward_loss,
    get_amp_dtype,
    load_gen_eval_baseline,
    uses_dual_branch_logging,
    uses_full_sequence,
)
from train.metrics import (
    EVAL_CSV_FIELDS,
    TRAIN_CSV_FIELDS,
    _rank0_log,
    _train_log,
    append_csv_row,
    build_train_row,
    format_interval_summary,
    init_csv_header,
    loss_to_ppl,
    truncate_csv_for_resume,
    update_ppl_plots,
)
from train.muon import build_optimizer, scaled_lr, schedule_optimizer_lrs

# TF32: follow Inductor's recommendation. Filters: residual Dynamo/HF noise.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo does not know how to trace the builtin.*posix\.(l?stat).*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.*",
)
logging.getLogger("torch._dynamo.backends.distributed").setLevel(logging.ERROR)

# =============================================================================
# Inductor workaround / compile helpers
# =============================================================================


def _preload_frozen_encoders(model: nn.Module) -> None:
    """Load frozen HF encoders in eager mode before ``torch.compile``.

    ELF keeps T5 in a non-registered holder and loads it on first encode; if
    that happens under Dynamo, HF ``from_pretrained`` / peft / posix.stat paths
    get traced and spam warnings (and may re-load weights mid-compile).
    """
    backbone = getattr(model, "backbone", None)
    ensure = getattr(backbone, "_ensure_encoder", None)
    if callable(ensure):
        ensure()


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
        # Rank0 ELF gen-eval (SDE/32 + GPT-2 score) can exceed the default
        # 10min NCCL watchdog while peers wait on a 1-element all_reduce in
        # _sync_after_rank0_work; use a longer PG timeout to avoid false hangs.
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(minutes=60),
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != cfg.world_size:
        raise RuntimeError(
            f"Configured world_size={cfg.world_size}, but {world_size} processes were launched."
        )

    return rank, world_size, torch.device(f"cuda:{local_rank}"), True


ALLOWED_WORLD_SIZES = frozenset({1, 2, 4, 8})


def _resolve_launch_world_size() -> int:
    """Auto-detect visible GPU count; must be in {1, 2, 4, 8} (CPU-only → 1)."""
    if not torch.cuda.is_available():
        return 1
    n = torch.cuda.device_count()
    if n not in ALLOWED_WORLD_SIZES:
        raise SystemExit(
            f"训练需要 1/2/4/8 张可见 GPU，当前 device_count={n}"
        )
    return n


def _spawn_worker(
    local_rank: int,
    model_name: str,
    train_config: str,
    world_size: int,
    dataset: str,
    preprocess: str,
    generate: str,
    overrides: dict | None,
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
        generate=generate,
        world_size=world_size,
        overrides=overrides,
    )
    size = train_config.rsplit("-", 1)[0]
    run_training(model_name, size, cfg)


# =============================================================================
# Training helpers
# =============================================================================


def get_lr(step: int, cfg: FL_TrainConfig) -> float:
    return scaled_lr(step, cfg, cfg.learning_rate)


def _all_ranks_true(local_ok: bool, device: torch.device, is_distributed: bool) -> bool:
    """Return True only if every rank reports ``local_ok``."""
    if not is_distributed:
        return local_ok
    flag = torch.tensor([int(local_ok)], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return flag.item() > 0


def _wait_for_file(
    path: Path, *, timeout_s: float = 180.0, interval_s: float = 0.5,
) -> Path:
    """Wait until ``path`` is openable (BeeGFS: ``is_file`` can lie vs ``open``)."""
    path = path.resolve()
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(path, os.O_RDONLY)
            os.close(fd)
            return path
        except FileNotFoundError:
            pass
        except OSError:
            # ESTALE / transient BeeGFS client errors — keep polling.
            pass
        if time.time() >= deadline:
            raise FileNotFoundError(f"Timed out waiting for checkpoint: {path}")
        time.sleep(interval_s)


def _try_open_ckpt(path: Path, *, timeout_s: float = 2.0) -> Path | None:
    """Return ``path`` if briefly openable; BeeGFS ghosts can pass ``is_file``."""
    try:
        return _wait_for_file(path, timeout_s=timeout_s, interval_s=0.2)
    except FileNotFoundError:
        return None


def _pick_resume_ckpt(run_dir: Path) -> Path | None:
    """Prefer latest, then explicit resume_*, then newest step_* that can open."""
    candidates: list[Path] = [run_dir / "checkpoint_latest.pt"]
    candidates.extend(sorted(run_dir.glob("checkpoint_resume_*.pt"), reverse=True))
    step_ckpts = sorted(
        run_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: p.name,
        reverse=True,
    )
    candidates.extend(step_ckpts)
    seen: set[Path] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        opened = _try_open_ckpt(resolved)
        if opened is not None:
            return opened
    return None


def _resume_stage_job_id() -> str:
    """多进程共用的 /tmp staging 目录后缀。

    Slurm 用 ``SLURM_JOB_ID``；common / ``mp.spawn`` 各 rank 的 ``getpid()``
    不同——若用 pid，非 rank0 会等错路径并超时。优先显式 ``BDELF_JOB_ID``
    （父进程 spawn 前写入），否则退回 ``getppid()``。
    """
    for key in ("SLURM_JOB_ID", "BDELF_JOB_ID"):
        val = os.environ.get(key)
        if val:
            return str(val)
    return str(os.getppid())


def _stage_ckpt_to_local(
    src: Path, *, rank: int, is_distributed: bool,
) -> Path:
    """Rank0 copies BeeGFS ckpt to node-local /tmp; all ranks load from there."""
    local = Path(f"/tmp/bdelf-resume-{_resume_stage_job_id()}") / src.name
    if rank == 0:
        src = _wait_for_file(src)
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_suffix(local.suffix + ".partial")
        shutil.copyfile(src, partial)
        partial.replace(local)
        _train_log(f"Staged resume checkpoint to {local}")
    if is_distributed:
        dist.barrier()
    return _wait_for_file(local)


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
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    is_distributed: bool,
) -> None:
    amp_dtype = get_amp_dtype(cfg.dtype)
    from train.hardware import (
        HardwareMismatchError,
        detect_train_hardware,
        ensure_hardware_lock,
    )
    from train.run_path import checkpoint_run_dir_from_cfg

    run_dir = checkpoint_run_dir_from_cfg(cfg)

    # 硬件锁定（不进 hash）：探测可见 GPU，首次写入 hardware.json，续跑必须一致。
    hw_err = ""
    current_hw = None
    try:
        current_hw = detect_train_hardware()
    except HardwareMismatchError as exc:
        hw_err = str(exc)

    if is_distributed:
        gathered: list[str] | None = [""] * world_size if rank == 0 else None
        dist.gather_object(hw_err, gathered, dst=0)
        if rank == 0 and gathered is not None:
            for msg in gathered:
                if msg:
                    hw_err = msg
                    break
        err_box = [hw_err]
        dist.broadcast_object_list(err_box, src=0)
        hw_err = err_box[0]
    if hw_err:
        raise SystemExit(hw_err)

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            assert current_hw is not None
            hw = ensure_hardware_lock(run_dir, current_hw)
            _train_log(
                f"Hardware lock: {hw.gpu_count}× {hw.gpu_name} "
                f"({hw.memory_gb_per_gpu} GiB each)"
            )
            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"train": asdict(cfg), "model": model_meta},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            from train.hash_guide import upsert_hash_guide_row

            upsert_hash_guide_row(asdict(cfg))
        except HardwareMismatchError as exc:
            hw_err = str(exc)

    if is_distributed:
        err_box = [hw_err]
        dist.broadcast_object_list(err_box, src=0)
        hw_err = err_box[0]
    if hw_err:
        raise SystemExit(hw_err)

    # Eager-load frozen HF encoders before Dynamo can trace into from_pretrained.
    _preload_frozen_encoders(model)

    # Compile before DDP. ELF/BDELF mixed-branch keeps a single dynamic graph.
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
    # 预先写表头：log_plot_step 可能早于首次 eval，避免 plot 时 eval_log 尚不存在。
    if rank == 0:
        init_csv_header(train_csv, TRAIN_CSV_FIELDS)
        init_csv_header(eval_csv, EVAL_CSV_FIELDS)
    # Absolute path: relative cache/ can race under BeeGFS when ranks disagree
    # on cwd visibility right after a cross-node resume.
    latest_ckpt = (run_dir / "checkpoint_latest.pt").resolve()

    step = 0
    optimizer.zero_grad(set_to_none=True)
    ema_decay = float(cfg.extra.get("ema_decay", 0.0) or 0.0)
    ema_state: dict[str, torch.Tensor] | None = (
        init_ema(model) if ema_decay > 0.0 else None
    )
    if rank == 0 and ema_state is not None:
        _train_log(f"EMA enabled: decay={ema_decay:g}")

    # Rank0 picks an *openable* ckpt (BeeGFS ghosts: is_file ok, open fails).
    resume_src: Path | None = None
    if rank == 0 and cfg.resume:
        resume_src = _pick_resume_ckpt(run_dir)
        if resume_src is not None and resume_src != latest_ckpt:
            _train_log(
                f"checkpoint_latest not openable here; resuming from {resume_src.name}",
            )
    resume_from_ckpt = resume_src is not None
    if is_distributed:
        flag = torch.tensor([int(resume_from_ckpt)], device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        resume_from_ckpt = bool(flag.item())
        name_list = [resume_src.name if resume_src is not None else ""]
        dist.broadcast_object_list(name_list, src=0)
        if resume_from_ckpt:
            resume_src = (run_dir / name_list[0]).resolve()

    if resume_from_ckpt:
        # Every rank must load weights/optimizer state: DDP only broadcasts
        # parameters at construction time (above), so a rank-0-only load would
        # leave the other ranks on their random init.
        # Stage off BeeGFS first: concurrent multi-rank open of a ~1GB file
        # has hit FileNotFoundError even after is_file()/short waits.
        assert resume_src is not None
        resume_path = _stage_ckpt_to_local(
            resume_src, rank=rank, is_distributed=is_distributed,
        )
        step, loaded_ema = load_checkpoint(
            resume_path, model, optimizer, device,
            cfg=cfg, model_meta=model_meta, restore_rng=(rank == 0),
        )
        if ema_state is not None:
            if loaded_ema:
                for k, v in loaded_ema.items():
                    if k in ema_state:
                        ema_state[k].copy_(v.to(device=ema_state[k].device))
            elif rank == 0:
                _train_log(
                    "Checkpoint has no EMA state; re-initialized EMA from model weights",
                )
                ema_state = init_ema(model)
        if rank == 0:
            kept_train = truncate_csv_for_resume(train_csv, step)
            kept_eval = truncate_csv_for_resume(eval_csv, step)
            update_ppl_plots(
                train_csv,
                eval_csv,
                run_dir,
                tokens_per_micro_step=cfg.tokens_per_micro_step,
            )
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
    mixed_branch = bool(getattr(unwrap_model(model), "mixed_branch_training", False))
    if rank == 0 and dual_branch:
        if cfg.model == "late_ce":
            bb = unwrap_model(model).backbone
            decoder_prob = float(getattr(bb, "decoder_prob", 0.2))
            _train_log(
                f"LATE_CE 变体 B: per-example denoise:decode ≈ "
                f"{max(0.0, 1.0 - decoder_prob):g}:{decoder_prob:g} "
                f"+ 晚窗轨迹 CE (mode={getattr(bb, 'late_ce_mode', '?')}, "
                f"delta={getattr(bb, 'late_ce_delta', '?')}, "
                f"weight={getattr(bb, 'late_ce_weight', '?')}, "
                f"region={getattr(bb, 'late_ce_region', '?')}, "
                f"t={getattr(bb, 'time_schedule', '?')}); "
                "metrics: mse / ce / late_ce",
            )
        else:
            decoder_prob = float(
                getattr(unwrap_model(model).backbone, "decoder_prob", 0.2)
            )
            denoise_prob = max(0.0, 1.0 - decoder_prob)
            if mixed_branch:
                _train_log(
                    f"{cfg.model.upper()} dual-branch: per-example mix "
                    f"denoise:decode ≈ {denoise_prob:g}:{decoder_prob:g} "
                    "(official ELF train_step); metrics/plots use decode CE",
                )
            else:
                _train_log(
                    f"{cfg.model.upper()} dual-branch: denoise:decode ≈ "
                    f"{denoise_prob:g}:{decoder_prob:g} loss mix; "
                    "metrics/plots use decode CE",
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
                    group["lr"] = lr * float(group.get("lr_scale", 1.0))

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                # ELF/BDELF: mixed_branch_training → model mixes per-example internally.
                # 其它 dual-branch 模型：整步互斥 denoise/decode。
                if dual_branch and not mixed_branch:
                    train_branch: str | None = _sample_synced_train_branch(
                        model, device, is_distributed=is_distributed,
                    )
                else:
                    train_branch = None
                micro_loss = forward_loss(model, batch, branch=train_branch)

            loss_ok = _all_ranks_true(
                bool(torch.isfinite(micro_loss).item()),
                device,
                is_distributed,
            )
            if loss_ok:
                # Skip DDP all-reduce until the last micro-step of the accum
                # window; math is unchanged, communication drops ~accum×.
                sync_ctx = (
                    model.no_sync()
                    if is_distributed and (step + 1) % cfg.grad_accum_steps != 0
                    else nullcontext()
                )
                with sync_ctx:
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
                    if ema_state is not None:
                        ema_update(ema_state, model, ema_decay)
                elif rank == 0:
                    _train_log(
                        f"Skipping optimizer step at step {step}: non-finite gradients",
                    )
                optimizer.zero_grad(set_to_none=True)

            train_loss = micro_loss.item() if loss_ok else float("nan")
            # Prefer the sampled branch over model.last_loss_branch so logging
            # stays correct under torch.compile / DDP wrappers for BDELF.
            # ELF mixed path reads branch metrics from the backbone.
            raw_for_log = unwrap_model(model)
            if mixed_branch:
                loss_branch = "mixed"
                denoise_mse = raw_for_log.last_l2_loss
                decode_ce = raw_for_log.last_ce_loss
                late_ce = getattr(raw_for_log, "last_late_ce_loss", float("nan"))
            elif dual_branch:
                loss_branch = train_branch if train_branch else ""
                denoise_mse = None
                decode_ce = None
                late_ce = None
            else:
                loss_branch = ""
                denoise_mse = None
                decode_ce = None
                late_ce = None
            elapsed = time.time() - t0
            # 每微批消耗一整份数据 batch（与是否 denoise/decode 混合无关）。
            seq_tokens = batch.size(0) * (
                batch.size(1) if uses_full_sequence(model) else batch.size(1) - 1
            )
            tokens_per_sec = seq_tokens / max(elapsed, 1e-6)

            row = build_train_row(
                step,
                cfg.tokens_seen_after_step(step),
                train_loss,
                lr,
                tokens_per_sec,
                dual_branch=dual_branch, loss_branch=loss_branch,
                denoise_mse=denoise_mse, decode_ce=decode_ce,
                late_ce=late_ce,
            )

            # 在线 eval 在各卡均摊（held-out 分片 + gen 分担）；写盘仍仅 rank0。
            do_eval = (
                (step + 1) % cfg.eval_step == 0 and eval_loader is not None
            )
            if do_eval:
                with swap_ema_weights(model, ema_state):
                    eval_loss, eval_ppl = eval_model_ppl(
                        unwrap_model(model),
                        eval_loader,
                        device,
                        amp_dtype,
                        pbar_parent=pbar,
                        is_distributed=is_distributed,
                        log=(rank == 0),
                    )
                    gen_loss: float | None = None
                    gen_ppl: float | None = None
                    gen_uniq_mean: float | None = None
                    gen_nonempty_frac: float | None = None
                    if gpt2_model is not None:
                        (
                            gen_loss,
                            gen_ppl,
                            gen_uniq_mean,
                            gen_nonempty_frac,
                        ) = eval_one_batch_gen_ppl(
                            model,
                            gpt2_model,
                            cfg=cfg,
                            train_device=device,
                            train_amp_dtype=amp_dtype,
                            seed=cfg.seed + step,
                            pbar_parent=pbar,
                            rank=rank,
                            world_size=world_size,
                            is_distributed=is_distributed,
                            log=(rank == 0),
                        )
                if rank == 0:
                    eval_row = {
                        "step": step,
                        "tokens": cfg.tokens_seen_after_step(step),
                        "eval_loss": round(eval_loss, 6),
                        "eval_ppl": round(eval_ppl, 4),
                        "gen_loss": (
                            round(gen_loss, 6) if gen_loss is not None else ""
                        ),
                        "gen_ppl": (
                            round(gen_ppl, 4) if gen_ppl is not None else ""
                        ),
                        "gen_uniq_mean": (
                            round(gen_uniq_mean, 2)
                            if gen_uniq_mean is not None
                            else ""
                        ),
                        "gen_nonempty_frac": (
                            round(gen_nonempty_frac, 4)
                            if gen_nonempty_frac is not None
                            else ""
                        ),
                        "lr": lr,
                    }
                    append_csv_row(eval_csv, EVAL_CSV_FIELDS, eval_row)

            rank0_sync = False
            if rank == 0:
                if mixed_branch:
                    postfix = {
                        "loss": f"{train_loss:.3f}",
                        "lr": f"{lr:.2e}",
                        "tok_s": f"{tokens_per_sec:.0f}",
                    }
                    if decode_ce == decode_ce:
                        postfix["ce"] = f"{decode_ce:.3f}"
                        postfix["ppl"] = f"{loss_to_ppl(decode_ce):.1f}"
                    if denoise_mse == denoise_mse:
                        postfix["mse"] = f"{denoise_mse:.3f}"
                    if late_ce is not None and late_ce == late_ce:
                        postfix["late_ce"] = f"{late_ce:.3f}"
                elif dual_branch and loss_branch == "decode":
                    postfix = {
                        "ce": f"{train_loss:.3f}",
                        "ppl": f"{loss_to_ppl(train_loss):.1f}",
                        "lr": f"{lr:.2e}",
                        "tok_s": f"{tokens_per_sec:.0f}",
                    }
                elif dual_branch:
                    postfix = {
                        "mse": f"{train_loss:.3f}",
                        "lr": f"{lr:.2e}",
                        "tok_s": f"{tokens_per_sec:.0f}",
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
                    rank0_sync = True
                    for line in format_interval_summary(step, cfg.max_steps, row):
                        _rank0_log(line, pbar)

                if (step + 1) % cfg.log_plot_step == 0:
                    update_ppl_plots(
                        train_csv,
                        eval_csv,
                        run_dir,
                        tokens_per_micro_step=cfg.tokens_per_micro_step,
                    )
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
                        ema_state=ema_state,
                    )
                    if do_snapshot:
                        save_checkpoint(
                            run_dir / f"checkpoint_step_{next_step:07d}.pt",
                            model, optimizer, next_step, cfg, model_meta,
                            ema_state=ema_state,
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
            save_checkpoint(
                latest_ckpt, model, optimizer, next_step, cfg, model_meta,
                ema_state=ema_state,
            )
            update_ppl_plots(
                train_csv,
                eval_csv,
                run_dir,
                tokens_per_micro_step=cfg.tokens_per_micro_step,
            )
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
        save_checkpoint(
            latest_ckpt, model, optimizer, step, cfg, model_meta,
            ema_state=ema_state,
        )
        final_snapshot = run_dir / f"checkpoint_step_{step:07d}.pt"
        save_checkpoint(
            final_snapshot, model, optimizer, step, cfg, model_meta,
            ema_state=ema_state,
        )
        update_ppl_plots(
            train_csv,
            eval_csv,
            run_dir,
            tokens_per_micro_step=cfg.tokens_per_micro_step,
        )
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
            "--dataset owt --preprocess default --generate eval\n"
            "  python train.py --model ar2 --config 100m-full "
            "--dataset owt --preprocess default --generate eval\n"
            "  python train.py --model ar1_5 --config 100m-full "
            "--dataset owt --preprocess default --generate eval\n"
            "  python train.py --model bdelf --config 100m-full "
            "--dataset owt --preprocess default --generate eval\n"
            "  python train.py --model elf --config 100m-fast "
            "--dataset owt --preprocess elf --generate eval\n"
            "  python train.py --model elf --config 100m-full "
            "--dataset owt --preprocess elf --generate eval\n\n"
            f"Available models: {', '.join(models)}\n"
            f"Available datasets: {', '.join(datasets)}\n"
            f"Available preprocess configs: {', '.join(preprocess_names)}\n"
            "Train configs: 100m-{fast,full}\n"
            "Generate configs: config/generate/<model>/<name>.yaml\n"
            "Overrides: --set section.key=value "
            "(sections: optimizer, batch, schedule, eval, generate, model)"
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
        help="Train config name, e.g. 100m-fast / 100m-full",
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
        "--generate",
        required=True,
        help=(
            "Generate config name under config/generate/<model>/ "
            "(train online gen-eval: eval; standalone generate.py: generate)"
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="SECTION.KEY=VALUE",
        help=(
            "Override a train hyperparameter after loading YAML "
            "(repeatable). Examples: optimizer.learning_rate=1e-3, "
            "batch.batch_size=16, schedule.target_tokens=1e9, "
            "eval.gen_eval_samples=64, generate.temperature=0.8"
        ),
    )
    parser.add_argument(
        "--gpus",
        default=None,
        metavar="IDS",
        help=(
            "Visible CUDA device indices (comma-separated physical IDs), "
            "e.g. 0,1 or 2,3. Sets CUDA_VISIBLE_DEVICES before launch. "
            "Required on common remotes where all GPUs are visible; "
            "optional elsewhere (Slurm usually already masks devices)."
        ),
    )
    return parser


def apply_cuda_visible_devices(gpus: str | None) -> list[int] | None:
    """解析 ``--gpus``，写入 ``CUDA_VISIBLE_DEVICES``；返回物理卡号列表。

    须在首次 ``torch.cuda.device_count()`` / 分配设备之前调用。
    """
    if gpus is None:
        return None
    raw = gpus.strip()
    if not raw:
        raise SystemExit("--gpus 不能为空（例: --gpus 0,1）")
    ids: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            raise SystemExit(f"--gpus 格式无效: {gpus!r}（例: 0,1）")
        try:
            idx = int(p)
        except ValueError as exc:
            raise SystemExit(f"--gpus 含非整数 {p!r}: {gpus!r}") from exc
        if idx < 0:
            raise SystemExit(f"--gpus 卡号须 >= 0，收到 {idx}")
        ids.append(idx)
    if len(ids) != len(set(ids)):
        raise SystemExit(f"--gpus 有重复卡号: {gpus!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)
    return ids


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
            f"Model {args.model!r} has no train recipe. Available: {', '.join(train_models)}"
        )

    configs = list_train_configs(args.model)
    if args.train_config not in configs:
        raise SystemExit(
            f"Unknown train config {args.train_config!r}. {args.model} available: "
            f"{', '.join(configs)}\n"
            f"Naming format: 100m-{{fast,full}}"
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

    generate_names = list_generate(args.model)
    if args.generate not in generate_names:
        raise SystemExit(
            f"Unknown generate config {args.generate!r}. {args.model} available: "
            f"{', '.join(generate_names) or '<none>'}\n"
            f"Config directory: config/generate/{args.model}/"
        )

    try:
        overrides = parse_train_overrides(args.overrides)
    except ValueError as exc:
        raise SystemExit(f"Invalid --set override: {exc}") from exc

    try:
        launch_world_size = _resolve_launch_world_size()
        cfg = get_train_config(
            args.model,
            args.train_config,
            dataset=args.dataset,
            preprocess=args.preprocess,
            generate=args.generate,
            world_size=launch_world_size,
            overrides=overrides,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Failed to load train config: {exc}") from exc

    return args.model, size, cfg


# =============================================================================
# Entrypoint
# =============================================================================


def run_training(model_name: str, model_size: str, cfg: FL_TrainConfig) -> None:
    rank, world_size, device, is_distributed = setup_distributed(cfg)
    set_seed(cfg.seed, rank)

    if rank == 0:
        _train_log(f"Model: {model_name}/{model_size}")
        _train_log(
            f"Train run: {cfg.extra.get('run_relpath', cfg.name)} "
            f"(hash={cfg.name}, variant={cfg.variant})"
        )
        overrides = (cfg.extra.get("config_refs") or {}).get("overrides") or {}
        if overrides:
            flat = []
            for section, mapping in sorted(overrides.items()):
                for key, value in sorted(mapping.items()):
                    flat.append(f"{section}.{key}={value!r}")
            _train_log(f"Overrides: {', '.join(flat)}")
        if cfg.use_muon:
            _train_log(
                f"Optimizer: Muon+AdamW hybrid "
                f"(muon_lr={cfg.muon_learning_rate}, adam_lr={cfg.learning_rate})"
            )
        _train_log(
            f"Data: dataset={cfg.dataset}, preprocess={cfg.preprocess}, "
            f"generate={cfg.generate}"
        )
        _train_log(f"Device: {device}, world_size={world_size}")
        if cfg.target_tokens is not None:
            opt_steps = int(cfg.extra.get("max_optimizer_steps", 0)) or (
                cfg.max_steps // max(1, cfg.grad_accum_steps)
            )
            _train_log(
                f"Token budget: {cfg.target_tokens:,} data tokens "
                f"({cfg.tokens_per_optimizer_step:,}/opt-step) → "
                f"{opt_steps:,} optimizer steps "
                f"({cfg.max_steps:,} data micro-steps, accum={cfg.grad_accum_steps})",
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
    if len(eval_ds) == 0:
        if rank == 0:
            _train_log("WARNING: eval dataset is empty; eval will be skipped")
    else:
        eval_ds_local = shard_eval_dataset(
            eval_ds, rank=rank, world_size=world_size,
        )
        eval_loader = DataLoader(
            eval_ds_local,
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
    model_overrides = (
        (cfg.extra.get("config_refs") or {}).get("overrides") or {}
    ).get("model")
    if model_overrides:
        model_cfg = {**model_cfg, **dict(model_overrides)}

    if model_name == "cola":
        # Help Stage-2 auto-resolve matching cola_vae-{size}-{variant}* checkpoints.
        model_cfg = dict(model_cfg)
        model_cfg["train_variant"] = cfg.variant
        # Resume packs VAE+DiT; skip Stage-1 disk resolve so missing VAE dirs
        # do not block loading an existing cola run checkpoint.
        from train.run_path import checkpoint_run_dir_from_cfg

        latest_cola = checkpoint_run_dir_from_cfg(cfg) / "checkpoint_latest.pt"
        if cfg.resume and latest_cola.is_file():
            model_cfg["load_vae_weights"] = False

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
        if world_size > 1 and eval_loader is not None:
            _train_log(
                f"eval sharded across {world_size} ranks "
                f"(~{eval_run_size // world_size} samples/rank)",
            )
        _train_log(
            f"gen. ppl: {cfg.gen_eval_samples} samples / eval via "
            f"{cfg.gen_eval_model} "
            f"({cfg.gen_eval_model_dtype} on {cfg.gen_eval_model_device}"
            + (f", sharded×{world_size}" if world_size > 1 else "")
            + ")",
        )

    # 各卡加载 gpt2-large；缓存未命中时 rank0 先下，避免并发写 cache。
    gpt2_model: nn.Module | None
    if is_distributed:
        if rank == 0:
            gpt2_model = load_gen_eval_baseline(cfg)
            dist.barrier()
        else:
            dist.barrier()
            gpt2_model = load_gen_eval_baseline(cfg)
    else:
        gpt2_model = load_gen_eval_baseline(cfg)
    if rank == 0:
        _train_log(
            f"Loaded gen-eval baseline {cfg.gen_eval_model} "
            f"on {cfg.gen_eval_model_device} (all {world_size} ranks)",
        )

    train_loop(
        model,
        cfg,
        model_meta,
        train_ds,
        eval_loader,
        gpt2_model,
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
    # 须在 validate_args → device_count 之前屏蔽可见卡（common 机看得到全部 GPU）
    gpu_ids = apply_cuda_visible_devices(args.gpus)
    if gpu_ids is not None:
        _train_log(
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
            f"(physical gpu_ids={gpu_ids})"
        )
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
        # 子进程继承；供 _stage_ckpt_to_local 共用 /tmp 目录（勿用各 rank pid）
        os.environ.setdefault("BDELF_JOB_ID", str(os.getpid()))
        _train_log(
            f"Auto-spawning {cfg.world_size} processes (spawn), "
            f"MASTER_PORT={os.environ.get('MASTER_PORT', '29500')}",
        )
        mp.spawn(
            _spawn_worker,
            args=(
                model_name,
                args.train_config,
                cfg.world_size,
                args.dataset,
                args.preprocess,
                args.generate,
                (cfg.extra.get("config_refs") or {}).get("overrides") or None,
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
