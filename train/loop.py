"""训练环：``train_loop`` 与步内辅助函数。"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.train import FL_TrainConfig
from train.batching import (
    TokenChunkDataset,
    fetch_train_batch,
)
from train.checkpoint import (
    load_checkpoint,
    load_init_weights,
    save_checkpoint,
    unwrap_model,
)
from train.ema import ema_update, init_ema, swap_ema_weights
from train.eval import (
    eval_model_ppl,
    eval_one_batch_gen_ppl,
    forward_loss,
    get_amp_dtype,
    release_eval_cuda_scratch,
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
from train.scratch import (
    _preload_frozen_encoders,
    _scratch_job_id,
    _scratch_root,
)


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


def _stage_ckpt_to_local(
    src: Path, *, rank: int, is_distributed: bool,
) -> Path:
    """Rank0 把 BeeGFS ckpt 拷到节点本地 scratch；各 rank 从该副本 load。"""
    scratch = _scratch_root() or Path("/tmp")
    local = scratch / f"bdelf-resume-{_scratch_job_id()}" / src.name
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


def _release_staged_ckpt(
    local: Path, *, rank: int, is_distributed: bool,
) -> None:
    """各 rank 已 load 进内存后，rank0 删掉本 job 的 staged 副本。"""
    if is_distributed:
        dist.barrier()
    if rank != 0:
        return
    parent = local.parent
    if not parent.name.startswith("bdelf-resume-"):
        return
    try:
        local.unlink(missing_ok=True)
        local.with_suffix(local.suffix + ".partial").unlink(missing_ok=True)
        try:
            next(parent.iterdir())
        except StopIteration:
            parent.rmdir()
    except OSError:
        pass


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
    # 暂时断开训练环内在线 eval 与指标 log（csv / plot / last_*_loss / gen-eval）。
    # 下列 ``if False:`` 块保留原逻辑，后续有安排再接回；重构模型时不必维护这些路径。
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

    # Compile before DDP. ELF mixed-branch keeps a single dynamic graph.
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
    # 暂时断开指标 log：不预建 csv 表头。
    if False:
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
        _release_staged_ckpt(
            resume_path, rank=rank, is_distributed=is_distributed,
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
            # 暂时断开指标 log：续训不裁 csv、不重画 ppl 图。
            if False:
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
            _train_log(f"Resuming from checkpoint: step {step}")
        if step >= cfg.max_steps:
            if rank == 0:
                _train_log(
                    f"Reached max_steps={cfg.max_steps}; training is already complete"
                )
            return
    else:
        init_spec = cfg.extra.get("init_ckpt")
        if init_spec:
            repo = Path(__file__).resolve().parent
            init_path = Path(str(init_spec))
            if not init_path.is_absolute():
                init_path = (repo / init_path).resolve()
            init_path = _stage_ckpt_to_local(
                init_path, rank=rank, is_distributed=is_distributed,
            )
            loaded_ema, init_info = load_init_weights(init_path, model, device)
            _release_staged_ckpt(
                init_path, rank=rank, is_distributed=is_distributed,
            )
            if ema_state is not None:
                if loaded_ema:
                    n_ema = 0
                    for k, v in loaded_ema.items():
                        if k in ema_state:
                            ema_state[k].copy_(v.to(device=ema_state[k].device))
                            n_ema += 1
                    if rank == 0:
                        _train_log(
                            f"init-ckpt EMA 覆盖: {n_ema}/{len(ema_state)} 键"
                        )
                elif rank == 0:
                    _train_log("init-ckpt 无 EMA；已用加载后的模型权重重建 EMA")
                    ema_state = init_ema(model)
            if rank == 0:
                miss = init_info["missing"]
                unexp = init_info["unexpected"]
                _train_log(
                    f"init-ckpt: {init_info['path']} "
                    f"(src_model={init_info['src_model']!r} "
                    f"src_run={init_info['src_run']!r} "
                    f"src_step={init_info['src_step']}) "
                    f"overlap={init_info['n_overlap']}/{init_info['n_src']} "
                    f"missing={len(miss)} unexpected={len(unexp)}"
                )
                if miss:
                    _train_log(
                        "init-ckpt missing (ok if new buffers): "
                        + ", ".join(miss[:12])
                        + (" ..." if len(miss) > 12 else "")
                    )
                if unexp:
                    _train_log(
                        "init-ckpt unexpected: "
                        + ", ".join(unexp[:12])
                        + (" ..." if len(unexp) > 12 else "")
                    )
            step = 0

    if is_distributed:
        dist.barrier()

    dual_branch = uses_dual_branch_logging(model)
    mixed_branch = bool(getattr(unwrap_model(model), "mixed_branch_training", False))
    # 暂时断开指标 log：不打印各变体 denoise/decode 配比与 metrics 说明。
    if False:
        if rank == 0 and dual_branch:
            raw = unwrap_model(model)
            msg = raw.describe_training()
            if msg:
                _train_log(msg)
            else:
                decoder_prob = float(
                    getattr(raw.backbone, "decoder_prob", 0.2)
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
                # ELF: mixed_branch_training → model mixes per-example internally.
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
                    unwrap_model(model).on_optimizer_step(
                        model=model,
                        ema_state=ema_state,
                        opt_step=(step + 1) // cfg.grad_accum_steps,
                        variant=cfg.variant,
                        generate_sampling=cfg.generate_sampling,
                        rank=rank,
                        world_size=world_size,
                        is_distributed=is_distributed,
                        device=device,
                        log=_train_log if rank == 0 else None,
                    )
                elif rank == 0:
                    _train_log(
                        f"Skipping optimizer step at step {step}: non-finite gradients",
                    )
                optimizer.zero_grad(set_to_none=True)

            # 暂时断开训练期指标 log 与在线 eval（读 last_*_loss / csv / plot / gen-eval）。
            # 接回时：恢复上方 import，并将本段 if False 改回原逻辑。
            if False:
                train_loss = micro_loss.item() if loss_ok else float("nan")
                # Prefer the sampled branch over model.last_loss_branch so logging
                # stays correct under torch.compile / DDP wrappers.
                # ELF mixed path reads branch metrics from the backbone.
                raw_for_log = unwrap_model(model)
                if mixed_branch:
                    loss_branch = "mixed"
                    metrics = raw_for_log.train_metrics()
                elif dual_branch:
                    loss_branch = train_branch if train_branch else ""
                    metrics = None
                else:
                    loss_branch = ""
                    metrics = None
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
                    dual_branch=dual_branch,
                    loss_branch=loss_branch,
                    metrics=metrics,
                )

                # 在线 eval 在各卡均摊（held-out 分片 + gen 分担）；写盘仍仅 rank0。
                do_eval = (
                    (not cfg.skip_eval)
                    and (step + 1) % cfg.eval_step == 0
                    and eval_loader is not None
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
                    # EMA 权重已换回；丢掉 Flex mask / 采样临时块，避免池子钉在 gen 峰值。
                    release_eval_cuda_scratch(model, log=(rank == 0))
                    if is_distributed:
                        dist.barrier()
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

                if rank == 0:
                    if mixed_branch:
                        m = metrics or {}
                        decode_ce = m.get("decode_ce", float("nan"))
                        denoise_mse = m.get("denoise_mse", float("nan"))
                        late_ce = m.get("late_ce")
                        lex_ce = m.get("lex_ce")
                        attr_loss = m.get("attr")
                        chart_ce = m.get("chart_ce")
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
                        if lex_ce is not None and lex_ce == lex_ce:
                            postfix["lex_ce"] = f"{lex_ce:.3f}"
                        if attr_loss is not None and attr_loss == attr_loss:
                            postfix["attr"] = f"{attr_loss:.3f}"
                        if chart_ce is not None and chart_ce == chart_ce:
                            postfix["chart_ce"] = f"{chart_ce:.3f}"
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
                    append_csv_row(train_csv, TRAIN_CSV_FIELDS, row)

                    interval_done = (
                        (step + 1) % cfg.eval_step == 0 or (step + 1) >= cfg.max_steps
                    )
                    if interval_done:
                        for line in format_interval_summary(step, cfg.max_steps, row):
                            _rank0_log(line, pbar)

                    if (step + 1) % cfg.log_plot_step == 0:
                        update_ppl_plots(
                            train_csv,
                            eval_csv,
                            run_dir,
                            tokens_per_micro_step=cfg.tokens_per_micro_step,
                        )

            rank0_sync = False
            if rank == 0:
                if pbar is not None:
                    pbar.update(1)

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
            # 暂时断开指标 log：中断时不重画 ppl 图。
            if False:
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
        # 暂时断开指标 log：结束时不重画 ppl 图。
        if False:
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

