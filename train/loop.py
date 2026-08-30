"""训练环：``train_loop`` 与步内辅助函数。"""

from __future__ import annotations

import json
import os
import re
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import kind_of
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
from train.ema import ema_absorb_new, ema_merge_loaded, ema_update, init_ema, swap_ema_weights
from train.eval import (
    forward_loss,
    get_amp_dtype,
    uses_dual_branch_logging,
    uses_full_sequence,
)
from train.eval_pipeline import run_online_eval
from train.metrics import (
    TRAIN_OFFICIAL_FIELDS_LM,
    _rank0_log,
    _train_log,
    append_csv_row,
    build_latent_train_row,
    build_train_core_row,
    build_train_official_row,
    format_interval_summary,
    loss_to_ppl,
    train_csv_fields,
)
from train.latent_curriculum import LatentCurriculumSampler
from train.latent_eval import LatentCurriculumEvalContext, curriculum_in_observation_window
from train.async_log import shutdown as shutdown_async_log
from train.run_logs import prepare_run_logs, train_official_csv
from train.muon import build_optimizer, scaled_lr, schedule_optimizer_lrs
from train.scratch import (
    _preload_frozen_encoders,
    _scratch_job_id,
    _scratch_root,
)


def get_lr(
    step: int,
    cfg: FL_TrainConfig,
    *,
    effective_tokens: int | None = None,
) -> float:
    return scaled_lr(
        step, cfg, cfg.learning_rate, effective_tokens=effective_tokens,
    )


def _grad_accum_steps(
    cfg: FL_TrainConfig,
    curriculum_sampler: LatentCurriculumSampler | None,
) -> int:
    if curriculum_sampler is not None:
        return curriculum_sampler.grad_accum_steps
    return cfg.grad_accum_steps


def _curriculum_run_desc(cfg: FL_TrainConfig) -> str:
    """进度条前缀：模型名 + --set 的 readout / 方向 / B / D，便于多 run 串跑区分。"""
    refs = cfg.extra.get("config_refs") or {}
    ov = refs.get("overrides") or {}
    mo = ov.get("model") or {}
    parts = [str(cfg.model)]
    bi = mo.get("bidirectional")
    if bi is True:
        parts.append("bi")
    elif bi is False:
        parts.append("uni")
    readout = mo.get("readout")
    if readout is not None:
        parts.append(str(readout))
    dec_bi = mo.get("decoder_bidirectional")
    if dec_bi is False:
        parts.append("dec-causal")
    elif dec_bi is True:
        parts.append("dec-bi")
    if "latent_dim" in mo:
        parts.append(f"B={mo['latent_dim']}")
    if "block_size" in mo:
        parts.append(f"D={mo['block_size']}")
    return " ".join(parts)


def _fmt_hms(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf") or seconds < 0:
        return "?"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _pbar_kwargs() -> dict[str, Any]:
    """stderr 走 async_log 包装后，tqdm 刷新入队折叠，不再同步打 BeeGFS。"""
    kwargs: dict[str, Any] = {"file": sys.stderr}
    if sys.stderr.isatty():
        kwargs.update(mininterval=0.1, dynamic_ncols=True)
    else:
        kwargs.update(mininterval=2.0, dynamic_ncols=False, ncols=160)
    return kwargs


def _redraw_pbar(pbar: tqdm) -> None:
    """按 mininterval 刷新；``set_postfix(refresh=True)`` 会绕过节流。"""
    pbar.update(0)


def _open_curriculum_stage_pbar(
    sampler: LatentCurriculumSampler,
    *,
    run_desc: str,
) -> tqdm:
    done, budget = sampler.tokens_in_stage()
    stage = sampler.current_stage
    return tqdm(
        total=budget,
        initial=done,
        unit="tok",
        unit_scale=True,
        unit_divisor=1000,
        leave=True,
        smoothing=0.0,
        desc=f"{run_desc} | {stage.name}",
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}]{postfix}"
        ),
        **_pbar_kwargs(),
    )


def _close_stage_pbar(pbar: tqdm | None) -> None:
    if pbar is None:
        return
    if pbar.total:
        pbar.n = pbar.total
        pbar.refresh()
    pbar.close()


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


_STAGE_SNAP_RE = re.compile(r"^(s\d+)-checkpoint_step_(\d+)\.pt$", re.I)
_OLD_SNAP_RE = re.compile(r"^checkpoint_step_(\d+)\.pt$")


def _snapshot_sort_key(path: Path) -> tuple[int, int]:
    """课程快照按 (阶段号, 阶段内 step)；旧 ``checkpoint_step_*`` 视为阶段 0。"""
    m = _STAGE_SNAP_RE.match(path.name)
    if m:
        stage = m.group(1).lower()
        rank = int(stage[1:]) if stage[1:].isdigit() else 0
        return (rank, int(m.group(2)))
    m2 = _OLD_SNAP_RE.match(path.name)
    if m2:
        return (0, int(m2.group(1)))
    return (0, 0)


def _snapshot_ckpt_path(
    run_dir: Path,
    *,
    global_step: int,
    sampler: LatentCurriculumSampler | None,
) -> Path:
    if sampler is None:
        return run_dir / f"checkpoint_step_{global_step:07d}.pt"
    return run_dir / (
        f"{sampler.current_stage.name}-checkpoint_step_"
        f"{sampler.stage_micro_step:07d}.pt"
    )


def _pick_resume_ckpt(run_dir: Path) -> Path | None:
    """Prefer latest, then explicit resume_*, then newest step_* that can open."""
    candidates: list[Path] = [run_dir / "checkpoint_latest.pt"]
    candidates.extend(sorted(run_dir.glob("checkpoint_resume_*.pt"), reverse=True))
    snaps = list(run_dir.glob("checkpoint_step_*.pt"))
    snaps.extend(run_dir.glob("s*-checkpoint_step_*.pt"))
    snaps = sorted(snaps, key=_snapshot_sort_key, reverse=True)
    candidates.extend(snaps)
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


def _grad_nonfinite_flag(model: nn.Module, device: torch.device) -> torch.Tensor:
    """各参数 ``~isfinite.any()`` 累加到 GPU 标量；热路径不逐参数 ``.item()``。"""
    bad = torch.zeros((), device=device, dtype=torch.int32)
    for param in model.parameters():
        grad = param.grad
        if grad is not None:
            bad = bad + (~torch.isfinite(grad)).any().to(dtype=torch.int32)
    return bad


def _params_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return False
    return True


def _sync_after_rank0_work(
    *,
    is_distributed: bool,
    rank0_work: bool,
) -> None:
    """仅当各 rank 都知道 rank0 刚做了非集合写盘（ckpt）时才 barrier。

    ``rank0_work`` 必须在所有 rank 上相同（按 step 判定），不要每步 all_reduce 探测。
    """
    if not is_distributed or not rank0_work:
        return
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


def _latent_tokens_seen(cfg: FL_TrainConfig, run_tokens: int) -> int:
    """解冻用的累计 token：本 run 已见 + Stage2 绑定的 Stage1 已见。

    Stage2 续训的 ``run_tokens`` 只有本阶段（扩展 0–5B）。``_thawed`` 不进
    checkpoint，进程重建后 mid 从 False 起；若不加上 ``stage1_tokens_seen``，
    会按 5B 门槛判成未解冻。无前置作业该键为空，行为与只传本 run 相同。
    """
    prior = int(cfg.extra.get("stage1_tokens_seen") or 0)
    return int(run_tokens) + prior


def _notify_tokens_seen(model: nn.Module, n: int, optimizer: Any) -> bool:
    """向前兼容：模型若实现 ``on_tokens_seen`` 则通知累计 token，否则 no-op。

    返回是否刚解冻（``mid`` 首次越过门槛）。
    """
    raw = unwrap_model(model)
    fn = getattr(raw, "on_tokens_seen", None)
    if not callable(fn):
        fn = getattr(getattr(raw, "backbone", None), "on_tokens_seen", None)
    if callable(fn):
        return bool(fn(int(n), optimizer))
    return False


def _rewrap_ddp_if_thawed(
    model: nn.Module,
    *,
    thawed: bool,
    is_distributed: bool,
    device: torch.device,
    rank: int,
) -> nn.Module:
    """``mid`` 解冻后重建 DDP，让新 ``requires_grad`` 的入口参数进入 reducer。

    DDP 只在构造时登记当时可训参数；解冻前入口全冻，不重建则多卡梯度不会
    allreduce。只剥 DDP、保留 ``torch.compile``。
    """
    if not thawed or not is_distributed:
        return model
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    inner = model.module if isinstance(model, DDP) else model
    wrapped = DDP(
        inner,
        device_ids=[device.index],
        output_device=device.index,
    )
    if rank == 0:
        _train_log("latent mid 已解冻，已重建 DDP 以同步入口梯度")
    return wrapped


def _latent_bundle(model: nn.Module) -> Any:
    """BELF ``bundle`` / RELF ``latent``；其它模型为 ``None``。"""
    raw = unwrap_model(model)
    bb = getattr(raw, "backbone", raw)
    for name in ("bundle", "latent"):
        obj = getattr(bb, name, None)
        if obj is not None and hasattr(obj, "snapshot_ref_from_live"):
            return obj
    return None


def _refresh_s2_ref(
    model: nn.Module,
    cfg: FL_TrainConfig,
    *,
    from_live: bool,
) -> None:
    """Stage2 的 ``q_ref`` 冻成 Stage1 checkpoint EMA，与 artifacts 目录无关。

    首启：EMA 已写入 live，从 live 再冻。
    续训：live 已是本 run 中段，从 ``extra.init_ckpt`` 的 EMA 重冻。
    """
    if not bool(cfg.extra.get("init_from_ema")):
        return
    bundle = _latent_bundle(model)
    if bundle is None:
        return
    if from_live:
        bundle.snapshot_ref_from_live()
        return
    spec = str(cfg.extra.get("init_ckpt") or "").strip()
    if not spec:
        return
    repo = Path(__file__).resolve().parents[1]
    path = Path(spec)
    if not path.is_absolute():
        path = (repo / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage2 重冻 q_ref：找不到 Stage1 ckpt {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ema = ck.get("ema")
    if not isinstance(ema, dict) or not ema:
        raise ValueError(f"{path}: Stage2 重冻 q_ref 需要 Stage1 EMA")
    bundle.snapshot_ref_from_named_tensors(ema, unwrap_model(model))


def train_loop(
    model: nn.Module,
    cfg: FL_TrainConfig,
    model_meta: dict[str, Any],
    train_ds: TokenChunkDataset | None,
    eval_loader: DataLoader | None,
    gpt2_model: nn.Module | None,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    is_distributed: bool,
    curriculum_sampler: LatentCurriculumSampler | None = None,
    curriculum_eval_ctx: LatentCurriculumEvalContext | None = None,
    latent_probe_pool: Dataset | None = None,
    latent_pad_token_id: int | None = None,
) -> None:
    amp_dtype = get_amp_dtype(cfg.dtype)
    from train.hardware import (
        HardwareMismatchError,
        detect_train_hardware,
        ensure_hardware_lock,
    )
    from train.run_path import checkpoint_run_dir_from_cfg

    run_dir = checkpoint_run_dir_from_cfg(cfg)
    is_latent = kind_of(cfg.model) == "latent"
    train_fields = train_csv_fields(cfg.model, cfg)

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
    if rank == 0:
        prepare_run_logs(run_dir, model=cfg.model, start_step=None, cfg=cfg)
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

    def _curriculum_ckpt() -> dict[str, Any] | None:
        if curriculum_sampler is None:
            return None
        return curriculum_sampler.curriculum_state()

    curriculum_stage_idx = (
        curriculum_sampler._stage_idx if curriculum_sampler is not None else -1
    )

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
        step, loaded_ema, loaded_curriculum = load_checkpoint(
            resume_path, model, optimizer, device,
            cfg=cfg, model_meta=model_meta, restore_rng=(rank == 0),
        )
        if curriculum_sampler is not None and loaded_curriculum:
            curriculum_sampler.effective_tokens_global = int(
                loaded_curriculum.get("effective_tokens_global", 0)
            )
            curriculum_sampler._stage_idx = int(loaded_curriculum.get("stage_idx", 0))
            has_stage_micro = "stage_micro_step" in loaded_curriculum
            if has_stage_micro:
                curriculum_sampler.stage_micro_step = int(
                    loaded_curriculum["stage_micro_step"]
                )
            curriculum_sampler.sync_stage()
            curriculum_stage_idx = curriculum_sampler._stage_idx
        else:
            has_stage_micro = False
        _release_staged_ckpt(
            resume_path, rank=rank, is_distributed=is_distributed,
        )
        if ema_state is not None:
            if loaded_ema:
                ema_merge_loaded(ema_state, loaded_ema)
            elif rank == 0:
                _train_log(
                    "Checkpoint has no EMA state; re-initialized EMA from model weights",
                )
                ema_state = init_ema(model)
        if curriculum_sampler is not None:
            seen_n = int(curriculum_sampler.effective_tokens_global)
        elif step == 0:
            seen_n = 0
        else:
            seen_n = cfg.tokens_seen_after_step(step - 1)
        thawed = _notify_tokens_seen(
            model, _latent_tokens_seen(cfg, seen_n), optimizer,
        )
        model = _rewrap_ddp_if_thawed(
            model,
            thawed=thawed,
            is_distributed=is_distributed,
            device=device,
            rank=rank,
        )
        _refresh_s2_ref(model, cfg, from_live=False)
        if ema_state is not None:
            ema_absorb_new(ema_state, model)
        if rank == 0:
            curr_resume = None
            if curriculum_sampler is not None and has_stage_micro:
                curr_resume = (
                    curriculum_sampler.current_stage.name,
                    curriculum_sampler.stage_micro_step,
                )
            kept = prepare_run_logs(
                run_dir,
                model=cfg.model,
                start_step=step,
                cfg=cfg,
                curriculum_resume=curr_resume,
            )
            _train_log(
                f"Resuming from checkpoint: step {step} "
                f"(train_log {kept.get('train_log', 0)} rows, "
                f"eval_log {kept.get('eval_log', 0)} rows, "
                f"samples_dirs_removed {kept.get('eval_samples_dirs', 0)})",
            )
        if step >= cfg.max_steps:
            if rank == 0:
                _train_log(
                    f"Reached max_steps={cfg.max_steps}; training is already complete"
                )
                from train.stage_chain import write_complete_marker

                write_complete_marker(
                    run_dir,
                    step=step,
                    cfg=cfg,
                    curriculum_state=_curriculum_ckpt(),
                )
            return
    else:
        init_spec = cfg.extra.get("init_ckpt")
        if init_spec:
            repo = Path(__file__).resolve().parents[1]
            init_path = Path(str(init_spec))
            if not init_path.is_absolute():
                init_path = (repo / init_path).resolve()
            init_path = _stage_ckpt_to_local(
                init_path, rank=rank, is_distributed=is_distributed,
            )
            from_ema = bool(cfg.extra.get("init_from_ema"))
            loaded_ema, init_info = load_init_weights(
                init_path, model, device, from_ema=from_ema,
            )
            _release_staged_ckpt(
                init_path, rank=rank, is_distributed=is_distributed,
            )
            if from_ema:
                prior = int(cfg.extra.get("stage1_tokens_seen") or 0)
                thawed = False
                if prior > 0:
                    thawed = _notify_tokens_seen(
                        model, _latent_tokens_seen(cfg, 0), optimizer,
                    )
                model = _rewrap_ddp_if_thawed(
                    model,
                    thawed=thawed,
                    is_distributed=is_distributed,
                    device=device,
                    rank=rank,
                )
                _refresh_s2_ref(model, cfg, from_live=True)
                if ema_state is not None:
                    ema_state = init_ema(model)
                if rank == 0:
                    _train_log(
                        f"init-ckpt 从 EMA 写入 live 权重 "
                        f"(n_ema_applied={init_info.get('n_ema_applied', 0)})；"
                        f"stage1_tokens_seen={prior} 已通知解冻并重建 EMA；"
                        f"q_ref 已按 Stage1 EMA 重冻"
                    )
            elif ema_state is not None:
                if loaded_ema:
                    n_ema = ema_merge_loaded(ema_state, loaded_ema)
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
                    "(official ELF train_step); train_ppl uses decode CE",
                )
            else:
                _train_log(
                    f"{cfg.model.upper()} dual-branch: denoise:decode ≈ "
                    f"{denoise_prob:g}:{decoder_prob:g} loss mix; "
                    "train_ppl uses decode CE",
                )

    model.train()
    t0 = time.time()
    step_backward_done = False
    pbar: tqdm | None = None
    pbar_stage_idx = -1
    pbar_rate_t0: float | None = None
    pbar_rate_n0 = 0
    run_desc = _curriculum_run_desc(cfg)
    if rank == 0:
        if curriculum_sampler is not None:
            pbar = _open_curriculum_stage_pbar(
                curriculum_sampler, run_desc=run_desc,
            )
            pbar_stage_idx = curriculum_sampler._stage_idx
            done0, _ = curriculum_sampler.tokens_in_stage()
            pbar_rate_t0 = None
            pbar_rate_n0 = done0
        else:
            pbar = tqdm(
                total=cfg.max_steps,
                initial=step,
                unit="step",
                leave=True,
                **_pbar_kwargs(),
            )

    prefetch: tuple[torch.Tensor, int] | None = None

    def _next_cpu_batch(fetch_step: int) -> tuple[torch.Tensor, int]:
        if curriculum_sampler is not None:
            return curriculum_sampler.fetch_batch(fetch_step, rank)
        assert train_ds is not None
        return fetch_train_batch(
            train_ds, fetch_step, cfg.batch_size, world_size, rank, cfg.seed,
        ), 0

    try:
        while step < cfg.max_steps:
            if curriculum_sampler is not None and curriculum_sampler.is_complete():
                if rank == 0:
                    _train_log(
                        "Reached curriculum effective token budget; stopping training",
                    )
                break

            log_stage_idx = -1
            log_stage_name = ""
            log_csv_step = step
            if prefetch is not None:
                batch, eff_rank = prefetch
                prefetch = None
            else:
                batch, eff_rank = _next_cpu_batch(step)
            if curriculum_sampler is not None:
                log_stage_idx = curriculum_sampler._stage_idx
                log_stage_name = curriculum_sampler.current_stage.name
                log_csv_step = curriculum_sampler.stage_micro_step
            batch = batch.to(device, non_blocking=True)

            if curriculum_sampler is not None:
                eff_t = torch.tensor([eff_rank], device=device, dtype=torch.int64)
                if is_distributed:
                    dist.all_reduce(eff_t, op=dist.ReduceOp.SUM)
                curriculum_sampler.add_effective_tokens(int(eff_t.item()))
                if curriculum_sampler._stage_idx != curriculum_stage_idx:
                    optimizer.zero_grad(set_to_none=True)
                    if rank == 0:
                        _train_log(
                            f"Curriculum stage -> {curriculum_sampler.current_stage.name} "
                            f"peak_L={curriculum_sampler.graph_l} "
                            f"micro={curriculum_sampler.current_batch_size} "
                            f"accum={curriculum_sampler.grad_accum_steps}; "
                            "reset grad accum",
                        )
                    curriculum_stage_idx = curriculum_sampler._stage_idx

            accum = _grad_accum_steps(cfg, curriculum_sampler)
            eff_tokens_lr = (
                curriculum_sampler.effective_tokens_global
                if curriculum_sampler is not None
                else None
            )
            lr = get_lr(step, cfg, effective_tokens=eff_tokens_lr)
            if cfg.use_muon:
                schedule_optimizer_lrs(
                    optimizer,
                    adam_lr=lr,
                    muon_lr=scaled_lr(
                        step,
                        cfg,
                        cfg.muon_learning_rate,
                        effective_tokens=eff_tokens_lr,
                    ),
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
                    if is_distributed and (step + 1) % accum != 0
                    else nullcontext()
                )
                with sync_ctx:
                    (micro_loss / accum).backward()
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

            if (step + 1) % accum == 0:
                bad_grads = _grad_nonfinite_flag(model, device)
                if is_distributed:
                    dist.all_reduce(bad_grads, op=dist.ReduceOp.MAX)
                grads_ok = int(bad_grads.item()) == 0
                if grads_ok:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    if ema_state is not None:
                        ema_update(ema_state, model, ema_decay)
                    unwrap_model(model).on_optimizer_step(
                        model=model,
                        ema_state=ema_state,
                        opt_step=(step + 1) // accum,
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
                if not grads_ok and not _params_are_finite(raw):
                    raise RuntimeError(
                        f"Non-finite model weights at step {step}; "
                        "resume from an earlier checkpoint",
                    )
                optimizer.zero_grad(set_to_none=True)

            # 训练指标主表 + 官方卫星；在线 eval（HeldOut 永开 + 共享样本组件）。
            train_loss = float(micro_loss.item()) if loss_ok else float("nan")
            raw_for_log = unwrap_model(model)
            if mixed_branch:
                loss_branch = "mixed"
                metrics = raw_for_log.train_metrics()
            elif dual_branch:
                loss_branch = train_branch if train_branch else ""
                metrics = raw_for_log.train_metrics()
            else:
                loss_branch = ""
                metrics = raw_for_log.train_metrics()
            elapsed = time.time() - t0
            # 每微批消耗一整份数据 batch；tok/s 为全卡合计。
            seq_tokens = batch.size(0) * (
                batch.size(1) if uses_full_sequence(model) else batch.size(1) - 1
            )
            tokens_per_sec = (seq_tokens * max(1, world_size)) / max(elapsed, 1e-6)

            log_tokens = (
                curriculum_sampler.effective_tokens_global
                if curriculum_sampler is not None
                else cfg.tokens_seen_after_step(step)
            )
            thawed = _notify_tokens_seen(
                model, _latent_tokens_seen(cfg, log_tokens), optimizer,
            )
            model = _rewrap_ddp_if_thawed(
                model,
                thawed=thawed,
                is_distributed=is_distributed,
                device=device,
                rank=rank,
            )
            if ema_state is not None:
                ema_absorb_new(ema_state, model)
            if is_latent:
                train_row = build_latent_train_row(
                    log_csv_step,
                    log_tokens,
                    train_loss,
                    lr,
                    tokens_per_sec,
                    metrics=metrics,
                    curriculum_stage=(
                        log_stage_name if curriculum_sampler is not None else ""
                    ),
                    observation_window=(
                        curriculum_in_observation_window(curriculum_sampler)
                        if curriculum_sampler is not None
                        else None
                    ),
                )
                core_row = train_row
                official_row = None
            else:
                core_row = build_train_core_row(
                    step,
                    cfg.tokens_seen_after_step(step),
                    train_loss,
                    lr,
                    tokens_per_sec,
                    dual_branch=dual_branch,
                    loss_branch=loss_branch,
                    metrics=metrics,
                )
                official_row = build_train_official_row(
                    step,
                    dual_branch=dual_branch,
                    loss_branch=loss_branch,
                    train_loss=train_loss,
                    metrics=metrics,
                )
                train_row = core_row

            do_eval = (step + 1) % cfg.eval_step == 0 and (
                eval_loader is not None or curriculum_eval_ctx is not None
            )
            if do_eval:
                run_online_eval(
                    model,
                    cfg=cfg,
                    run_dir=run_dir,
                    step=step,
                    lr=lr,
                    eval_loader=eval_loader,
                    gpt2_model=gpt2_model,
                    device=device,
                    amp_dtype=amp_dtype,
                    rank=rank,
                    world_size=world_size,
                    is_distributed=is_distributed,
                    pbar_parent=pbar,
                    ema_state=ema_state,
                    swap_ema_weights=swap_ema_weights,
                    curriculum_eval_ctx=curriculum_eval_ctx,
                    curriculum_sampler=curriculum_sampler,
                    eval_tokens=log_tokens if is_latent else None,
                    latent_probe_pool=latent_probe_pool,
                    latent_pad_token_id=latent_pad_token_id,
                    log_step=log_csv_step if curriculum_sampler is not None else None,
                    log_stage=log_stage_name or None,
                )

            if rank == 0:
                m = metrics or {}
                if is_latent:
                    postfix = {
                        "loss": f"{train_loss:.3f}",
                        "lr": f"{lr:.2e}",
                        "tok_s": f"{tokens_per_sec:.0f}",
                    }
                    recon_ce = m.get("recon_ce", float("nan"))
                    token_acc = m.get("token_acc", float("nan"))
                    if recon_ce == recon_ce:
                        postfix["recon"] = f"{recon_ce:.3f}"
                    if token_acc == token_acc:
                        postfix["acc"] = f"{token_acc:.3f}"
                elif mixed_branch:
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
                    decode_ce = m.get("decode_ce", float("nan"))
                    if decode_ce == decode_ce:
                        postfix["ppl"] = f"{loss_to_ppl(decode_ce):.1f}"
                if (
                    curriculum_sampler is not None
                    and pbar is not None
                ):
                    if curriculum_sampler._stage_idx != pbar_stage_idx:
                        _close_stage_pbar(pbar)
                        pbar = _open_curriculum_stage_pbar(
                            curriculum_sampler, run_desc=run_desc,
                        )
                        pbar_stage_idx = curriculum_sampler._stage_idx
                        pbar_rate_t0 = None
                        pbar_rate_n0, _ = curriculum_sampler.tokens_in_stage()
                    done, budget = curriculum_sampler.tokens_in_stage()
                    now = time.perf_counter()
                    if pbar_rate_t0 is None:
                        # 本阶段第一微批（常含 compile）只推进进度；速度从下一拍起算
                        pbar_rate_t0 = now
                        pbar_rate_n0 = done
                        postfix["tok/s"] = "?"
                        postfix["eta"] = "?"
                    else:
                        gained = max(0, done - pbar_rate_n0)
                        elapsed_sn = max(now - pbar_rate_t0, 1e-9)
                        tok_s_sn = gained / elapsed_sn
                        remain = max(0, budget - done)
                        postfix["tok/s"] = f"{tok_s_sn:.0f}"
                        postfix["eta"] = _fmt_hms(
                            remain / tok_s_sn if tok_s_sn > 0 else float("inf")
                        )
                    postfix.pop("tok_s", None)
                    pbar.n = done
                    pbar.set_postfix(**postfix, refresh=False)
                    _redraw_pbar(pbar)
                elif pbar is not None:
                    pbar.set_postfix(**postfix, refresh=False)
                    _redraw_pbar(pbar)
                append_csv_row(train_csv, train_fields, train_row)
                if official_row is not None:
                    append_csv_row(
                        train_official_csv(run_dir),
                        TRAIN_OFFICIAL_FIELDS_LM,
                        official_row,
                    )

                interval_done = (
                    (step + 1) % cfg.eval_step == 0 or (step + 1) >= cfg.max_steps
                )
                if interval_done:
                    for line in format_interval_summary(
                        step, cfg.max_steps, core_row, official_row,
                    ):
                        _rank0_log(line, pbar)

            if curriculum_sampler is not None:
                curriculum_sampler.stage_micro_step += 1
                if curriculum_sampler._stage_idx != log_stage_idx:
                    curriculum_sampler.stage_micro_step = 0

            next_step = step + 1
            do_save = next_step % cfg.save_step == 0
            do_snapshot = next_step % cfg.snapshot_step == 0
            if rank == 0:
                if pbar is not None and curriculum_sampler is None:
                    pbar.update(1)

                # save_step / snapshot_step are independent intervals; do not nest
                # snapshot under save (snapshot_step need not divide save_step).
                if do_save or do_snapshot:
                    # Always refresh latest when writing any durable checkpoint.
                    save_checkpoint(
                        latest_ckpt, model, optimizer, next_step, cfg, model_meta,
                        ema_state=ema_state,
                        curriculum_state=_curriculum_ckpt(),
                    )
                    if do_snapshot:
                        save_checkpoint(
                            _snapshot_ckpt_path(
                                run_dir,
                                global_step=next_step,
                                sampler=curriculum_sampler,
                            ),
                            model, optimizer, next_step, cfg, model_meta,
                            ema_state=ema_state,
                            curriculum_state=_curriculum_ckpt(),
                        )
                    _rank0_log(f"  [ckpt] saved at step {next_step}", pbar)

            _sync_after_rank0_work(
                is_distributed=is_distributed,
                rank0_work=do_save or do_snapshot,
            )

            # 存盘后再预取，避免 curriculum_state 的 bucket 被写成下一步。
            if (
                step + 1 < cfg.max_steps
                and not (
                    curriculum_sampler is not None
                    and curriculum_sampler.is_complete()
                )
            ):
                prefetch = _next_cpu_batch(step + 1)

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
                curriculum_state=_curriculum_ckpt(),
            )
            _train_log(f"Saved; resume from step {next_step} on next run")
            shutdown_async_log()
        if is_distributed:
            dist.barrier()
        return

    # Always persist the finished run. Periodic saves only fire when
    # max_steps is a multiple of save_step/snapshot_step; the final write
    # covers the common case where it is not.
    if rank == 0:
        if pbar is not None:
            if curriculum_sampler is not None and curriculum_sampler.is_complete():
                _close_stage_pbar(pbar)
            else:
                pbar.close()
        save_checkpoint(
            latest_ckpt, model, optimizer, step, cfg, model_meta,
            ema_state=ema_state,
            curriculum_state=_curriculum_ckpt(),
        )
        final_snapshot = _snapshot_ckpt_path(
            run_dir, global_step=step, sampler=curriculum_sampler,
        )
        save_checkpoint(
            final_snapshot, model, optimizer, step, cfg, model_meta,
            ema_state=ema_state,
            curriculum_state=_curriculum_ckpt(),
        )
        from train.stage_chain import write_complete_marker

        write_complete_marker(
            run_dir,
            step=step,
            cfg=cfg,
            curriculum_state=_curriculum_ckpt(),
        )
        _train_log(
            f"Training finished after {step} steps; "
            f"saved {latest_ckpt.name} and {final_snapshot.name} in {run_dir}"
        )
        shutdown_async_log()
    if is_distributed:
        # Keep peers alive until rank 0 finishes the (often multi-GB) write;
        # otherwise destroy_process_group can race with the final save.
        dist.barrier()

