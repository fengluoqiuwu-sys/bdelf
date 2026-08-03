#!/usr/bin/env python3
"""将旧版扁平 checkpoint 目录迁移到 ``{fast|full}/{model}/{hash}/``。

- 补齐新版 train 字段（generate / generate_sampling / global_batch_size 等）
- 按旧超参构造 ``--set``，用与 ``resolve_checkpoint.py`` 相同规则算哈希
- 写入 ``hardware.json``（默认 4×RTX 4090 24GiB，不进哈希）
- 更新 ``config.json`` 与全部 ``checkpoint*.pt`` 内 ``train_config``
- 移动目录；写 ``launch_hint.json`` 便于续训

用法::

    .venv/bin/python migrate_checkpoints.py --root cache/checkpoints [--dry-run]
    .venv/bin/python migrate_checkpoints.py --root cache/checkpoints --only ar-100m-full-muon
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from train import get_train_config, parse_train_overrides
from train.generate_config import get_generate
from train.hardware import TrainHardware, save_hardware
from train.run_path import (
    build_train_fingerprint,
    checkpoint_run_dir_from_cfg,
    config_hash_from_fingerprint,
    run_relpath,
)

# 用户确认：这些旧 run 均在 4×4090 上训练
DEFAULT_HARDWARE = TrainHardware(
    gpu_name="NVIDIA GeForce RTX 4090",
    gpu_count=4,
    memory_gb_per_gpu=24,
)

# 旧扁平目录名 → 迁移时使用的 preprocess（其余从 config 读）
_KNOWN_OLD_DIRS = {
    "ar-100m-full-muon",
    "ar2-100m-full-muon",
    "ar2-300m-full-muon",
    "elf-cfg-100m-full-muon",
}


def _log(msg: str) -> None:
    print(f"[migrate] {msg}", flush=True)


def _overrides_from_old(
    old: dict[str, Any],
    *,
    fresh: Any,
) -> list[str]:
    """相对「当前 YAML + generate=eval」需要保留的旧超参 → ``--set`` 列表。"""
    sets: list[str] = []
    extra = old.get("extra") or {}
    old_gbs = old.get("global_batch_size")
    if old_gbs is None:
        old_gbs = extra.get("global_batch_size")
    if old_gbs is None:
        bs = int(old["batch_size"])
        accum = int(old["grad_accum_steps"])
        ws = int(old.get("world_size") or 1)
        old_gbs = bs * accum * ws

    if int(old["batch_size"]) != int(fresh.batch_size):
        sets.append(f"batch.batch_size={int(old['batch_size'])}")
    if int(old_gbs) != int(fresh.global_batch_size):
        sets.append(f"batch.global_batch_size={int(old_gbs)}")

    for key in (
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "muon_learning_rate",
        "muon_momentum",
        "muon_ns_steps",
    ):
        if key in old and old[key] is not None and getattr(fresh, key) != old[key]:
            sets.append(f"optimizer.{key}={old[key]}")

    if old.get("muon_weight_decay") is not None:
        if float(old["muon_weight_decay"]) != float(fresh.muon_weight_decay):
            sets.append(f"optimizer.muon_weight_decay={old['muon_weight_decay']}")

    old_ema = extra.get("ema_decay")
    fresh_ema = fresh.extra.get("ema_decay", 0.0)
    if old_ema is not None and float(old_ema) != float(fresh_ema or 0.0):
        sets.append(f"optimizer.ema_decay={old_ema}")

    # gen_eval_batches → gen_eval_samples（旧语义保留数值）
    old_samples = old.get("gen_eval_samples")
    if old_samples is None and old.get("gen_eval_batches") is not None:
        old_samples = old["gen_eval_batches"]
    if old_samples is not None and int(old_samples) != int(fresh.gen_eval_samples):
        sets.append(f"eval.gen_eval_samples={int(old_samples)}")

    if old.get("eval_sample_count") is not None:
        if old["eval_sample_count"] != fresh.eval_sample_count:
            sets.append(f"eval.eval_sample_count={old['eval_sample_count']}")

    return sets


def _migrate_train_dict(
    old: dict[str, Any],
    *,
    composed: Any,
) -> dict[str, Any]:
    """用 compose 结果补齐新字段，但保留旧日程微步计数（续训进度 / LR）。"""
    new = asdict(composed)
    # 保留旧 run 的日程进度相关字段，避免 max_steps 变化导致提前结束
    for key in (
        "max_steps",
        "warmup_steps",
        "eval_step",
        "save_step",
        "snapshot_step",
        "log_plot_step",
        "world_size",
        "grad_accum_steps",
        "batch_size",
    ):
        if key in old and old[key] is not None:
            new[key] = old[key]

    old_extra = dict(old.get("extra") or {})
    new_extra = dict(new.get("extra") or {})
    for key in (
        "target_tokens",
        "max_optimizer_steps",
        "tokens_per_optimizer_step",
        "effective_tokens_per_optimizer_step",
        "chunk_length",
        "decoder_prob",
        "ema_decay",
        "compile",
    ):
        if key in old_extra and old_extra[key] is not None:
            new_extra[key] = old_extra[key]

    # 清理已废弃 / 硬件相关旧字段
    for key in (
        "hardware_profile",
        "estimated_peak_vram_gb",
        "memory_budget_gb",
        "global_batch_size",  # 已升到顶层
    ):
        new_extra.pop(key, None)

    new_extra["config_hash"] = composed.name
    new_extra["run_relpath"] = composed.extra.get("run_relpath")
    new_extra["migrated_from"] = old.get("name")
    refs = dict(new_extra.get("config_refs") or {})
    refs.pop("hardware", None)
    refs.pop("optimizer", None)
    refs.pop("batch", None)
    new_extra["config_refs"] = refs

    new["extra"] = new_extra
    new["name"] = composed.name
    new["generate"] = composed.generate
    new["generate_sampling"] = composed.generate_sampling
    new["global_batch_size"] = composed.global_batch_size

    # 去掉旧顶层废弃键
    new.pop("eval_use_fast_infer", None)
    new.pop("eval_gen_steps", None)
    new.pop("gen_eval_batches", None)
    return new


def _wait_openable(path: Path, *, timeout_s: float = 180.0) -> Path:
    """BeeGFS: ``is_file`` 可能通过但 ``open`` 失败；轮询直到可打开。"""
    path = path.resolve()
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(path, os.O_RDONLY)
            os.close(fd)
            return path
        except (FileNotFoundError, OSError):
            pass
        if time.time() >= deadline:
            raise FileNotFoundError(f"Timed out waiting for openable file: {path}")
        time.sleep(0.5)


def _stage_local(src: Path) -> Path:
    """拷到节点本地盘再 load，避开 BeeGFS 幽灵 dentry。"""
    src = _wait_openable(src)
    job = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    local_dir = Path(os.environ.get("SLURM_TMPDIR") or "/tmp") / f"bdelf-migrate-{job}"
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / src.name
    partial = local.with_suffix(local.suffix + ".partial")
    shutil.copyfile(src, partial)
    partial.replace(local)
    return _wait_openable(local)


def _update_checkpoint_pt(path: Path, train_config: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        _log(f"  would update {path.name}")
        return
    local = _stage_local(path)
    try:
        ck = torch.load(local, map_location="cpu", weights_only=False)
        ck["train_config"] = train_config
        out_local = local.with_suffix(local.suffix + ".migrate_out")
        torch.save(ck, out_local)
        # 先写 BeeGFS 临时名再 replace，减少半截文件
        bee_tmp = path.with_suffix(path.suffix + ".migrate_tmp")
        shutil.copyfile(out_local, bee_tmp)
        bee_tmp.replace(path)
        out_local.unlink(missing_ok=True)
    finally:
        local.unlink(missing_ok=True)


def _migrate_300m(old_dir: Path, *, dry_run: bool, hardware: TrainHardware) -> Path:
    """300m 已不在 train CLI 内；按补齐后的 dict 自算指纹并搬迁。"""
    from train.train import (
        FL_BatchConfig,
        FL_EvalConfig,
        FL_OptimizerConfig,
        FL_ScheduleConfig,
    )

    cfg_path = old_dir / "config.json"
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    old = dict(payload["train"])
    extra = dict(old.get("extra") or {})
    model = old["model"]
    variant = old["variant"]
    model_config = old["model_config"]
    dataset = old["dataset"]
    preprocess = old["preprocess"]
    generate = "eval"
    gen_cfg = get_generate(model, generate)

    gbs = old.get("global_batch_size") or extra.get("global_batch_size")
    if gbs is None:
        gbs = int(old["batch_size"]) * int(old["grad_accum_steps"]) * int(old["world_size"])

    opt = FL_OptimizerConfig(
        name=model_config,
        dtype=old.get("dtype", "bf16"),
        learning_rate=float(old["learning_rate"]),
        weight_decay=float(old["weight_decay"]),
        beta1=float(old["beta1"]),
        beta2=float(old["beta2"]),
        grad_clip=float(old["grad_clip"]),
        muon_learning_rate=float(old.get("muon_learning_rate", 0.003)),
        muon_weight_decay=float(old.get("muon_weight_decay", 0.01)),
        muon_momentum=float(old.get("muon_momentum", 0.95)),
        muon_ns_steps=int(old.get("muon_ns_steps", 5)),
        extra={"ema_decay": float(extra.get("ema_decay") or 0.0)},
    )
    batch = FL_BatchConfig(
        name=f"{model_config}-{variant}",
        batch_size=int(old["batch_size"]),
        global_batch_size=int(gbs),
    )
    # schedule：用旧 extra 里能还原的字段 + 当前 full 文件缺省
    sched_path = Path("config/train/schedule") / f"{variant}.yaml"
    import yaml

    with open(sched_path, encoding="utf-8") as f:
        sched_raw = yaml.safe_load(f) or {}
    schedule = FL_ScheduleConfig(
        name=variant,
        variant=variant,
        target_tokens=int(extra.get("target_tokens") or sched_raw.get("target_tokens")),
        warmup_ratio=float(sched_raw.get("warmup_ratio", 0.005)),
        min_lr_ratio=float(old.get("min_lr_ratio", 0.1)),
        log_plot_step=int(sched_raw.get("log_plot_step", 100)),
        eval_step=int(sched_raw.get("eval_step", 500)),
        save_step=int(sched_raw.get("save_step", 1000)),
        snapshot_step=int(sched_raw.get("snapshot_step", 5000)),
        resume=bool(old.get("resume", True)),
        seed=int(old.get("seed", 42)),
        use_muon=bool(old.get("use_muon", True)),
        extra={"compile": bool(extra.get("compile", sched_raw.get("compile", False)))},
    )
    samples = old.get("gen_eval_samples")
    if samples is None:
        samples = old.get("gen_eval_batches", 256)
    eval_cfg = FL_EvalConfig(
        name="default",
        eval_sample_count=old.get("eval_sample_count"),
        eval_sample_seed=int(old.get("eval_sample_seed", 42)),
        gen_eval_model=str(old.get("gen_eval_model", "gpt2-large")),
        gen_eval_model_dtype=old.get("gen_eval_model_dtype", "bf16"),
        gen_eval_model_device=str(old.get("gen_eval_model_device", "cuda")),
        gen_eval_samples=int(samples),
    )
    fp = build_train_fingerprint(
        model=model,
        model_config=model_config,
        variant=variant,
        dataset=dataset,
        preprocess=preprocess,
        generate=generate,
        optimizer=opt,
        batch=batch,
        schedule=schedule,
        eval_cfg=eval_cfg,
        generate_cfg=gen_cfg,
        overrides={},
    )
    config_hash = config_hash_from_fingerprint(fp)
    rel = run_relpath(variant=variant, model=model, config_hash=config_hash)
    dest = Path("cache/checkpoints") / rel

    new_train = dict(old)
    new_train["name"] = config_hash
    new_train["generate"] = generate
    new_train["generate_sampling"] = gen_cfg.to_sampling_cfg()
    new_train["global_batch_size"] = int(gbs)
    if new_train.get("muon_weight_decay") is None:
        new_train["muon_weight_decay"] = 0.01
    new_train["gen_eval_samples"] = int(samples)
    new_train.pop("eval_use_fast_infer", None)
    new_train.pop("eval_gen_steps", None)
    new_train.pop("gen_eval_batches", None)
    new_extra = dict(extra)
    for key in (
        "hardware_profile",
        "estimated_peak_vram_gb",
        "memory_budget_gb",
        "global_batch_size",
    ):
        new_extra.pop(key, None)
    new_extra["config_hash"] = config_hash
    new_extra["run_relpath"] = rel
    new_extra["migrated_from"] = old.get("name")
    new_extra["note"] = (
        "model_config=300m；当前 train CLI 仅支持 100m，"
        "续训需恢复 300m recipe 或单独加载权重"
    )
    refs = dict(new_extra.get("config_refs") or {})
    refs.pop("hardware", None)
    new_extra["config_refs"] = refs
    new_train["extra"] = new_extra

    _log(f"{old_dir.name} → {rel} (300m special)")
    if dry_run:
        return dest

    if dest.exists():
        raise FileExistsError(f"destination already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    (old_dir / "config.json").write_text(
        json.dumps({"train": new_train, "model": payload.get("model")}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    save_hardware(old_dir, hardware)
    hint = {
        "old_dir": old_dir.name,
        "run_relpath": rel,
        "config_hash": config_hash,
        "warning": "300m not supported by current train.py CLI",
        "hardware": hardware.to_dict(),
    }
    (old_dir / "launch_hint.json").write_text(
        json.dumps(hint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    for pt in sorted(old_dir.glob("checkpoint*.pt")):
        _log(f"  updating {pt.name}")
        _update_checkpoint_pt(pt, new_train, dry_run=False)

    old_dir.rename(dest)
    _log(f"  moved → {dest}")
    return dest


def migrate_one(
    old_dir: Path,
    *,
    dry_run: bool,
    hardware: TrainHardware,
    root: Path,
) -> Path:
    cfg_path = old_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing {cfg_path}")

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    old = payload["train"]
    model = old["model"]
    model_config = old["model_config"]
    variant = old["variant"]
    dataset = old["dataset"]
    preprocess = old["preprocess"]

    if model_config != "100m":
        return _migrate_300m(old_dir, dry_run=dry_run, hardware=hardware)

    train_config_name = f"{model_config}-{variant}"
    generate = "eval"
    ws = int(old.get("world_size") or 4)

    fresh = get_train_config(
        model,
        train_config_name,
        dataset=dataset,
        preprocess=preprocess,
        generate=generate,
        world_size=ws,
    )
    set_list = _overrides_from_old(old, fresh=fresh)
    overrides = parse_train_overrides(set_list)
    composed = get_train_config(
        model,
        train_config_name,
        dataset=dataset,
        preprocess=preprocess,
        generate=generate,
        world_size=ws,
        overrides=overrides,
    )
    new_train = _migrate_train_dict(old, composed=composed)
    dest = checkpoint_run_dir_from_cfg(composed)
    # root 可能非默认
    if root.resolve() != Path("cache/checkpoints").resolve():
        dest = root / composed.variant / composed.model / composed.name

    rel = composed.extra.get("run_relpath")
    _log(f"{old_dir.name} → {rel}")
    _log(f"  overrides: {set_list or '(none)'}")
    _log(f"  hash={composed.name}")

    hint = {
        "old_dir": old_dir.name,
        "run_relpath": rel,
        "config_hash": composed.name,
        "train_cmd": (
            f"python train.py --model {model} --config {train_config_name} "
            f"--dataset {dataset} --preprocess {preprocess} --generate {generate}"
            + ("" if not set_list else " " + " ".join(f"--set {s}" for s in set_list))
        ),
        "resolve_cmd": (
            f"python resolve_checkpoint.py --model {model} --config {train_config_name} "
            f"--dataset {dataset} --preprocess {preprocess} --generate {generate}"
            + ("" if not set_list else " " + " ".join(f"--set {s}" for s in set_list))
        ),
        "hardware": hardware.to_dict(),
    }

    if dry_run:
        _log(f"  dest={dest}")
        _log(f"  launch: {hint['train_cmd']}")
        return dest

    if dest.exists():
        raise FileExistsError(f"destination already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    (old_dir / "config.json").write_text(
        json.dumps(
            {"train": new_train, "model": payload.get("model")},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    save_hardware(old_dir, hardware)
    (old_dir / "launch_hint.json").write_text(
        json.dumps(hint, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    pts = sorted(old_dir.glob("checkpoint*.pt"))
    _log(f"  updating {len(pts)} checkpoint pt files")
    for pt in pts:
        _log(f"  • {pt.name}")
        _update_checkpoint_pt(pt, new_train, dry_run=False)

    old_dir.rename(dest)
    _log(f"  moved → {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("cache/checkpoints"),
        help="checkpoints root",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Only migrate this old dir name (repeatable)",
    )
    args = parser.parse_args()
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    candidates = sorted(
        p for p in root.iterdir()
        if p.is_dir()
        and p.name not in ("fast", "full")
        and not p.name.endswith(".migrated_bak")
        and not p.name.endswith(".migrating")
        and (p / "config.json").is_file()
    )
    if args.only:
        want = set(args.only)
        candidates = [p for p in candidates if p.name in want]

    if not candidates:
        raise SystemExit("no old-style checkpoint dirs to migrate")

    _log(f"hardware lock: {DEFAULT_HARDWARE}")
    for old_dir in candidates:
        migrate_one(
            old_dir,
            dry_run=args.dry_run,
            hardware=DEFAULT_HARDWARE,
            root=root,
        )
    _log("all done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
