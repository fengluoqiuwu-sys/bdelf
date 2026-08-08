#!/usr/bin/env python3
"""单卡显存探针：模拟 rank0（训练模型 + 优化器 + EMA + gpt2-large）测 micro-batch 峰值。

用法（仓库根）::

    .venv/bin/python scripts/vram_probe.py \\
      --model elf --config 100m-full \\
      --dataset owt --preprocess elf --generate eval \\
      --batches 8,16,24,32

只打印结果，不写配置 / checkpoint。本机可跑但不建议；以远端目标卡为准。
目的：测各 micro-batch 上限，写入本地 ``temp/vram-probe/`` 表；具体选用在开训时按
``global_batch_size`` 整除约束再定（见 skill ``vram-probe``）。
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import repo_env

repo_env.ensure_repo_root()

import hf_config  # noqa: F401
import torch
import torch.nn as nn
import yaml

# 与 train.py 一致：开 TF32 matmul（Inductor bool-eq patch 见下方）
from dataset import list_datasets
from models import (
    build_model,
    list_model_configs,
    list_models,
    resolve_model_config_path,
)
from preprocess import get_preprocessed, list_preprocess
from train import (
    FL_TrainConfig,
    get_train_config,
    list_generate,
    list_train_configs,
    list_train_models,
    parse_train_overrides,
)
from train.batching import TokenChunkDataset, fetch_train_batch
from train.checkpoint import unwrap_model
from train.ema import ema_update, init_ema
from train.eval import (
    forward_loss,
    get_amp_dtype,
    load_gen_eval_baseline,
    uses_dual_branch_logging,
)
from train.muon import build_optimizer

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def _patch_inductor_bool_eq() -> None:
    """与 train.py 同名补丁：避免 compile 时 bool ValueRanges.eq 炸掉。"""
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
            if a.is_singleton() and b.is_singleton():
                return ValueRanges.wrap(sympy.false)
            return ValueRanges(sympy.false, sympy.true)
        if a.lower > b.upper or b.lower > a.upper:
            return ValueRanges.wrap(sympy.false)
        return ValueRanges(sympy.false, sympy.true)

    SymPyValueRangeAnalysis.eq = _eq


_patch_inductor_bool_eq()

# 固定候选集合（与 skill / auto-train 约定一致）
ALLOWED_BATCHES: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)
# compile 首几步在编内核；多留 warmup 再记峰
WARMUP_STEPS = 3
MEASURE_STEPS = 2


@dataclass
class ProbeRow:
    batch_size: int
    allocated_peak_gib: float | None
    reserved_peak_gib: float | None
    smi_used_gib: float | None
    status: str  # ok | oom | skip


def _bytes_to_gib(n: int | float) -> float:
    return float(n) / (1024**3)


def _preload_frozen_encoders(model: nn.Module) -> None:
    backbone = getattr(model, "backbone", None)
    ensure = getattr(backbone, "_ensure_encoder", None)
    if callable(ensure):
        ensure()


def _sample_train_branch(model: nn.Module, device: torch.device) -> str:
    raw = unwrap_model(model)
    p = float(raw.backbone.decoder_prob)
    return "decode" if torch.rand((), device=device) < p else "denoise"


def _nvidia_smi_used_gib(device_index: int) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        # nvidia-smi 报告 MiB
        return float(out.splitlines()[0]) / 1024.0
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _parse_batches(raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return list(ALLOWED_BATCHES)
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError as exc:
            raise SystemExit(f"Invalid --batches entry {part!r}") from exc
        if v not in ALLOWED_BATCHES:
            raise SystemExit(
                f"batch_size={v} not in allowed set {list(ALLOWED_BATCHES)}"
            )
        if v not in out:
            out.append(v)
    if not out:
        raise SystemExit("--batches produced an empty list")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    models = list_models() or ["<none>"]
    datasets = list_datasets() or ["<none>"]
    preprocess_names = list_preprocess() or ["<none>"]
    parser = argparse.ArgumentParser(
        description=(
            "Single-GPU VRAM probe: train model + optimizer (+EMA) + gpt2-large; "
            "measure peak memory across micro-batch sizes (stop on first OOM)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Allowed batches: "
            + ",".join(str(x) for x in ALLOWED_BATCHES)
            + "\n"
            "Probes all requested sizes (no global_batch filter). "
            "Pick train batch later from temp/vram-probe table + global_bs.\n"
            "Example:\n"
            "  python scripts/vram_probe.py --model elf --config 100m-full "
            "--dataset owt --preprocess elf --generate eval "
            "--batches 8,16,24,32\n"
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model family; options: {', '.join(models)}",
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
        help=f"Dataset (config/datasets/); options: {', '.join(datasets)}",
    )
    parser.add_argument(
        "--preprocess",
        required=True,
        help=(
            f"Preprocess (config/preprocess/); "
            f"options: {', '.join(preprocess_names)}"
        ),
    )
    parser.add_argument(
        "--generate",
        required=True,
        help="Generate config under config/generate/<model>/ (e.g. eval)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="SECTION.KEY=VALUE",
        help="Same as train.py --set (repeatable)",
    )
    parser.add_argument(
        "--batches",
        default=None,
        help=(
            "Comma-separated micro-batch sizes to try "
            f"(subset of {list(ALLOWED_BATCHES)}); default=all allowed"
        ),
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        dest="intended_world_size",
        help=(
            "Intended multi-GPU world size (metadata only, default 4 for AI full). "
            "Does not filter candidates; probe always runs on 1 GPU."
        ),
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Override schedule: disable torch.compile (default follows recipe)",
    )
    return parser


def _validate_and_load(
    args: argparse.Namespace,
) -> tuple[str, str, FL_TrainConfig]:
    models = list_models()
    if args.model not in models:
        raise SystemExit(
            f"Unknown model {args.model!r}. Available: {', '.join(models)}"
        )
    train_models = list_train_models()
    if args.model not in train_models:
        raise SystemExit(
            f"Model {args.model!r} has no train recipe. "
            f"Available: {', '.join(train_models)}"
        )
    configs = list_train_configs(args.model)
    if args.train_config not in configs:
        raise SystemExit(
            f"Unknown train config {args.train_config!r}. "
            f"{args.model} available: {', '.join(configs)}"
        )
    size = args.train_config.rsplit("-", 1)[0]
    try:
        resolve_model_config_path(args.model, size)
    except FileNotFoundError as exc:
        available = ", ".join(list_model_configs(args.model)) or "<none>"
        raise SystemExit(
            f"Model architecture config not found for {args.model}/{size}. "
            f"Available: {available}\n{exc}"
        ) from exc

    datasets = list_datasets()
    if args.dataset not in datasets:
        raise SystemExit(
            f"Unknown dataset {args.dataset!r}. "
            f"Available: {', '.join(datasets) or '<none>'}"
        )
    preprocess_names = list_preprocess()
    if args.preprocess not in preprocess_names:
        raise SystemExit(
            f"Unknown preprocess {args.preprocess!r}. "
            f"Available: {', '.join(preprocess_names) or '<none>'}"
        )
    generate_names = list_generate(args.model)
    if args.generate not in generate_names:
        raise SystemExit(
            f"Unknown generate config {args.generate!r}. "
            f"{args.model} available: {', '.join(generate_names) or '<none>'}"
        )

    if args.intended_world_size < 1:
        raise SystemExit("--world-size must be >= 1")

    try:
        overrides = parse_train_overrides(args.overrides)
    except ValueError as exc:
        raise SystemExit(f"Invalid --set override: {exc}") from exc

    overrides = dict(overrides)
    if args.no_compile:
        schedule_ov = dict(overrides.get("schedule") or {})
        schedule_ov["compile"] = False
        overrides["schedule"] = schedule_ov

    # 探针进程始终 world_size=1；intended_world_size 仅用于 batch 过滤与 accum
    try:
        cfg = get_train_config(
            args.model,
            args.train_config,
            dataset=args.dataset,
            preprocess=args.preprocess,
            generate=args.generate,
            world_size=1,
            overrides=overrides,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Failed to load train config: {exc}") from exc
    return args.model, size, cfg


def _one_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    batch: torch.Tensor,
    cfg: FL_TrainConfig,
    amp_dtype: torch.dtype,
    device: torch.device,
    dual_branch: bool,
    mixed_branch: bool,
    ema_state: dict[str, torch.Tensor] | None,
    ema_decay: float,
    grad_accum_steps: int,
) -> None:
    """对齐 train_loop 单 micro-step：autocast → loss/accum → backward → step。"""
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
        if dual_branch and not mixed_branch:
            train_branch: str | None = _sample_train_branch(model, device)
        else:
            train_branch = None
        micro_loss = forward_loss(model, batch, branch=train_branch)
    if not torch.isfinite(micro_loss):
        raise RuntimeError("non-finite loss during VRAM probe")
    (micro_loss / grad_accum_steps).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    if ema_state is not None:
        ema_update(ema_state, model, ema_decay)


def _print_table(rows: list[ProbeRow]) -> None:
    headers = (
        "batch_size",
        "alloc_peak_GiB",
        "reserved_peak_GiB",
        "smi_used_GiB",
        "status",
    )
    lines = ["\t".join(headers)]
    for r in rows:
        alloc = (
            f"{r.allocated_peak_gib:.2f}" if r.allocated_peak_gib is not None else "-"
        )
        reserved = (
            f"{r.reserved_peak_gib:.2f}" if r.reserved_peak_gib is not None else "-"
        )
        smi = f"{r.smi_used_gib:.2f}" if r.smi_used_gib is not None else "-"
        lines.append(
            f"{r.batch_size}\t{alloc}\t{reserved}\t{smi}\t{r.status}"
        )
    print("\n".join(lines))


def run_probe(
    model_name: str,
    model_size: str,
    cfg: FL_TrainConfig,
    *,
    batches: list[int],
    intended_world_size: int,
) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("VRAM probe requires CUDA")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    total_gib = _bytes_to_gib(props.total_memory)
    hostname = socket.gethostname()

    # 不按 recipe global_bs 过滤：测绝对上限；开训时再按整除约束选型
    candidates = sorted(batches)
    print("=== VRAM probe ===")
    print(f"host: {hostname}")
    print(f"gpu: {props.name} (index={device.index})")
    print(f"total_memory_GiB: {total_gib:.2f}")
    print(
        f"model: {model_name}/{model_size}  variant={cfg.variant}  "
        f"hash={cfg.name}"
    )
    print(
        f"recipe_global_batch_size={cfg.global_batch_size}  "
        f"(metadata only; not used to filter)  "
        f"intended_world_size={intended_world_size}  "
        f"(probe world_size=1)"
    )
    compile_enabled = bool(cfg.extra.get("compile", False))
    print(f"torch.compile: {compile_enabled}")
    print(f"probe_batches: {candidates}")
    print("grad_accum_steps: 1 (peak of one micro-step; independent of global_bs)")
    if not candidates:
        raise SystemExit("No batch sizes to probe")

    print("Loading preprocessed train split ...")
    try:
        preprocessed = get_preprocessed(cfg.preprocess, cfg.dataset)
    except FileNotFoundError as exc:
        raise SystemExit(f"Preprocessed data unavailable: {exc}") from exc
    splits = preprocessed.get_splits()
    if "train" not in splits:
        raise SystemExit(f"Dataset missing train split; splits={splits}")
    train_ds = TokenChunkDataset(preprocessed.load_split("train"))

    model_cfg_path = resolve_model_config_path(model_name, model_size)
    with open(model_cfg_path, encoding="utf-8") as f:
        model_cfg: dict[str, Any] = yaml.safe_load(f) or {}
    if model_name == "cola":
        model_cfg = dict(model_cfg)
        model_cfg["train_variant"] = cfg.variant

    print("Building train model + optimizer ...")
    model = build_model(model_name, model_cfg).to(device)
    _preload_frozen_encoders(model)
    if compile_enabled:
        print("torch.compile enabled (same as full train schedule)")
        model = torch.compile(model)
    optimizer = build_optimizer(unwrap_model(model), cfg)
    ema_decay = float(cfg.extra.get("ema_decay", 0.0) or 0.0)
    ema_state: dict[str, torch.Tensor] | None = (
        init_ema(model) if ema_decay > 0.0 else None
    )
    if ema_state is not None:
        print(f"EMA enabled: decay={ema_decay:g}")

    print(
        f"Loading gen-eval baseline {cfg.gen_eval_model} "
        f"({cfg.gen_eval_model_dtype} on {cfg.gen_eval_model_device}) ..."
    )
    gpt2_model = load_gen_eval_baseline(cfg)
    # 保持引用，防止被 GC；常驻显存
    _gpt2_anchor = gpt2_model  # noqa: F841

    amp_dtype = get_amp_dtype(cfg.dtype)
    dual_branch = uses_dual_branch_logging(model)
    mixed_branch = bool(getattr(unwrap_model(model), "mixed_branch_training", False))
    model.train()

    rows: list[ProbeRow] = []
    exit_code = 0
    step_counter = 0

    for bs in candidates:
        # accum=1：单 micro-step 峰值与真实 accum 无关（仅 loss 缩放）
        print(f"\n--- probing batch_size={bs} (grad_accum=1) ---")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for i in range(WARMUP_STEPS + MEASURE_STEPS):
                batch = fetch_train_batch(
                    train_ds,
                    step_counter,
                    bs,
                    world_size=1,
                    rank=0,
                    seed=cfg.seed,
                )
                step_counter += 1
                batch = batch.to(device, non_blocking=True)
                if i == WARMUP_STEPS:
                    # warmup（含 compile）后重新记峰值
                    torch.cuda.reset_peak_memory_stats(device)
                _one_train_step(
                    model,
                    optimizer,
                    batch=batch,
                    cfg=cfg,
                    amp_dtype=amp_dtype,
                    device=device,
                    dual_branch=dual_branch,
                    mixed_branch=mixed_branch,
                    ema_state=ema_state,
                    ema_decay=ema_decay,
                    grad_accum_steps=1,
                )
            torch.cuda.synchronize(device)
            alloc = _bytes_to_gib(torch.cuda.max_memory_allocated(device))
            reserved = _bytes_to_gib(torch.cuda.max_memory_reserved(device))
            smi = _nvidia_smi_used_gib(int(device.index or 0))
            row = ProbeRow(
                batch_size=bs,
                allocated_peak_gib=alloc,
                reserved_peak_gib=reserved,
                smi_used_gib=smi,
                status="ok",
            )
            rows.append(row)
            print(
                f"ok: alloc_peak={alloc:.2f} GiB  "
                f"reserved_peak={reserved:.2f} GiB  "
                f"smi_used={smi if smi is not None else '-'}"
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            rows.append(
                ProbeRow(
                    batch_size=bs,
                    allocated_peak_gib=None,
                    reserved_peak_gib=None,
                    smi_used_gib=_nvidia_smi_used_gib(int(device.index or 0)),
                    status="oom",
                )
            )
            print(f"OOM at batch_size={bs}; stopping.")
            exit_code = 2
            break

    print("\n=== results ===")
    _print_table(rows)
    ok_batches = [r.batch_size for r in rows if r.status == "ok"]
    if ok_batches:
        print(f"max_ok_batch_size: {max(ok_batches)}")
    else:
        print("max_ok_batch_size: (none)")
    return exit_code


def main() -> None:
    args = build_arg_parser().parse_args()
    model_name, model_size, cfg = _validate_and_load(args)
    batches = _parse_batches(args.batches)
    code = run_probe(
        model_name,
        model_size,
        cfg,
        batches=batches,
        intended_world_size=args.intended_world_size,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupt received; exiting.", file=sys.stderr)
        raise SystemExit(130) from None
