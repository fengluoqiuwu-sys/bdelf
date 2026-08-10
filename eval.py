#!/usr/bin/env python3
"""本机 TriFluency 离线评测入口（仅 ``full`` checkpoint）。

布局::

    cache/eval/{model}/{model-hash}/{step}/{generate-hash}/
      samples.txt   # 参数说明 + 逐样本（多指标）
      summary.txt   # 参数说明 + 语料汇总 / 等级

``generate-hash`` = sha256(生成配置 + 样本参数含 seed；不含 step)[:16]。
``model-hash`` 即训练 config-hash（与 ``cache/checkpoints/full/{model}/{hash}/`` 一致）。

单组（兼容旧用法）::

    .venv/bin/python eval.py --run full/elf/<model-hash> --micro-bs 8
    .venv/bin/python eval.py --run full/elf/<hash> --generate eval \\
        --set self_cond_cfg_scale=2.0 --num-samples 1024 --seed 42 --micro-bs 8

多组扫参（生成模型与 gpt2/CoLA 各只加载一次）::

    .venv/bin/python eval.py --run full/odar/<hash> \\
        --table odar-sc-ace --micro-bs 8

``--table`` 解析 ``config/eval/tables/<name>.yaml``，或接受显式 ``*.yaml`` 路径；
扫参模式必须同时给 ``--micro-bs``（本机生成 micro-batch）。
流程：先按表逐组生成并落盘指纹，释放生成模型后再加载 gpt2/CoLA 统一打分。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

import hf_config  # noqa: F401
from generate import (
    generate_tokens,
    list_checkpoint_runs,
    load_model_meta,
    resolve_checkpoint,
    resolve_device,
    resolve_dtype,
    _checkpoint_root,
)
from models import build_model
from tokenizer import get_tokenizer
from train.ema import swap_ema_weights
from train.generate_config import get_generate
from train.run_path import CONFIG_HASH_LEN, canonical_json

_EVAL_LOG = "[eval]"
EVAL_ROOT = Path("cache/eval")
EVAL_TABLE_DIR = Path("config/eval/tables")


def _log(msg: str, *, file=None) -> None:
    if file is None:
        file = sys.stdout
    print(f"{_EVAL_LOG} {msg}", file=file, flush=True)


def parse_generate_sets(items: list[str] | None) -> dict[str, Any]:
    """解析 ``--set key=value``（生成 YAML 根级键；值用 YAML 标量）。"""
    out: dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --set {raw!r}; expected key=value")
        key, value_raw = raw.split("=", 1)
        key = key.strip()
        if not key or "." in key or key.startswith("_"):
            raise ValueError(
                f"Invalid --set key {key!r}; use a single generate-config field "
                f"(e.g. self_cond_cfg_scale=2.0)"
            )
        try:
            value = yaml.safe_load(value_raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid --set value for {key!r}: {value_raw!r}") from exc
        out[key] = value
    return out


def parse_full_run(ckpt_path: Path) -> tuple[str, str]:
    """从 checkpoint 路径解析 ``(model, model_hash)``；要求 variant=full。"""
    root = _checkpoint_root().resolve()
    try:
        rel = ckpt_path.resolve().parent.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Checkpoint not under {root}: {ckpt_path}. "
            "eval.py only accepts full runs under cache/checkpoints/full/..."
        ) from exc
    parts = rel.parts
    if len(parts) != 3:
        raise ValueError(
            f"Expected cache/checkpoints/{{variant}}/{{model}}/{{hash}}/, got {rel}"
        )
    variant, model, model_hash = parts
    if variant != "full":
        raise ValueError(
            f"eval.py only supports variant=full (got {variant!r} from {rel})"
        )
    return model, model_hash


def generate_fingerprint(
    *,
    generate_profile: str,
    sampling_cfg: dict[str, Any],
    num_samples: int,
    num_tokens: int,
    seed: int,
    micro_bs: int,
    use_ema: bool,
    gen_eval_model: str,
    gen_eval_dtype: str,
    protocol: str = "trifluency-v1",
) -> dict[str, Any]:
    """进入 generate-hash 的字段（不含 step；step 在目录路径中）。"""
    return {
        "generate_profile": generate_profile,
        "sampling": dict(sampling_cfg),
        "num_samples": int(num_samples),
        "num_tokens": int(num_tokens),
        "seed": int(seed),
        "micro_bs": int(micro_bs),
        "use_ema": bool(use_ema),
        "gen_eval_model": gen_eval_model,
        "gen_eval_dtype": gen_eval_dtype,
        "protocol": protocol,
    }


def generate_hash_from_fingerprint(fp: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(fp).encode("utf-8")).hexdigest()
    return digest[:CONFIG_HASH_LEN]


def eval_out_dir(model: str, model_hash: str, step: int, generate_hash: str) -> Path:
    return EVAL_ROOT / model / model_hash / str(int(step)) / generate_hash


def resolve_eval_table_path(table: str) -> Path:
    """``name`` → ``config/eval/tables/<name>.yaml``；否则当作路径。"""
    raw = Path(table)
    if raw.suffix in (".yaml", ".yml") or "/" in table or "\\" in table:
        path = raw
    else:
        path = EVAL_TABLE_DIR / f"{table}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Eval table not found: {path}")
    if path.stem == "prototype":
        raise ValueError("Prototype eval table cannot be used with --table")
    return path


@dataclass
class EvalJob:
    """一组待跑的 generate 评测。"""

    generate_profile: str
    overrides: dict[str, Any]
    num_samples: int
    num_tokens: int
    seed: int
    index: int


@dataclass
class PendingScore:
    """已生成、待打分的一组。"""

    job: EvalJob
    params: dict[str, Any]
    out_dir: Path
    texts: list[str]


def load_eval_table(path: Path) -> tuple[dict[str, Any], list[EvalJob]]:
    """加载扫参表；返回 (表头元数据, jobs)。"""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    name = raw.get("name")
    if not name or name == "prototype":
        raise ValueError(f"{path}: invalid name {name!r}")
    generate_profile = str(raw.get("generate") or "eval")
    num_samples = int(raw.get("num_samples", 1024))
    num_tokens = int(raw.get("num_tokens", 1024))
    seed = int(raw.get("seed", 42))
    runs = raw.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{path}: runs must be a non-empty list")
    jobs: list[EvalJob] = []
    for i, item in enumerate(runs):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: runs[{i}] must be a mapping of overrides")
        overrides = {
            k: v
            for k, v in item.items()
            if not str(k).startswith("_")
        }
        jobs.append(
            EvalJob(
                generate_profile=generate_profile,
                overrides=overrides,
                num_samples=num_samples,
                num_tokens=num_tokens,
                seed=seed,
                index=i,
            )
        )
    meta = {
        "table": str(path),
        "table_name": name,
        "n_runs": len(jobs),
    }
    return meta, jobs


def load_model_with_ema(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, int, dict | None, dict[str, torch.Tensor] | None]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_meta = load_model_meta(ckpt_path, ck)
    model_cfg = dict(model_meta["config"] or {})
    if model_meta["name"] == "cola":
        model_cfg["load_vae_weights"] = False
    model = build_model(model_meta["name"], model_cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    train_cfg = ck.get("train_config")
    dtype = resolve_dtype(device, train_cfg)
    model = model.to(device=device, dtype=dtype)
    ema_raw = ck.get("ema")
    ema_state: dict[str, torch.Tensor] | None = None
    if isinstance(ema_raw, dict) and ema_raw:
        ema_state = {k: v.to(device=device) for k, v in ema_raw.items()}
    return model, model_meta, int(ck.get("step", 0)), train_cfg, ema_state


def generate_texts_chunked(
    model: torch.nn.Module,
    *,
    tokenizer,
    num_samples: int,
    num_tokens: int,
    seed: int,
    micro_bs: int,
    device: torch.device,
    dtype: torch.dtype,
    sampling_cfg: dict[str, Any],
) -> tuple[list[str], int]:
    """按 micro batch 生成。

    第 ``k`` 个 batch 使用 ``seed + k * 100003``。复现须固定 ``micro_bs``
    （已写入 generate-hash）。
    """
    texts: list[str] = []
    last_nfe = 0
    remaining = num_samples
    chunk_i = 0
    while remaining > 0:
        bs = min(micro_bs, remaining)
        tokens, nfe = generate_tokens(
            model,
            num_tokens=num_tokens,
            num_samples=bs,
            seed=seed + chunk_i * 100003,
            device=device,
            dtype=dtype,
            sampling_cfg=sampling_cfg,
        )
        last_nfe = nfe
        for row in tokens.detach().cpu():
            texts.append(
                tokenizer.decode(row.tolist(), skip_special_tokens=True)
            )
        remaining -= bs
        chunk_i += 1
        _log(f"generated {len(texts)}/{num_samples}")
    return texts, last_nfe


def build_sampling_cfg(
    model_name: str,
    generate_profile: str,
    overrides: dict[str, Any],
    *,
    temperature: float | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    gen_cfg = get_generate(
        model_name, generate_profile, overrides=overrides or None,
    )
    sampling_cfg = gen_cfg.to_sampling_cfg()
    if temperature is not None:
        sampling_cfg["temperature"] = temperature
    if top_k is not None:
        sampling_cfg["top_k"] = top_k
    return sampling_cfg


def prepare_job_params(
    *,
    job: EvalJob,
    model_name: str,
    model_hash: str,
    step: int,
    ckpt_path: Path,
    use_ema: bool,
    micro_bs: int,
    gen_eval_model: str,
    gen_eval_dtype: str,
    temperature: float | None,
    top_k: int | None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    sampling_cfg = build_sampling_cfg(
        model_name,
        job.generate_profile,
        job.overrides,
        temperature=temperature,
        top_k=top_k,
    )
    fp = generate_fingerprint(
        generate_profile=job.generate_profile,
        sampling_cfg=sampling_cfg,
        num_samples=job.num_samples,
        num_tokens=job.num_tokens,
        seed=job.seed,
        micro_bs=micro_bs,
        use_ema=use_ema,
        gen_eval_model=gen_eval_model,
        gen_eval_dtype=gen_eval_dtype,
    )
    ghash = generate_hash_from_fingerprint(fp)
    out_dir = eval_out_dir(model_name, model_hash, step, ghash)
    params: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "run": f"full/{model_name}/{model_hash}",
        "model": model_name,
        "model_hash": model_hash,
        "generate_hash": ghash,
        "step": step,
        "out_dir": str(out_dir),
        **fp,
        "generate_overrides": job.overrides,
    }
    return params, out_dir, sampling_cfg


def score_and_write(
    *,
    pending: PendingScore,
    device: torch.device,
    n_min_accept: int,
    gen_eval_model: str,
    gen_eval_dtype: str,
    scorers,
) -> dict[str, Any]:
    from eval.protocol import _sanitize, compute_trifluency
    from eval.report import write_samples_report, write_summary_report

    job = pending.job
    params = pending.params
    out_dir = pending.out_dir
    texts = pending.texts

    per_sample, summary = compute_trifluency(
        texts,
        device=device,
        max_length=job.num_tokens,
        n_min_accept=n_min_accept,
        gen_eval_model=gen_eval_model,
        gen_eval_dtype=gen_eval_dtype,
        scorers=scorers,
    )

    samples_path = out_dir / "samples.txt"
    summary_path = out_dir / "summary.txt"
    write_samples_report(
        samples_path, params=params, texts=texts, per_sample=per_sample,
    )
    write_summary_report(summary_path, params=params, summary=summary)
    (out_dir / "summary.json").write_text(
        json.dumps(_sanitize(summary), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    _log(f"wrote {samples_path}")
    _log(f"wrote {summary_path}")
    clean = summary["clean_ppl"]
    clean_s = f"{clean:.4f}" if isinstance(clean, float) and clean == clean else "nan"
    _log(
        f"DONE[{job.index + 1}] accept@human={summary['accept_at_human']:.4f} "
        f"median_rep={summary['median_rep']:.4f} "
        f"nonword%={summary['nonword_word_pct']:.3f} "
        f"clean_ppl={clean_s} "
        f"cola_g={summary['cola_g']:.4f} "
        f"raw_gen_ppl={summary['raw_gen_ppl']:.4f}",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TriFluency offline eval for full checkpoints",
    )
    p.add_argument(
        "--checkpoint",
        help="Path to a .pt under cache/checkpoints/full/...",
    )
    p.add_argument(
        "--run",
        help="Run relpath: full/{model}/{train-hash}",
    )
    p.add_argument(
        "--table",
        default=None,
        help=(
            "Multi-run generate table: name under config/eval/tables/ "
            "or a YAML path. Requires --micro-bs."
        ),
    )
    p.add_argument(
        "--generate",
        default=None,
        help="Generate config name under config/generate/<model>/ (default: eval)",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set_items",
        metavar="KEY=VALUE",
        help="Override generate YAML fields (repeatable); not allowed with --table",
    )
    p.add_argument("--num-tokens", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed; batch k uses seed + k*100003 (micro_bs enters generate-hash)",
    )
    p.add_argument(
        "--micro-bs",
        type=int,
        default=None,
        help=(
            "Generate micro-batch size (part of generate-hash). "
            "Required with --table; default 8 for single-run"
        ),
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--gen-eval-model", default="gpt2-large")
    p.add_argument(
        "--gen-eval-dtype",
        default="bf16",
        choices=("bf16", "fp16", "fp32"),
    )
    p.add_argument("--n-min-accept", type=int, default=50)
    p.add_argument("--no-ema", action="store_true", help="Do not swap EMA weights")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if summary.json already exists for the same generate-hash",
    )
    p.add_argument(
        "--list-runs",
        action="store_true",
        help="List full runs with checkpoint_latest.pt and exit",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.list_runs:
        runs = [
            r for r in list_checkpoint_runs()
            if r.relative_to(_checkpoint_root()).parts[0] == "full"
        ]
        if not runs:
            _log(f"No full checkpoints under {_checkpoint_root()}")
            return
        for run_dir in runs:
            ckpt = run_dir / "checkpoint_latest.pt"
            rel = run_dir.relative_to(_checkpoint_root())
            _log(f"{rel}\t{ckpt}")
        return

    table_meta: dict[str, Any] | None = None
    if args.table is not None:
        if args.micro_bs is None:
            raise SystemExit("--table requires --micro-bs (local generate batch)")
        if args.set_items:
            raise SystemExit(
                "--set cannot be combined with --table (put overrides in runs:)"
            )
        micro_bs = int(args.micro_bs)
        table_path = resolve_eval_table_path(args.table)
        table_meta, jobs = load_eval_table(table_path)
        for job in jobs:
            if args.num_samples is not None:
                job.num_samples = int(args.num_samples)
            if args.num_tokens is not None:
                job.num_tokens = int(args.num_tokens)
            if args.seed is not None:
                job.seed = int(args.seed)
            if args.generate is not None:
                job.generate_profile = args.generate
        _log(
            f"table={table_meta['table']} name={table_meta['table_name']} "
            f"n_runs={table_meta['n_runs']} micro_bs={micro_bs}",
        )
    else:
        micro_bs = 8 if args.micro_bs is None else int(args.micro_bs)
        overrides = parse_generate_sets(args.set_items)
        jobs = [
            EvalJob(
                generate_profile=args.generate or "eval",
                overrides=overrides,
                num_samples=int(args.num_samples or 1024),
                num_tokens=int(args.num_tokens or 1024),
                seed=int(args.seed if args.seed is not None else 42),
                index=0,
            )
        ]

    if micro_bs < 1:
        raise SystemExit(f"--micro-bs must be >= 1, got {micro_bs}")

    ckpt_path = resolve_checkpoint(checkpoint=args.checkpoint, run=args.run)
    model_name, model_hash = parse_full_run(ckpt_path)
    device = resolve_device(args.device)

    _log(f"Loading checkpoint: {ckpt_path}")
    model, model_meta, step, train_cfg, ema_state = load_model_with_ema(
        ckpt_path, device,
    )
    if model_meta["name"] != model_name:
        _log(
            f"Warning: path model={model_name} vs meta name={model_meta['name']}",
            file=sys.stderr,
        )
        model_name = model_meta["name"]
    dtype = resolve_dtype(device, train_cfg)
    tokenizer_name = model_meta["config"].get("tokenizer")
    if not tokenizer_name:
        raise ValueError("Model config missing tokenizer")
    tokenizer = get_tokenizer(tokenizer_name)

    if model_name in ("elf", "odar"):
        from models.elf.ace import attach_ace_identity

        attach_ace_identity(
            model, model_hash=model_hash, step=step, tokenizer=tokenizer_name,
        )

    use_ema = (not args.no_ema) and bool(ema_state)
    # 同 generate-hash 已有 summary.json 则跳过（--force 才重跑）
    skip_existing = not args.force
    pending: list[PendingScore] = []

    # ---------- 阶段 1：生成模型只加载一次，扫完全部 generate 配置 ----------
    for job in jobs:
        params, out_dir, sampling_cfg = prepare_job_params(
            job=job,
            model_name=model_name,
            model_hash=model_hash,
            step=step,
            ckpt_path=ckpt_path,
            use_ema=use_ema,
            micro_bs=micro_bs,
            gen_eval_model=args.gen_eval_model,
            gen_eval_dtype=args.gen_eval_dtype,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        ghash = params["generate_hash"]
        if skip_existing and (out_dir / "summary.json").is_file():
            _log(
                f"[{job.index + 1}/{len(jobs)}] skip existing "
                f"generate_hash={ghash} overrides={job.overrides}",
            )
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "fingerprint.json").write_text(
            json.dumps(params, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _log(
            f"[{job.index + 1}/{len(jobs)}] generate generate_hash={ghash} "
            f"step={step} samples={job.num_samples}x{job.num_tokens} "
            f"seed={job.seed} micro_bs={micro_bs} ema={use_ema} "
            f"overrides={job.overrides} sampling={sampling_cfg}",
        )
        _log(f"out_dir={out_dir}")

        ctx = swap_ema_weights(model, ema_state if use_ema else None)
        with ctx:
            texts, nfe = generate_texts_chunked(
                model,
                tokenizer=tokenizer,
                num_samples=job.num_samples,
                num_tokens=job.num_tokens,
                seed=job.seed,
                micro_bs=micro_bs,
                device=device,
                dtype=dtype,
                sampling_cfg=sampling_cfg,
            )
        _log(f"Generation finished nfe_last={nfe}")
        pending.append(
            PendingScore(job=job, params=params, out_dir=out_dir, texts=texts)
        )

    # 释放生成模型，给 gpt2-large / CoLA 腾显存
    del model, ema_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log(f"Released generative model; pending_score={len(pending)}")

    if not pending:
        _log("Nothing to score (all skipped or empty).")
        return

    # ---------- 阶段 2：gpt2 / CoLA 只加载一次，打完所有组 ----------
    from eval.protocol import load_trifluency_scorers

    scorers = load_trifluency_scorers(
        device=device,
        gen_eval_model=args.gen_eval_model,
        gen_eval_dtype=args.gen_eval_dtype,
    )
    try:
        for item in pending:
            _log(
                f"[{item.job.index + 1}/{len(jobs)}] score "
                f"generate_hash={item.params['generate_hash']}",
            )
            score_and_write(
                pending=item,
                device=device,
                n_min_accept=args.n_min_accept,
                gen_eval_model=args.gen_eval_model,
                gen_eval_dtype=args.gen_eval_dtype,
                scorers=scorers,
            )
    finally:
        scorers.close()

    if table_meta is not None:
        _log(
            f"ALL DONE table={table_meta['table_name']} "
            f"n_runs={table_meta['n_runs']} scored={len(pending)} step={step}",
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("Interrupt received; exiting.")
        raise SystemExit(130)
    except Exception as exc:
        _log(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
