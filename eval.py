#!/usr/bin/env python3
"""本机 TriFluency 离线评测入口（仅 ``full`` checkpoint）。

布局::

    cache/eval/{model}/{train-hash}/{eval-hash}/
      samples.txt   # 参数说明 + 逐样本（多指标）
      summary.txt   # 参数说明 + 语料汇总 / 等级

``eval-hash`` = sha256(step + 生成配置 + 样本参数含 seed)[:16]。

Usage::

    .venv/bin/python eval.py --run full/elf/<train-hash>
    .venv/bin/python eval.py --run full/elf/<hash> --generate eval \\
        --set self_cond_cfg_scale=2.0 --num-samples 1024 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
    """从 checkpoint 路径解析 ``(model, train_hash)``；要求 variant=full。"""
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
    variant, model, train_hash = parts
    if variant != "full":
        raise ValueError(
            f"eval.py only supports variant=full (got {variant!r} from {rel})"
        )
    return model, train_hash


def eval_fingerprint(
    *,
    step: int,
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
    return {
        "step": int(step),
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


def eval_hash_from_fingerprint(fp: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(fp).encode("utf-8")).hexdigest()
    return digest[:CONFIG_HASH_LEN]


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
    （已写入 eval-hash）。
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
        "--generate",
        default="eval",
        help="Generate config name under config/generate/<model>/ (default: eval)",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set_items",
        metavar="KEY=VALUE",
        help="Override generate YAML fields (repeatable), e.g. --set self_cond_cfg_scale=2",
    )
    p.add_argument("--num-tokens", type=int, default=1024)
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; batch k uses seed + k*100003 (micro_bs enters eval-hash)",
    )
    p.add_argument(
        "--micro-bs",
        type=int,
        default=8,
        help="Generate micro-batch size (default: 8; part of eval-hash)",
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

    ckpt_path = resolve_checkpoint(checkpoint=args.checkpoint, run=args.run)
    model_name, train_hash = parse_full_run(ckpt_path)
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

    if model_name == "elf":
        from models.elf.ace import attach_ace_identity

        attach_ace_identity(
            model, model_hash=train_hash, step=step, tokenizer=tokenizer_name,
        )

    overrides = parse_generate_sets(args.set_items)
    gen_cfg = get_generate(model_name, args.generate, overrides=overrides or None)
    sampling_cfg = gen_cfg.to_sampling_cfg()
    if args.temperature is not None:
        sampling_cfg["temperature"] = args.temperature
    if args.top_k is not None:
        sampling_cfg["top_k"] = args.top_k

    use_ema = (not args.no_ema) and bool(ema_state)
    fp = eval_fingerprint(
        step=step,
        generate_profile=args.generate,
        sampling_cfg=sampling_cfg,
        num_samples=args.num_samples,
        num_tokens=args.num_tokens,
        seed=args.seed,
        micro_bs=args.micro_bs,
        use_ema=use_ema,
        gen_eval_model=args.gen_eval_model,
        gen_eval_dtype=args.gen_eval_dtype,
    )
    ehash = eval_hash_from_fingerprint(fp)
    out_dir = EVAL_ROOT / model_name / train_hash / ehash
    out_dir.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "run": f"full/{model_name}/{train_hash}",
        "model": model_name,
        "train_hash": train_hash,
        "eval_hash": ehash,
        "step": step,
        "out_dir": str(out_dir),
        **fp,
        "generate_overrides": overrides,
    }
    (out_dir / "fingerprint.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _log(
        f"model={model_name} train_hash={train_hash} eval_hash={ehash} "
        f"step={step} samples={args.num_samples}x{args.num_tokens} "
        f"seed={args.seed} ema={use_ema} generate={args.generate} "
        f"sampling={sampling_cfg}",
    )
    _log(f"out_dir={out_dir}")

    ctx = swap_ema_weights(model, ema_state if use_ema else None)
    with ctx:
        texts, nfe = generate_texts_chunked(
            model,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            num_tokens=args.num_tokens,
            seed=args.seed,
            micro_bs=args.micro_bs,
            device=device,
            dtype=dtype,
            sampling_cfg=sampling_cfg,
        )
    _log(f"Generation finished nfe_last={nfe}")

    from eval.protocol import compute_trifluency
    from eval.report import write_samples_report, write_summary_report

    per_sample, summary = compute_trifluency(
        texts,
        device=device,
        max_length=args.num_tokens,
        n_min_accept=args.n_min_accept,
        gen_eval_model=args.gen_eval_model,
        gen_eval_dtype=args.gen_eval_dtype,
    )

    samples_path = out_dir / "samples.txt"
    summary_path = out_dir / "summary.txt"
    write_samples_report(
        samples_path, params=params, texts=texts, per_sample=per_sample,
    )
    write_summary_report(summary_path, params=params, summary=summary)

    from eval.protocol import _sanitize

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
        f"DONE accept@human={summary['accept_at_human']:.4f} "
        f"median_rep={summary['median_rep']:.4f} "
        f"nonword%={summary['nonword_word_pct']:.3f} "
        f"clean_ppl={clean_s} "
        f"cola_g={summary['cola_g']:.4f} "
        f"raw_gen_ppl={summary['raw_gen_ppl']:.4f}",
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
