#!/usr/bin/env python3
"""Generate text from the latest training checkpoint.

Usage:
    python generate.py
    python generate.py --run bdelf-100m-full-muon
    python generate.py --checkpoint cache/checkpoints/bdelf-100m-full-muon/checkpoint_latest.pt
    python generate.py --num-tokens 1024 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import hf_config  # noqa: F401
from models import build_model
from tokenizer import get_tokenizer
from train import CHECKPOINT_ROOT

_GENERATE_LOG = "[generate]"


def _log(msg: str, *, file=None) -> None:
    if file is None:
        file = sys.stdout
    print(f"{_GENERATE_LOG} {msg}", file=file, flush=True)


def _checkpoint_root() -> Path:
    return Path(CHECKPOINT_ROOT)


def list_checkpoint_runs(root: Path | None = None) -> list[Path]:
    """Return run directories that contain ``checkpoint_latest.pt``."""
    root = root or _checkpoint_root()
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "checkpoint_latest.pt").is_file()
    )


def find_latest_checkpoint(root: Path | None = None) -> Path:
    """Pick the most recently modified ``checkpoint_latest.pt`` under ``root``."""
    root = root or _checkpoint_root()
    candidates = [
        run_dir / "checkpoint_latest.pt"
        for run_dir in list_checkpoint_runs(root)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_latest.pt found under {root}. "
            "Train a model first or pass --checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_checkpoint(
    *,
    checkpoint: str | None,
    run: str | None,
    root: Path | None = None,
) -> Path:
    root = root or _checkpoint_root()
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if run:
        path = root / run / "checkpoint_latest.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    return find_latest_checkpoint(root)


def load_model_meta(ckpt_path: Path, ck: dict) -> dict:
    """Read model metadata from the checkpoint payload or sibling ``config.json``."""
    meta = ck.get("model_meta") or {}
    if meta.get("name") and meta.get("config"):
        return meta

    config_json = ckpt_path.parent / "config.json"
    if config_json.is_file():
        with open(config_json, encoding="utf-8") as f:
            saved = json.load(f)
        model_meta = saved.get("model") or {}
        if model_meta.get("name") and model_meta.get("config"):
            return model_meta

    raise ValueError(
        f"Checkpoint {ckpt_path} is missing model_meta and usable config.json"
    )


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, train_cfg: dict | None) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    dtype_name = (train_cfg or {}).get("dtype", "bf16")
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return torch.bfloat16


def load_model_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, int, dict | None]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_meta = load_model_meta(ckpt_path, ck)
    model = build_model(model_meta["name"], model_meta["config"])
    model.load_state_dict(ck["model"])
    model.eval()

    train_cfg = ck.get("train_config")
    dtype = resolve_dtype(device, train_cfg)
    model = model.to(device=device, dtype=dtype)
    return model, model_meta, int(ck.get("step", 0)), train_cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_tokens(
    model: torch.nn.Module,
    *,
    num_tokens: int,
    num_samples: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float | None,
    top_k: int | None,
) -> tuple[torch.Tensor, int]:
    set_seed(seed)
    # Only override model YAML sampling when the user explicitly passes values.
    # Important for ELF: yaml default temperature=0 (argmax); a hard-coded
    # CLI default of 1.0 would silently switch to multinomial sampling.
    sampling_cfg: dict[str, float | int | None] = {}
    if temperature is not None:
        sampling_cfg["temperature"] = temperature
    if top_k is not None:
        sampling_cfg["top_k"] = top_k
    with torch.no_grad():
        with torch.amp.autocast(
            device.type,
            dtype=dtype,
            enabled=device.type == "cuda",
        ):
            return model.generate(
                num_samples=num_samples,
                seqlen=num_tokens,
                sampling_cfg=sampling_cfg or None,
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text from a training checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Explicit checkpoint path (default: newest checkpoint_latest.pt)",
    )
    parser.add_argument(
        "--run",
        help="Run name under cache/checkpoints (uses its checkpoint_latest.pt)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=1024,
        help="Number of tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of independent samples to generate (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature; omit to use the model YAML default "
            "(ELF: 0=argmax; AR/BDELF: typically 1.0)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k filter; omit for model default / full-vocab multinomial",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--device",
        help="Torch device, e.g. cuda, cuda:0, cpu (default: cuda if available)",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List runs with checkpoint_latest.pt and exit",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.list_runs:
        runs = list_checkpoint_runs()
        if not runs:
            _log(f"No checkpoints under {_checkpoint_root()}")
            return
        for run_dir in runs:
            ckpt = run_dir / "checkpoint_latest.pt"
            mtime = ckpt.stat().st_mtime
            _log(f"{run_dir.name}\tstep=? mtime={mtime:.0f}\t{ckpt}")
        return

    ckpt_path = resolve_checkpoint(checkpoint=args.checkpoint, run=args.run)
    device = resolve_device(args.device)

    _log(f"Loading checkpoint: {ckpt_path}")
    model, model_meta, step, train_cfg = load_model_from_checkpoint(ckpt_path, device)
    dtype = resolve_dtype(device, train_cfg)

    tokenizer_name = model_meta["config"].get("tokenizer")
    if not tokenizer_name:
        raise ValueError("Model config is missing tokenizer name")
    tokenizer = get_tokenizer(tokenizer_name)

    _log(
        f"Model={model_meta['name']}, step={step}, "
        f"device={device}, dtype={dtype}, num_tokens={args.num_tokens}, "
        f"temperature={args.temperature if args.temperature is not None else 'yaml'}, "
        f"top_k={args.top_k if args.top_k is not None else 'yaml'}, seed={args.seed}",
    )

    tokens, nfe = generate_tokens(
        model,
        num_tokens=args.num_tokens,
        num_samples=args.num_samples,
        seed=args.seed,
        device=device,
        dtype=dtype,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    _log(f"Generation finished (nfe={nfe})")
    for sample_idx in range(tokens.size(0)):
        if args.num_samples > 1:
            _log(f"--- sample {sample_idx + 1}/{args.num_samples} ---")
        text = tokenizer.decode(tokens[sample_idx].tolist(), skip_special_tokens=False)
        print(text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("Interrupt received; exiting.")
    except Exception as exc:
        _log(f"Error: {exc}", file=sys.stderr)
        raise
