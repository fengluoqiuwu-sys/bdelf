#!/usr/bin/env python3
"""Generate text from the latest training checkpoint.

Usage:
    python generate.py
    python generate.py --run full/lm/elf/<hash>
    python generate.py --checkpoint cache/checkpoints/full/lm/elf/<hash>/checkpoint_latest.pt
    python generate.py --latent-model latent_vae --tag 100m-b32-d1
    python generate.py --num-tokens 1024 --seed 42
    python generate.py --prompt "Once upon a time" --num-tokens 256
    python generate.py --prompt-file prompt.txt --run full/ar/<hash>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import hf_config  # noqa: F401
from models import build_model
from models.latent.artifact_loader import load_latent_artifact
from tokenizer import get_token_layout, get_tokenizer
from train import CHECKPOINT_ROOT
from train.run_path import TRAIN_VARIANTS

_GENERATE_LOG = "[generate]"


def _log(msg: str, *, file=None) -> None:
    if file is None:
        file = sys.stdout
    print(f"{_GENERATE_LOG} {msg}", file=file, flush=True)


def _checkpoint_root() -> Path:
    return Path(CHECKPOINT_ROOT)


def list_checkpoint_runs(root: Path | None = None) -> list[Path]:
    """列出含 ``checkpoint_latest.pt`` 的 run 目录。

    支持 ``{variant}/{model}/{hash}``（旧）与 ``{variant}/{kind}/{model}/{hash}``（新）。
    """
    root = root or _checkpoint_root()
    if not root.is_dir():
        return []
    runs: list[Path] = []
    for variant_dir in sorted(root.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name not in TRAIN_VARIANTS:
            continue
        for child in sorted(variant_dir.iterdir()):
            if not child.is_dir() or child.name == "artifacts":
                continue
            if child.name in ("lm", "latent"):
                for model_dir in sorted(child.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    for hash_dir in sorted(model_dir.iterdir()):
                        if hash_dir.is_dir() and (hash_dir / "checkpoint_latest.pt").is_file():
                            runs.append(hash_dir)
                continue
            # legacy: variant/model/hash
            model_dir = child
            for hash_dir in sorted(model_dir.iterdir()):
                if hash_dir.is_dir() and (hash_dir / "checkpoint_latest.pt").is_file():
                    runs.append(hash_dir)
    return runs


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
        # 相对 checkpoint_root：新 full/lm/elf/<hash> 或旧 full/elf/<hash>
        path = root / run / "checkpoint_latest.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Use --run {fast|mid|full}/{kind}/{model}/{config-hash} "
                "or legacy {fast|mid|full}/{model}/{config-hash} "
                "(see scripts/resolve_checkpoint.py)."
            )
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
) -> tuple[torch.nn.Module, dict, int, dict | None, bool]:
    """加载 checkpoint；有 EMA 则拷进模型（与 ``eval.py`` 默认口径一致）。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_meta = load_model_meta(ckpt_path, ck)
    model_cfg = dict(model_meta["config"] or {})
    # Cola Stage-2 checkpoints already contain VAE weights; skip re-loading
    # cola_vae run artifacts (which may be absent on generate-only machines).
    if model_meta["name"] == "cola":
        model_cfg["load_vae_weights"] = False
    model = build_model(model_meta["name"], model_cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    train_cfg = ck.get("train_config")
    step = int(ck.get("step", 0))
    ema_raw = ck.get("ema")
    del ck
    dtype = resolve_dtype(device, train_cfg)
    model = model.to(device=device, dtype=dtype)
    used_ema = False
    if isinstance(ema_raw, dict) and ema_raw:
        from train.ema import apply_ema_weights

        used_ema = apply_ema_weights(model, ema_raw)
    return model, model_meta, step, train_cfg, used_ema


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_prompt_text(prompt: str | None, prompt_file: str | None) -> str | None:
    if prompt is not None and prompt_file is not None:
        raise ValueError("Pass only one of --prompt / --prompt-file")
    if prompt_file is not None:
        path = Path(prompt_file)
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")
    if prompt is not None:
        return prompt
    return None


def encode_prefix_tokens(
    prompt: str,
    *,
    tokenizer,
    tokenizer_name: str,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode prompt as ``(num_samples, L)`` prefix, prepending BOS like training."""
    layout = get_token_layout(tokenizer_name)
    ids = tokenizer.encode_preprocess(prompt)
    prefix_ids = [layout.bos_token_id, *ids]
    return (
        torch.tensor(prefix_ids, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(num_samples, -1)
        .contiguous()
    )


def model_supports_prefix(model: torch.nn.Module) -> bool:
    """Whether ``--prompt`` / prefix completion is allowed for this model."""
    return bool(getattr(model, "supports_prefix", True))


def generate_tokens(
    model: torch.nn.Module,
    *,
    num_tokens: int,
    num_samples: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    sampling_cfg: dict | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    prefix_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    set_seed(seed)
    # 以 generate 配置为底，CLI 显式传入的 temperature/top_k 再覆盖。
    merged: dict = dict(sampling_cfg or {})
    if temperature is not None:
        merged["temperature"] = temperature
    if top_k is not None:
        merged["top_k"] = top_k
    gen_kwargs: dict = dict(
        num_samples=num_samples,
        seqlen=num_tokens,
        sampling_cfg=merged or None,
    )
    if prefix_tokens is not None:
        if not model_supports_prefix(model):
            raise ValueError(
                "This model does not support prompt completion "
                "(generate lacks prefix_tokens)."
            )
        gen_kwargs["prefix_tokens"] = prefix_tokens
    with torch.no_grad():
        with torch.amp.autocast(
            device.type,
            dtype=dtype,
            enabled=device.type == "cuda",
        ):
            return model.generate(**gen_kwargs)


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
        help=(
            "Run under cache/checkpoints: {fast|mid|full}/{model}/{config-hash} "
            "(see scripts/resolve_checkpoint.py)"
        ),
    )
    parser.add_argument(
        "--latent-model",
        help=(
            "只读加载 artifacts/latent/<name>/<tag>/ "
            "（须与 --tag 同时给出；与 --run / --checkpoint 互斥）"
        ),
    )
    parser.add_argument(
        "--tag",
        help="artifacts 选用 tag（须与 --latent-model 同时给出）",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=1024,
        help=(
            "Total sequence length including the prompt prefix "
            "(default: 1024)"
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of independent samples to generate (default: 1)",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt text to continue (BOS is prepended automatically)",
    )
    parser.add_argument(
        "--prompt-file",
        help="Read prompt text from a UTF-8 file",
    )
    parser.add_argument(
        "--generate",
        default="generate",
        help=(
            "Generate config under config/generate/<model>/ "
            "(default: generate; train online eval uses eval)"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Override sampling temperature; omit to use --generate config "
            "(ELF: 0=argmax; AR: typically 1.0)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k; omit to use --generate config / full-vocab",
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
            rel = run_dir.relative_to(_checkpoint_root())
            _log(f"{rel}\tmtime={mtime:.0f}\t{ckpt}")
        return

    device = resolve_device(args.device)
    used_artifact = bool(args.latent_model or args.tag)
    if used_artifact:
        if not args.latent_model or not args.tag:
            raise ValueError("--latent-model 与 --tag 必须同时指定")
        if args.checkpoint or args.run:
            raise ValueError("--latent-model/--tag 与 --checkpoint / --run 互斥")
        loaded = load_latent_artifact(
            args.latent_model, args.tag, device=device
        )
        ckpt_path = loaded.ckpt_path
        _log(
            f"Loading artifacts: latent_model={loaded.latent_model} "
            f"tag={loaded.tag} path={ckpt_path}"
        )
        model = loaded.model
        model_meta = loaded.model_meta
        step = loaded.step
        train_cfg = loaded.train_cfg
        used_ema = loaded.used_ema
    else:
        ckpt_path = resolve_checkpoint(checkpoint=args.checkpoint, run=args.run)
        _log(f"Loading checkpoint: {ckpt_path}")
        model, model_meta, step, train_cfg, used_ema = load_model_from_checkpoint(
            ckpt_path, device
        )
    dtype = resolve_dtype(device, train_cfg)

    tokenizer_name = model_meta["config"].get("tokenizer")
    if not tokenizer_name:
        raise ValueError("Model config is missing tokenizer name")
    tokenizer = get_tokenizer(tokenizer_name)

    if getattr(model, "ace_attachable", False):
        from models.lm.elf.ace import attach_ace_identity, model_hash_from_checkpoint

        ace_hash = model_hash_from_checkpoint(ckpt_path)
        if ace_hash:
            attach_ace_identity(
                model, model_hash=ace_hash, step=step, tokenizer=tokenizer_name,
            )

    prompt_text = resolve_prompt_text(args.prompt, args.prompt_file)
    prefix_tokens = None
    prefix_len = 0
    if prompt_text is not None:
        if not model_supports_prefix(model):
            raise ValueError(
                f"{model_meta['name']} generation is unconditional; "
                "--prompt is not supported"
            )
        prefix_tokens = encode_prefix_tokens(
            prompt_text,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            num_samples=args.num_samples,
            device=device,
        )
        prefix_len = int(prefix_tokens.size(1))
        if prefix_len >= args.num_tokens:
            raise ValueError(
                f"Prompt encodes to {prefix_len} tokens (with BOS), which must be "
                f"shorter than --num-tokens={args.num_tokens}"
            )

    from train.generate_config import get_generate

    gen_cfg = get_generate(model_meta["name"], args.generate)
    sampling_cfg = gen_cfg.to_sampling_cfg()

    _log(
        f"Model={model_meta['name']}, step={step}, "
        f"device={device}, dtype={dtype}, num_tokens={args.num_tokens}, "
        f"prefix_len={prefix_len}, generate={args.generate}, "
        f"temperature="
        f"{args.temperature if args.temperature is not None else sampling_cfg.get('temperature', 'yaml')}, "
        f"top_k="
        f"{args.top_k if args.top_k is not None else sampling_cfg.get('top_k', 'yaml')}, "
        f"seed={args.seed}, ema={'yes' if used_ema else 'no'}",
    )

    tokens, nfe = generate_tokens(
        model,
        num_tokens=args.num_tokens,
        num_samples=args.num_samples,
        seed=args.seed,
        device=device,
        dtype=dtype,
        sampling_cfg=sampling_cfg,
        temperature=args.temperature,
        top_k=args.top_k,
        prefix_tokens=prefix_tokens,
    )

    _log(f"Generation finished (nfe={nfe})")
    for sample_idx in range(tokens.size(0)):
        if args.num_samples > 1:
            _log(f"--- sample {sample_idx + 1}/{args.num_samples} ---")
        if prefix_len > 0:
            prompt_decoded = tokenizer.decode(
                tokens[sample_idx, :prefix_len].tolist(),
                skip_special_tokens=True,
            )
            completion = tokenizer.decode(
                tokens[sample_idx, prefix_len:].tolist(),
                skip_special_tokens=True,
            )
            print(f"[prompt] {prompt_decoded}")
            print(f"[completion] {completion}")
        else:
            text = tokenizer.decode(
                tokens[sample_idx].tolist(), skip_special_tokens=True,
            )
            print(text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("Interrupt received; exiting.")
    except Exception as exc:
        _log(f"Error: {exc}", file=sys.stderr)
        raise
