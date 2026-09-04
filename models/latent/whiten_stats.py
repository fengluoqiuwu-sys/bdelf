"""Artifact 导出后离线估计逐维 μ 白化统计（不进 VAE 训练）。

在 ``artifacts/latent/{model}/{tag}/`` 上冻住 encoder，扫预处理缓存的一小段
train split，把 ``whitening_mean`` / ``whitening_std`` 写入 checkpoint。
默认只吃 ``DEFAULT_WHITEN_TOKENS`` 个有效 token，不跑完全集。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.latent.artifact_loader import (
    WHITEN_JSON,
    _LATEST,
    load_latent_artifact,
    pop_whiten_state,
    resolve_artifact_dir,
)

# 2^20 有效 token：约 2048 条 512 段，逐维 mean/std 已稳；远小于 OWT 全集。
DEFAULT_WHITEN_TOKENS = 1_048_576
DEFAULT_WHITEN_DATASET = "owt"
DEFAULT_WHITEN_PREPROCESS = "owt-seg512"
DEFAULT_WHITEN_SPLIT = "train"
DEFAULT_WHITEN_BATCH = 16
DEFAULT_WHITEN_SEED = 42
_WHITEN_STD_MIN = 1e-8


@dataclass(frozen=True)
class WhitenStats:
    """一次离线估计的逐维 μ 仿射。"""

    mean: torch.Tensor
    std: torch.Tensor
    n_valid: int
    n_seq: int
    dataset: str
    preprocess: str
    split: str
    latent_dim: int

    def as_meta(self) -> dict[str, Any]:
        std = self.std.detach().float().cpu()
        mean = self.mean.detach().float().cpu()
        std_min = float(std.min().item())
        std_max = float(std.max().item())
        return {
            "on": "mu",
            "n_valid": int(self.n_valid),
            "n_seq": int(self.n_seq),
            "n_tokens_budget": int(DEFAULT_WHITEN_TOKENS),
            "dataset": self.dataset,
            "preprocess": self.preprocess,
            "split": self.split,
            "latent_dim": int(self.latent_dim),
            "std_min": std_min,
            "std_max": std_max,
            "std_ratio": (std_max / max(std_min, _WHITEN_STD_MIN)),
            "mean": [float(x) for x in mean.tolist()],
            "std": [float(x) for x in std.tolist()],
        }


def _encode_module(model: torch.nn.Module) -> torch.nn.Module:
    bb = getattr(model, "backbone", model)
    if not callable(getattr(bb, "encode", None)):
        raise TypeError(f"{type(model).__name__} 无 encode，无法估计白化统计")
    return bb


def _pad_id(mod: torch.nn.Module) -> int | None:
    layout = getattr(mod, "token_layout", None)
    pad = getattr(layout, "pad_token_id", None)
    return int(pad) if pad is not None else None


def _require_cached_preprocessed(dataset: str, preprocess: str):
    """只读已完成的预处理缓存；缺缓存则报错，绝不在此切分全文。"""
    from dataset import get_dataset
    from preprocess.preprocess import (
        _cache_dir,
        _fingerprint,
        _load_manifest,
        get_preprocess,
        get_preprocessed,
    )

    source = get_dataset(dataset)
    config = get_preprocess(preprocess)
    cache_dir = _cache_dir(config, source)
    manifest = _load_manifest(cache_dir)
    if (
        not manifest
        or manifest.get("status") != "complete"
        or manifest.get("fingerprint") != _fingerprint(config, source)
    ):
        raise FileNotFoundError(
            f"预处理缓存不存在或未完成: {cache_dir}。"
            f"请先有 dataset={dataset!r} preprocess={preprocess!r} 的完整缓存，"
            "本步骤不从头切分全集。"
        )
    return get_preprocessed(preprocess, dataset)


def _merge_welford(
    mean: torch.Tensor,
    m2: torch.Tensor,
    n: int,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """把 ``[k, d]`` 并入总体 Welford（float64）。"""
    k = int(batch.size(0))
    if k == 0:
        return mean, m2, n
    batch_mean = batch.mean(dim=0)
    batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
    if n == 0:
        return batch_mean, batch_m2, k
    delta = batch_mean - mean
    n_tot = n + k
    mean_new = mean + delta * (k / n_tot)
    m2_new = m2 + batch_m2 + (delta * delta) * (n * k / n_tot)
    return mean_new, m2_new, n_tot


@torch.no_grad()
def estimate_mu_whiten(
    model: torch.nn.Module,
    *,
    dataset: str = DEFAULT_WHITEN_DATASET,
    preprocess: str = DEFAULT_WHITEN_PREPROCESS,
    split: str = DEFAULT_WHITEN_SPLIT,
    max_tokens: int = DEFAULT_WHITEN_TOKENS,
    batch_size: int = DEFAULT_WHITEN_BATCH,
    seed: int = DEFAULT_WHITEN_SEED,
    device: torch.device | str | None = None,
) -> WhitenStats:
    """冻 encoder、``sample=False``，对有效 token 的 μ 做逐维 mean/std。"""
    if max_tokens < 1:
        raise ValueError(f"max_tokens 须为正，收到 {max_tokens}")
    if batch_size < 1:
        raise ValueError(f"batch_size 须为正，收到 {batch_size}")

    if device is None:
        torch_device = next(model.parameters()).device
    else:
        torch_device = torch.device(device)
    # 块因果 SDPA 的 additive bias 在 bf16 下会与 query dtype 不一致；统计用 fp32。
    model = model.to(device=torch_device, dtype=torch.float32)
    model.eval()
    enc = _encode_module(model)
    pad_id = _pad_id(enc)
    pre = _require_cached_preprocessed(dataset, preprocess)
    view = pre.load_split(split)
    if len(view) < 1:
        raise RuntimeError(f"split={split!r} 为空，无法估计白化")

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    loader = DataLoader(
        view,
        batch_size=int(batch_size),
        shuffle=True,
        generator=gen,
        num_workers=0,
        drop_last=False,
    )

    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None
    n_valid = 0
    n_seq = 0
    dim: int | None = None
    pbar = tqdm(loader, desc="whiten-mu", leave=False)
    for batch in pbar:
        tokens = batch["input_ids"].to(device=torch_device, dtype=torch.long)
        _, mu, _ = enc.encode(tokens, sample=False)
        mu64 = mu.detach().to(dtype=torch.float64)
        if pad_id is None:
            valid = torch.ones(tokens.shape, device=tokens.device, dtype=torch.bool)
        else:
            valid = tokens != pad_id
        flat = mu64[valid]
        if dim is None:
            dim = int(mu64.size(-1))
            mean = torch.zeros(dim, dtype=torch.float64)
            m2 = torch.zeros(dim, dtype=torch.float64)
        if int(flat.size(0)) == 0:
            continue
        mean, m2, n_valid = _merge_welford(mean, m2, n_valid, flat)
        n_seq += int(tokens.size(0))
        pbar.set_postfix(n_valid=n_valid)
        if n_valid >= int(max_tokens):
            break

    if dim is None or mean is None or m2 is None or n_valid < 2:
        raise RuntimeError("有效 token 不足，无法估计逐维 std")
    var = m2 / float(n_valid)
    std = torch.sqrt(var.clamp(min=0.0)).clamp(min=_WHITEN_STD_MIN)
    return WhitenStats(
        mean=mean.to(dtype=torch.float32).cpu(),
        std=std.to(dtype=torch.float32).cpu(),
        n_valid=int(n_valid),
        n_seq=int(n_seq),
        dataset=str(dataset),
        preprocess=str(preprocess),
        split=str(split),
        latent_dim=int(dim),
    )


def _state_key_prefix(weights: dict[str, Any]) -> str:
    if any(str(k).startswith("backbone.") for k in weights):
        return "backbone."
    return ""


def _write_whiten_sidecar(dest: Path, stats: WhitenStats) -> None:
    meta = stats.as_meta()
    (dest / WHITEN_JSON).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cfg_path = dest / "config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            cfg = {}
    cfg["whiten"] = {
        "on": "mu",
        "n_valid": stats.n_valid,
        "dataset": stats.dataset,
        "preprocess": stats.preprocess,
        "split": stats.split,
        "latent_dim": stats.latent_dim,
        "std_min": meta["std_min"],
        "std_max": meta["std_max"],
        "std_ratio": meta["std_ratio"],
    }
    cfg_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_artifact_whiten(
    latent_model: str,
    tag: str,
    *,
    dataset: str = DEFAULT_WHITEN_DATASET,
    preprocess: str = DEFAULT_WHITEN_PREPROCESS,
    split: str = DEFAULT_WHITEN_SPLIT,
    max_tokens: int = DEFAULT_WHITEN_TOKENS,
    batch_size: int = DEFAULT_WHITEN_BATCH,
    seed: int = DEFAULT_WHITEN_SEED,
    device: str | None = None,
    force: bool = False,
    checkpoint_root: str | Path | None = None,
) -> WhitenStats:
    """估计 μ 白化并写入已有 artifact（加载器仍然只读）。"""
    dest = resolve_artifact_dir(
        latent_model, tag, checkpoint_root=checkpoint_root
    )
    ckpt_path = dest / _LATEST
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"artifacts checkpoint 不存在: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = ck.get("model")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{ckpt_path}: checkpoint 无 model 权重")
    existing = pop_whiten_state(dict(weights))
    if (
        not force
        and "whitening_mean" in existing
        and "whitening_std" in existing
    ):
        mean = existing["whitening_mean"].detach().float().reshape(-1).cpu()
        std = existing["whitening_std"].detach().float().reshape(-1).cpu()
        n_valid = 0
        dataset_s, preprocess_s, split_s = dataset, preprocess, split
        meta = ck.get("whiten_meta")
        if isinstance(meta, dict):
            n_valid = int(meta.get("n_valid") or 0)
            dataset_s = str(meta.get("dataset") or dataset_s)
            preprocess_s = str(meta.get("preprocess") or preprocess_s)
            split_s = str(meta.get("split") or split_s)
        sidecar = dest / WHITEN_JSON
        if sidecar.is_file():
            try:
                prev = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prev = {}
            if isinstance(prev, dict):
                n_valid = int(prev.get("n_valid") or 0)
                dataset_s = str(prev.get("dataset") or dataset)
                preprocess_s = str(prev.get("preprocess") or preprocess)
                split_s = str(prev.get("split") or split)
        return WhitenStats(
            mean=mean,
            std=std,
            n_valid=n_valid,
            n_seq=0,
            dataset=dataset_s,
            preprocess=preprocess_s,
            split=split_s,
            latent_dim=int(mean.numel()),
        )

    if device:
        torch_device: torch.device | str | None = device
    elif torch.cuda.is_available():
        torch_device = "cuda"
    else:
        torch_device = "cpu"
    loaded = load_latent_artifact(
        latent_model,
        tag,
        device=torch_device,
        apply_ema=True,
        checkpoint_root=checkpoint_root,
    )
    stats = estimate_mu_whiten(
        loaded.model,
        dataset=dataset,
        preprocess=preprocess,
        split=split,
        max_tokens=max_tokens,
        batch_size=batch_size,
        seed=seed,
        device=torch_device,
    )
    del loaded

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = ck.get("model")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{ckpt_path}: checkpoint 无 model 权重")
    pop_whiten_state(weights)
    prefix = _state_key_prefix(weights)
    weights[prefix + "whitening_mean"] = stats.mean.contiguous()
    weights[prefix + "whitening_std"] = stats.std.contiguous()
    ck["model"] = weights
    ck["whiten_meta"] = {
        k: v for k, v in stats.as_meta().items() if k not in ("mean", "std")
    }
    from models.latent.artifact_export import _atomic_torch_save

    _atomic_torch_save(ck, ckpt_path)
    _write_whiten_sidecar(dest, stats)
    return stats
