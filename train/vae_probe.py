"""latent VAE 在线 probe：Cola 图 4 前三张子图 + 汇总标量。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from train.checkpoint import unwrap_model
from train.latent_eval import _subsample_indices
from train.metrics import _train_log
from train.run_logs import sample_step_dir


def _latent_encode_module(model: nn.Module) -> nn.Module:
    raw = unwrap_model(model)
    if hasattr(raw, "encode") and callable(raw.encode):
        return raw
    backbone = getattr(raw, "backbone", None)
    if backbone is not None and hasattr(backbone, "encode") and callable(backbone.encode):
        return backbone
    raise AttributeError(f"{type(raw).__name__} has no encode()")


def model_supports_vae_probe(model: nn.Module) -> bool:
    try:
        _latent_encode_module(model)
    except AttributeError:
        return False
    return True


def _effective_length(tokens: torch.Tensor, pad_token_id: int) -> int:
    if tokens.numel() == 0:
        return 0
    non_pad = (tokens != pad_token_id).nonzero(as_tuple=False)
    if non_pad.numel() == 0:
        return int(tokens.numel())
    return int(non_pad[-1].item()) + 1


def select_probe_samples(
    pool: Dataset,
    count: int,
    seed: int,
) -> list[torch.Tensor]:
    if count <= 0 or len(pool) == 0:
        return []
    n = min(count, len(pool))
    indices = _subsample_indices(
        np.arange(len(pool), dtype=np.int64),
        n,
        seed,
    )
    out: list[torch.Tensor] = []
    for idx in indices:
        item = pool[int(idx)]
        if isinstance(item, tuple):
            item = item[0]
        if isinstance(item, dict):
            tokens = item["input_ids"]
        else:
            tokens = item
        out.append(tokens.detach().cpu().long())
    return out


@torch.compiler.disable
@torch.no_grad()
def _encode_mu(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    raw = _latent_encode_module(model)
    batch = tokens.unsqueeze(0).to(device, non_blocking=True)
    use_amp = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        z, _, _ = raw.encode(batch, sample=False)
    return z[0].float().cpu()


def _offdiag_cosine_mean(z: np.ndarray) -> float:
    if z.shape[0] < 2:
        return float("nan")
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    unit = z / norms
    sim = unit @ unit.T
    mask = ~np.eye(sim.shape[0], dtype=bool)
    return float(sim[mask].mean())


def _plot_latent_matrix(z: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-2.0, vmax=2.0)
    ax.set_xlabel("Latent Position")
    ax.set_ylabel("Latent Dimension")
    ax.set_title("VAE Latent Matrix")
    fig.colorbar(im, ax=ax, label="Scaled Latent Value")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_self_similarity(z: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    unit = z / norms
    sim = unit @ unit.T
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xlabel("Latent Position")
    ax.set_ylabel("Latent Position")
    ax.set_title("Token Self-Similarity")
    fig.colorbar(im, ax=ax, label="Cosine Similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_latent_distribution(values: np.ndarray, path: Path, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=60, density=True, histtype="step", linewidth=1.5, color="C0")
    xs = np.linspace(-3.0, 3.0, 200)
    std_pdf = np.exp(-0.5 * xs**2) / np.sqrt(2.0 * np.pi)
    ax.plot(xs, std_pdf, "r--", linewidth=1.5, label="Std Normal")
    ax.set_xlabel("Latent Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_vae_probe_dir(
    run_dir: Path,
    step: int,
    model: nn.Module,
    samples: list[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int,
    meta_extra: dict[str, Any] | None = None,
) -> Path | None:
    """落盘 ``eval_samples/step_*/vae_probe/``。"""
    if not samples or not model_supports_vae_probe(model):
        return None

    was_training = model.training
    model.eval()
    step_dir = sample_step_dir(run_dir, step)
    out_dir = step_dir / "vae_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample_metrics: list[dict[str, float]] = []
    pooled_values: list[np.ndarray] = []

    try:
        for i, tokens in enumerate(samples):
            eff = _effective_length(tokens, pad_token_id)
            if eff <= 0:
                continue
            z_t = _encode_mu(model, tokens[:eff], device, amp_dtype)
            z = z_t.numpy()
            pooled_values.append(z.reshape(-1))

            _plot_latent_matrix(z, out_dir / f"{i:02d}_latent_matrix.png")
            _plot_self_similarity(z, out_dir / f"{i:02d}_self_similarity.png")
            _plot_latent_distribution(
                z.reshape(-1),
                out_dir / f"{i:02d}_latent_dist.png",
                title="VAE Latent Distribution",
            )
            per_sample_metrics.append({
                "offdiag_cosine_mean": _offdiag_cosine_mean(z),
                "z_mean": float(z.mean()),
                "z_std": float(z.std()),
                "length": float(eff),
            })

        if pooled_values:
            pooled = np.concatenate(pooled_values)
            _plot_latent_distribution(
                pooled,
                out_dir / "pooled_latent_dist.png",
                title="VAE Latent Distribution (pooled)",
            )
            probe_metrics = {
                "step": step,
                "n": len(per_sample_metrics),
                "seed": seed,
                "per_sample": per_sample_metrics,
                "pooled": {
                    "z_mean": float(pooled.mean()),
                    "z_std": float(pooled.std()),
                },
            }
            (out_dir / "probe_metrics.json").write_text(
                json.dumps(probe_metrics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        meta = {
            "step": step,
            "probe_n": len(per_sample_metrics),
            "probe_seed": seed,
            **(meta_extra or {}),
        }
        step_meta_path = step_dir / "meta.json"
        if step_meta_path.exists():
            try:
                existing = json.loads(step_meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
            existing.update(meta)
            meta = existing
        step_meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return out_dir
    finally:
        if was_training:
            model.train()


def maybe_write_vae_probe(
    run_dir: Path,
    step: int,
    model: nn.Module,
    probe_pool: Dataset | None,
    *,
    pad_token_id: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    probe_samples: int,
    probe_seed: int,
    meta_extra: dict[str, Any] | None = None,
    log: bool = True,
) -> None:
    if probe_samples <= 0 or probe_pool is None or len(probe_pool) == 0:
        return
    if not model_supports_vae_probe(model):
        return
    try:
        samples = select_probe_samples(probe_pool, probe_samples, probe_seed)
        out = write_vae_probe_dir(
            run_dir,
            step,
            model,
            samples,
            pad_token_id=pad_token_id,
            device=device,
            amp_dtype=amp_dtype,
            seed=probe_seed,
            meta_extra=meta_extra,
        )
        if log and out is not None:
            _train_log(f"eval/vae_probe: wrote {out} ({len(samples)} samples)")
    except Exception as exc:  # noqa: BLE001 — probe 失败不中断训练
        if log:
            _train_log(f"eval/vae_probe failed (skip): {exc}")
