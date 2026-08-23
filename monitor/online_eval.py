"""在线 eval 详情（LM 样本 / latent VAE probe）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from monitor.samples_parse import load_online_lm_samples


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _eval_log_row(run_dir: Path, step: int) -> dict[str, Any]:
    path = run_dir / "eval_log.csv"
    if not path.is_file():
        return {}
    best: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                row_step = int(float(row.get("step") or -1))
            except ValueError:
                continue
            if row_step == step:
                best = {k: v for k, v in row.items()}
    return best


def _eval_official_row(run_dir: Path, step: int) -> dict[str, Any]:
    path = run_dir / "eval_components" / "official.csv"
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                if int(float(row.get("step") or -1)) == step:
                    return dict(row)
            except ValueError:
                continue
    return {}


def _probe_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "eval_samples" / f"step_{step:07d}" / "vae_probe"


def _image_url(run_dir: Path, png: Path, media_prefix: str) -> str:
    rel = png.relative_to(run_dir)
    return f"{media_prefix}/{rel.as_posix()}"


def _latent_item_summaries(run_dir: Path, step: int) -> list[dict[str, Any]]:
    probe_dir = _probe_dir(run_dir, step)
    metrics = _read_json(probe_dir / "probe_metrics.json")
    items: list[dict[str, Any]] = []
    for i, row in enumerate(metrics.get("per_sample") or []):
        if not isinstance(row, dict):
            continue
        items.append({"id": str(i), **row})
    if (probe_dir / "pooled_latent_dist.png").is_file():
        pooled = dict(metrics.get("pooled") or {})
        items.append({"id": "pooled", **pooled})
    return items


def _lm_item_summaries(sample_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in load_online_lm_samples(sample_dir):
        sid = row.get("id", "")
        items.append(
            {
                "id": str(sid),
                "gen_ppl": row.get("gen_ppl"),
                "entropy": row.get("entropy"),
            },
        )
    return items


def load_online_eval_detail(
    run_dir: Path,
    step: int,
    *,
    kind: str,
    media_prefix: str,
) -> dict[str, Any]:
    del media_prefix
    sample_dir = run_dir / "eval_samples" / f"step_{step:07d}"
    meta = _read_json(sample_dir / "meta.json")
    eval_row = _eval_log_row(run_dir, step)
    official = _eval_official_row(run_dir, step)
    latent = kind == "latent"
    items = (
        _latent_item_summaries(run_dir, step)
        if latent
        else _lm_item_summaries(sample_dir)
    )
    return {
        "step": step,
        "kind": kind,
        "view": "latent_probe" if latent else "lm_samples",
        "params": {
            "meta": meta,
            "eval_log": eval_row,
            "eval_official": official,
        },
        "items": items,
    }


def load_online_eval_item(
    run_dir: Path,
    step: int,
    item_id: str,
    *,
    kind: str,
    media_prefix: str,
) -> dict[str, Any] | None:
    sample_dir = run_dir / "eval_samples" / f"step_{step:07d}"
    if kind == "latent":
        probe_dir = _probe_dir(run_dir, step)
        if not probe_dir.is_dir():
            return None
        metrics = _read_json(probe_dir / "probe_metrics.json")
        images: list[dict[str, str]] = []
        row: dict[str, Any] = {}
        if item_id == "pooled":
            row = dict(metrics.get("pooled") or {})
            png = probe_dir / "pooled_latent_dist.png"
            if png.is_file():
                images.append(
                    {"name": png.name, "url": _image_url(run_dir, png, media_prefix)},
                )
        else:
            try:
                idx = int(item_id)
            except ValueError:
                return None
            per = metrics.get("per_sample") or []
            if 0 <= idx < len(per) and isinstance(per[idx], dict):
                row = dict(per[idx])
            prefix = f"{idx:02d}_"
            for png in sorted(probe_dir.glob(f"{prefix}*.png")):
                images.append(
                    {"name": png.name, "url": _image_url(run_dir, png, media_prefix)},
                )
            if not images and not row:
                return None
        return {
            "step": step,
            "id": item_id,
            "kind": kind,
            "view": "latent_probe",
            "metrics": row,
            "images": images,
        }

    samples = load_online_lm_samples(sample_dir)
    found: dict[str, Any] | None = None
    for row in samples:
        if str(row.get("id", "")) == str(item_id):
            found = row
            break
    if found is None:
        try:
            idx = int(item_id)
        except ValueError:
            return None
        if 0 <= idx < len(samples):
            found = samples[idx]
    if found is None:
        return None
    return {
        "step": step,
        "id": str(found.get("id", item_id)),
        "kind": kind,
        "view": "lm_samples",
        "sample": found,
    }
