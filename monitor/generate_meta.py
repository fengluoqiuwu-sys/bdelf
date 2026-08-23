"""Generate 页只读元数据：checkpoint 列表、默认采样、显存门槛（不 import torch）。"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from monitor.runs import resolve_run_dir

CONFIG_GENERATE = Path(__file__).resolve().parents[1] / "config" / "generate"

# 与 backbone.supports_prefix=False 对齐；缺省视为可续写。
NO_PREFIX_MODELS = frozenset({
    "elf",
    "loopsc",
    "lexce",
    "odar",
    "trace",
    "late_ce",
    "denoiser_chart",
    "jac_ellipsoid",
})

VRAM_USED_LIMIT_GIB = 6.0
_STEP_RE = re.compile(r"checkpoint_step_(\d+)\.pt$", re.I)
_STAGE_STEP_RE = re.compile(r"^(s\d+)-checkpoint_step_(\d+)\.pt$", re.I)


def model_supports_prefix(model: str) -> bool:
    return str(model or "") not in NO_PREFIX_MODELS


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_model_name(run_dir: Path) -> str:
    cfg = _read_json(run_dir / "config.json")
    model_meta = cfg.get("model") or {}
    name = str(model_meta.get("name") or "")
    if name:
        return name
    train = cfg.get("train") or {}
    return str(train.get("model") or run_dir.parent.name)


def _file_info(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "file": path.name,
        "size": st.st_size,
        "mtime": st.st_mtime,
    }


def _step_from_name(name: str) -> int | None:
    m = _STAGE_STEP_RE.match(name)
    if m:
        return int(m.group(2))
    m2 = _STEP_RE.search(name)
    if m2:
        return int(m2.group(1))
    return None


def list_run_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    latest = run_dir / "checkpoint_latest.pt"
    if latest.is_file():
        info = _file_info(latest)
        info.update({"id": "latest", "name": "latest", "step": None})
        items.append(info)
    snaps: list[Path] = []
    snaps.extend(run_dir.glob("checkpoint_step_*.pt"))
    snaps.extend(run_dir.glob("s*-checkpoint_step_*.pt"))
    seen = {latest.resolve()} if latest.is_file() else set()
    for path in snaps:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        step = _step_from_name(path.name)
        info = _file_info(path)
        info.update({
            "id": path.name,
            "name": f"step {step}" if step is not None else path.name,
            "step": step,
        })
        items.append(info)
    items.sort(key=lambda x: (0 if x["id"] == "latest" else 1, -(x.get("step") or 0), x["file"]))
    return items


def resolve_ckpt_file(run_dir: Path, ckpt_id: str) -> Path | None:
    raw = str(ckpt_id or "latest").strip()
    if not raw or raw == "latest":
        path = run_dir / "checkpoint_latest.pt"
        return path if path.is_file() else None
    direct = run_dir / raw
    if direct.is_file() and direct.suffix == ".pt":
        return direct
    if raw.isdigit():
        padded = f"checkpoint_step_{int(raw):07d}.pt"
        path = run_dir / padded
        if path.is_file():
            return path
        matches = list(run_dir.glob(f"*checkpoint_step_{int(raw):07d}.pt"))
        matches.extend(run_dir.glob(f"*checkpoint_step_{int(raw)}.pt"))
        if matches:
            return matches[0]
    return None


def resolve_generate_dir(model: str) -> Path:
    for kind in ("lm", "latent"):
        path = CONFIG_GENERATE / kind / model
        if path.is_dir():
            return path
    return CONFIG_GENERATE / model


def _list_generate_names(model: str) -> list[str]:
    model_dir = resolve_generate_dir(model)
    if not model_dir.is_dir():
        return []
    return sorted(
        path.stem
        for path in model_dir.glob("*.yaml")
        if path.stem != "prototype"
    )


def _load_sampling(model: str, profile: str) -> dict[str, Any]:
    path = resolve_generate_dir(model) / f"{profile}.yaml"
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k != "name" and not str(k).startswith("_")}


def default_generate_spec(run_dir: Path, *, profile: str = "generate") -> dict[str, Any]:
    model = run_model_name(run_dir)
    names = _list_generate_names(model)
    if profile not in names:
        profile = "generate" if "generate" in names else (names[0] if names else "generate")
    sampling = _load_sampling(model, profile) if names else {}
    return {
        "model": model,
        "supports_prefix": model_supports_prefix(model),
        "profile": profile,
        "profiles": names,
        "sampling": sampling,
        "num_tokens": 1024,
        "num_samples": 1,
        "seed": 42,
    }


def query_gpu_memory() -> dict[str, Any]:
    """读 nvidia-smi；无卡或命令失败则 ok=False。"""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ok": False,
            "reason": f"本机没有可用显卡或 nvidia-smi 失败：{exc}",
            "gpus": [],
            "used_gib": None,
            "limit_gib": VRAM_USED_LIMIT_GIB,
        }
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            used_mib = float(parts[2])
            total_mib = float(parts[3])
        except ValueError:
            continue
        gpus.append({
            "index": int(parts[0]) if parts[0].isdigit() else 0,
            "name": parts[1],
            "used_mib": used_mib,
            "total_mib": total_mib,
            "used_gib": used_mib / 1024.0,
            "total_gib": total_mib / 1024.0,
        })
    if not gpus:
        return {
            "ok": False,
            "reason": "nvidia-smi 没有列出任何 GPU",
            "gpus": [],
            "used_gib": None,
            "limit_gib": VRAM_USED_LIMIT_GIB,
        }
    used_gib = float(gpus[0]["used_gib"])
    if used_gib >= VRAM_USED_LIMIT_GIB:
        return {
            "ok": False,
            "reason": (
                f"GPU0 显存占用 {used_gib:.2f} GiB ≥ {VRAM_USED_LIMIT_GIB:.0f} GiB，拒绝生成"
            ),
            "gpus": gpus,
            "used_gib": used_gib,
            "limit_gib": VRAM_USED_LIMIT_GIB,
        }
    return {
        "ok": True,
        "reason": "",
        "gpus": gpus,
        "used_gib": used_gib,
        "limit_gib": VRAM_USED_LIMIT_GIB,
    }


def checkpoints_payload(checkpoint_root: Path, run: str) -> dict[str, Any] | None:
    run_dir = resolve_run_dir(checkpoint_root, run)
    if run_dir is None:
        return None
    model = run_model_name(run_dir)
    return {
        "run": run,
        "model": model,
        "supports_prefix": model_supports_prefix(model),
        "checkpoints": list_run_checkpoints(run_dir),
        "run_dir": str(run_dir),
    }
