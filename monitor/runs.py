"""训练 run 扫描与进度计算。"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monitor.config import CHECKPOINT_ROOT, LIVE_THRESHOLD_SEC, SOURCE_FILES
from monitor.hash_guide import guide_label, load_hash_guide


@dataclass
class RunRef:
    variant: str
    kind: str
    model: str
    hash: str

    @property
    def relpath(self) -> str:
        return f"{self.variant}/{self.kind}/{self.model}/{self.hash}"

    @property
    def legacy_relpath(self) -> str:
        return f"{self.variant}/{self.model}/{self.hash}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _fmt_ident_val(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    if val is None:
        return ""
    return str(val)


def _flatten_ident(obj: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            if str(key).startswith("_"):
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_ident(val, path))
        return out
    text = _fmt_ident_val(obj)
    if prefix and text != "":
        out[prefix] = text
    return out


def _run_identity(train: dict[str, Any], guide: dict[str, str] | None) -> dict[str, str]:
    extra = dict(train.get("extra") or {})
    refs = dict(extra.get("config_refs") or {})
    ident: dict[str, str] = {}
    cfg = train.get("model_config")
    variant = train.get("variant")
    if cfg:
        ident["config"] = f"{cfg}-{variant}" if variant else str(cfg)
    for src_key, out_key in (
        ("dataset", "dataset"),
        ("preprocess", "preprocess"),
        ("generate", "generate"),
    ):
        val = train.get(src_key) or refs.get(src_key) or ""
        if val:
            ident[out_key] = str(val)
    if refs.get("schedule"):
        ident["schedule"] = str(refs["schedule"])
    ov = refs.get("overrides") or {}
    if isinstance(ov, dict):
        ident.update(_flatten_ident(ov))
    if guide:
        raw_set = guide.get("set") or ""
        if raw_set and raw_set not in ("{}", ""):
            try:
                parsed = json.loads(raw_set)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                ident.update(_flatten_ident(parsed))
    return ident


def _infer_kind_from_train(train: dict[str, Any], path_kind: str | None) -> str:
    if path_kind in ("lm", "latent"):
        return path_kind
    model = str(train.get("model") or "")
    if model.startswith("latent_") or model in ("cola_vae", "latent_t5", "latent_vae"):
        return "latent"
    return "lm"


def _csv_columns(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            return list(header) if header else []
    except OSError:
        return []


def _last_csv_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as fb:
            header_raw = fb.readline()
            if not header_raw:
                return {}
            header = header_raw.decode("utf-8", errors="replace").strip().split(",")
            fb.seek(0, 2)
            size = fb.tell()
            chunk = min(max(0, size - len(header_raw)), 65536)
            if chunk <= 0:
                return {}
            fb.seek(-chunk, 2)
            tail = fb.read().decode("utf-8", errors="replace")
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        if not lines:
            return {}
        last_line = lines[-1]
        values = last_line.split(",")
        if len(values) != len(header):
            # 尾块可能从半行开始，用倒数第二行
            if len(lines) >= 2:
                values = lines[-2].split(",")
            if len(values) != len(header):
                return {}
        return {header[i]: values[i] for i in range(len(header))}
    except OSError:
        return {}


def _last_payload(progress: dict[str, Any], last_row: dict[str, str]) -> dict[str, Any]:
    return {
        "step": progress.get("step"),
        "tokens": progress.get("tokens"),
        "train_loss": _parse_float(last_row.get("train_loss")),
        "tokens_per_sec": progress.get("tokens_per_sec"),
        "curriculum_stage": last_row.get("curriculum_stage") or "",
    }


def _parse_int(val: str | None) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _parse_float(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _curriculum_stages(train: dict[str, Any]) -> list[dict[str, Any]]:
    extra = dict(train.get("extra") or {})
    cur = extra.get("curriculum_spec")
    if not isinstance(cur, dict):
        return []
    stages_raw = cur.get("stages")
    if not isinstance(stages_raw, list):
        return []
    stages: list[dict[str, Any]] = []
    cumulative = 0
    for item in stages_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"s{len(stages) + 1}")
        budget = int(item.get("effective_budget") or 0)
        start = cumulative
        cumulative += budget
        stages.append(
            {
                "name": name,
                "budget": budget,
                "start_tokens": start,
                "end_tokens": cumulative,
            },
        )
    return stages


def _compute_progress(
    train: dict[str, Any],
    last_row: dict[str, str],
    *,
    kind: str,
) -> dict[str, Any]:
    extra = dict(train.get("extra") or {})
    tokens = _parse_int(last_row.get("tokens")) or 0
    step = _parse_int(last_row.get("step")) or 0
    tps = _parse_float(last_row.get("tokens_per_sec"))

    stages_spec = _curriculum_stages(train)
    effective_target = int(
        extra.get("effective_target_tokens")
        or extra.get("curriculum_effective_tokens")
        or 0,
    )
    target_tokens = int(extra.get("target_tokens") or train.get("target_tokens") or 0)
    max_steps = int(train.get("max_steps") or extra.get("max_optimizer_steps") or 0)

    stages_out: list[dict[str, Any]] = []
    if stages_spec and effective_target > 0:
        overall = min(1.0, tokens / effective_target) if effective_target else 0.0
        for st in stages_spec:
            start = int(st["start_tokens"])
            end = int(st["end_tokens"])
            budget = int(st["budget"])
            if tokens >= end:
                frac = 1.0
                done = budget
            elif tokens <= start:
                frac = 0.0
                done = 0
            else:
                done = tokens - start
                frac = done / budget if budget else 0.0
            stages_out.append(
                {
                    "name": st["name"],
                    "budget": budget,
                    "start_tokens": start,
                    "end_tokens": end,
                    "done_tokens": done,
                    "fraction": round(frac, 4),
                },
            )
        return {
            "mode": "curriculum",
            "tokens": tokens,
            "target_tokens": effective_target,
            "fraction": round(overall, 4),
            "step": step,
            "tokens_per_sec": tps,
            "curriculum_stage": last_row.get("curriculum_stage") or "",
            "stages": stages_out,
        }

    if target_tokens > 0:
        frac = min(1.0, tokens / target_tokens)
    elif max_steps > 0:
        frac = min(1.0, step / max_steps)
    else:
        frac = 0.0

    return {
        "mode": "standard",
        "tokens": tokens,
        "target_tokens": target_tokens,
        "max_steps": max_steps,
        "fraction": round(frac, 4),
        "step": step,
        "tokens_per_sec": tps,
        "stages": [],
    }


def _is_live(run_dir: Path, now: float | None = None) -> bool:
    train_log = run_dir / "train_log.csv"
    if not train_log.is_file():
        return False
    ts = now if now is not None else time.time()
    return (ts - train_log.stat().st_mtime) < LIVE_THRESHOLD_SEC


def _list_eval_steps(run_dir: Path) -> list[int]:
    root = run_dir / "eval_samples"
    if not root.is_dir():
        return []
    steps: list[int] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        try:
            steps.append(int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(steps, reverse=True)


def _available_sources(run_dir: Path, *, columns: bool = False) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for source, rel in SOURCE_FILES.items():
        path = run_dir / rel
        if path.is_file():
            out[source] = _csv_columns(path) if columns else []
    return out


def discover_run_dirs(checkpoint_root: Path) -> list[tuple[RunRef, Path]]:
    found: list[tuple[RunRef, Path]] = []
    if not checkpoint_root.is_dir():
        return found

    for variant in ("full",):
        variant_dir = checkpoint_root / variant
        if not variant_dir.is_dir():
            continue
        for child in variant_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name in ("lm", "latent"):
                kind = child.name
                for model_dir in child.iterdir():
                    if not model_dir.is_dir():
                        continue
                    for hash_dir in model_dir.iterdir():
                        if hash_dir.is_dir() and (hash_dir / "train_log.csv").is_file():
                            found.append(
                                (
                                    RunRef(
                                        variant=variant,
                                        kind=kind,
                                        model=model_dir.name,
                                        hash=hash_dir.name,
                                    ),
                                    hash_dir,
                                ),
                            )
            else:
                # 旧布局 {variant}/{model}/{hash}；kind 在 scan 时再读 config
                model_name = child.name
                for hash_dir in child.iterdir():
                    if not hash_dir.is_dir():
                        continue
                    if not (hash_dir / "train_log.csv").is_file():
                        continue
                    found.append(
                        (
                            RunRef(
                                variant=variant,
                                kind=_infer_kind_from_train({"model": model_name}, None),
                                model=model_name,
                                hash=hash_dir.name,
                            ),
                            hash_dir,
                        ),
                    )
    return found


def scan_runs(checkpoint_root: Path) -> list[dict[str, Any]]:
    """只扫 ``full``；列表不列 eval_samples / 不读卫星表头。"""
    guide = load_hash_guide(checkpoint_root)
    now = time.time()
    runs: list[dict[str, Any]] = []
    for ref, run_dir in discover_run_dirs(checkpoint_root):
        cfg = _read_json(run_dir / "config.json")
        train = dict(cfg.get("train") or {})
        kind = _infer_kind_from_train(train, ref.kind)
        last = _last_csv_row(run_dir / "train_log.csv")
        live = _is_live(run_dir, now)
        progress = _compute_progress(train, last, kind=kind)
        g = guide.get(ref.hash)
        train_log = run_dir / "train_log.csv"
        mtime = train_log.stat().st_mtime if train_log.is_file() else 0.0
        item: dict[str, Any] = {
            "run": f"{ref.variant}/{kind}/{ref.model}/{ref.hash}",
            "variant": ref.variant,
            "kind": kind,
            "model": ref.model,
            "hash": ref.hash,
            "live": live,
            "progress": progress,
            "guide": guide_label(g),
            "identity": _run_identity(train, g),
            "last": _last_payload(progress, last),
            "mtime": mtime,
        }
        runs.append(item)

    runs.sort(key=lambda r: (not r["live"], -(r.get("mtime") or 0), r["run"]))
    return runs


def refresh_run_item(checkpoint_root: Path, item: dict[str, Any]) -> dict[str, Any]:
    """只重读该 hash 的 train_log 尾与 live，不重新扫目录。"""
    run_dir = resolve_run_dir(checkpoint_root, str(item.get("run") or ""))
    if run_dir is None:
        return item
    cfg = _read_json(run_dir / "config.json")
    train = dict(cfg.get("train") or {})
    kind = str(item.get("kind") or _infer_kind_from_train(train, None))
    last = _last_csv_row(run_dir / "train_log.csv")
    progress = _compute_progress(train, last, kind=kind)
    train_log = run_dir / "train_log.csv"
    mtime = train_log.stat().st_mtime if train_log.is_file() else 0.0
    next_item = dict(item)
    next_item["live"] = _is_live(run_dir)
    next_item["progress"] = progress
    next_item["mtime"] = mtime
    next_item["last"] = _last_payload(progress, last)
    return next_item


def refresh_runs_progress(
    checkpoint_root: Path,
    runs: list[dict[str, Any]],
    *,
    model: str | None = None,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    """按模型刷新进度；未指定则原样返回。"""
    if not model:
        return runs
    out: list[dict[str, Any]] = []
    for item in runs:
        if item.get("model") != model:
            out.append(item)
            continue
        if kind and item.get("kind") != kind:
            out.append(item)
            continue
        out.append(refresh_run_item(checkpoint_root, item))
    return out


def get_run_progress(checkpoint_root: Path, run_relpath: str) -> dict[str, Any] | None:
    """只读 train_log 尾与进度，不列 eval / 表头。"""
    run_dir = resolve_run_dir(checkpoint_root, run_relpath)
    if run_dir is None:
        return None
    parts = Path(run_relpath).parts
    kind = ""
    model = ""
    h = ""
    if len(parts) == 4:
        _, kind, model, h = parts
    elif len(parts) == 3:
        _, model, h = parts
    else:
        return None
    return refresh_run_item(
        checkpoint_root,
        {"run": run_relpath, "kind": kind, "model": model, "hash": h},
    )


def get_run_detail(checkpoint_root: Path, run_relpath: str) -> dict[str, Any] | None:
    parts = Path(run_relpath).parts
    if len(parts) == 4:
        variant, kind, model, h = parts
        run_dir = checkpoint_root / variant / kind / model / h
        ref = RunRef(variant=variant, kind=kind, model=model, hash=h)
    elif len(parts) == 3:
        variant, model, h = parts
        run_dir = checkpoint_root / variant / model / h
        if not run_dir.is_dir():
            run_dir = None
        cfg = _read_json((run_dir / "config.json") if run_dir else Path())
        train = dict(cfg.get("train") or {})
        kind = _infer_kind_from_train(train, None)
        ref = RunRef(variant=variant, kind=kind, model=model, hash=h)
        if run_dir is None or not run_dir.is_dir():
            run_dir = checkpoint_root / variant / kind / model / h
    else:
        return None

    if not run_dir.is_dir():
        # 再试旧布局
        if len(parts) == 4:
            variant, kind, model, h = parts
            alt = checkpoint_root / variant / model / h
            if alt.is_dir():
                run_dir = alt
        if not run_dir.is_dir():
            return None

    cfg = _read_json(run_dir / "config.json")
    train = dict(cfg.get("train") or {})
    guide = load_hash_guide(checkpoint_root).get(ref.hash)
    last = _last_csv_row(run_dir / "train_log.csv")
    sources = _available_sources(run_dir, columns=True)
    progress = _compute_progress(train, last, kind=ref.kind)
    return {
        "run": ref.relpath,
        "variant": ref.variant,
        "kind": ref.kind,
        "model": ref.model,
        "hash": ref.hash,
        "live": _is_live(run_dir),
        "progress": progress,
        "last": _last_payload(progress, last),
        "guide": guide_label(guide),
        "sources": sources,
        "eval_steps": _list_eval_steps(run_dir),
        "run_dir": str(run_dir),
    }


def resolve_run_dir(checkpoint_root: Path, run_relpath: str) -> Path | None:
    parts = Path(run_relpath).parts
    candidates: list[Path] = []
    if len(parts) == 4:
        variant, kind, model, h = parts
        candidates.append(checkpoint_root / variant / kind / model / h)
        candidates.append(checkpoint_root / variant / model / h)
    elif len(parts) == 3:
        variant, model, h = parts
        candidates.append(checkpoint_root / variant / model / h)
        for kind in ("lm", "latent"):
            candidates.append(checkpoint_root / variant / kind / model / h)
    for p in candidates:
        if p.is_dir():
            return p
    return None
