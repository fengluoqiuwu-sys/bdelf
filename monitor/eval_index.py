"""离线 eval 索引与详情。"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from eval.report import CSV_METRIC_KEYS, suggest_run_name
from monitor.hash_guide import load_hash_guide
from monitor.runs import _infer_kind_from_train, _run_identity

EVAL_METRIC_KEYS = ("name",) + CSV_METRIC_KEYS

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def scan_eval_tree(eval_root: Path) -> list[dict[str, Any]]:
    if not eval_root.is_dir():
        return []
    models: list[dict[str, Any]] = []
    for model_dir in sorted(eval_root.iterdir()):
        if not model_dir.is_dir():
            continue
        hashes: list[dict[str, Any]] = []
        for hash_dir in sorted(model_dir.iterdir()):
            if not hash_dir.is_dir():
                continue
            steps: list[dict[str, Any]] = []
            for step_dir in sorted(hash_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
                if not step_dir.is_dir() or not step_dir.name.isdigit():
                    continue
                step_info = _scan_step_dir(step_dir)
                if step_info:
                    steps.append(step_info)
            if steps:
                hashes.append({"hash": hash_dir.name, "steps": steps})
        if hashes:
            models.append({"model": model_dir.name, "hashes": hashes})
    return models


def _eval_kind(
    model: str,
    guide_row: dict[str, str] | None,
    run: dict[str, Any] | None,
) -> str:
    hint = ""
    if run and run.get("kind") in ("lm", "latent"):
        hint = str(run["kind"])
    elif guide_row and guide_row.get("kind") in ("lm", "latent"):
        hint = str(guide_row["kind"])
    return _infer_kind_from_train({"model": model}, hint or None)


def _eval_identity(
    model: str,
    guide_row: dict[str, str] | None,
    run: dict[str, Any] | None,
) -> dict[str, str]:
    if run and isinstance(run.get("identity"), dict) and run["identity"]:
        return dict(run["identity"])
    train: dict[str, Any] = {"model": model}
    if guide_row:
        if guide_row.get("config"):
            train["model_config"] = guide_row["config"]
        if guide_row.get("dataset"):
            train["dataset"] = guide_row["dataset"]
        if guide_row.get("Preprocess"):
            train["preprocess"] = guide_row["Preprocess"]
        if guide_row.get("generate"):
            train["generate"] = guide_row["generate"]
    return _run_identity(train, guide_row)


def _hash_counts(steps: list[dict[str, Any]]) -> tuple[int, int, int | None]:
    step_count = len(steps)
    run_count = 0
    latest = None
    for s in steps:
        run_count += int(s.get("run_count") or len(s.get("runs") or []))
        st = s.get("step")
        if isinstance(st, int) and (latest is None or st > latest):
            latest = st
    return step_count, run_count, latest


def enrich_eval_tree(
    models: list[dict[str, Any]],
    *,
    checkpoint_root: Path,
    runs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """按 kind 拆模型，并给每个 hash 补 identity（hash_guide / 训练 run）。"""
    guide = load_hash_guide(checkpoint_root)
    run_by_hash: dict[str, dict[str, Any]] = {}
    for r in runs or []:
        h = str(r.get("hash") or "")
        if h and h not in run_by_hash:
            run_by_hash[h] = r
    buckets: dict[str, dict[str, Any]] = {}
    for m in models:
        model = str(m.get("model") or "")
        for raw in m.get("hashes") or []:
            hid = str(raw.get("hash") or "")
            steps = list(raw.get("steps") or [])
            run = run_by_hash.get(hid)
            g = guide.get(hid)
            kind = _eval_kind(model, g, run)
            step_count, run_count, latest_step = _hash_counts(steps)
            key = f"{kind}/{model}"
            bucket = buckets.setdefault(
                key,
                {"kind": kind, "model": model, "hashes": []},
            )
            bucket["hashes"].append(
                {
                    "hash": hid,
                    "kind": kind,
                    "model": model,
                    "identity": _eval_identity(model, g, run),
                    "steps": steps,
                    "step_count": step_count,
                    "run_count": run_count,
                    "latest_step": latest_step,
                },
            )
    out: list[dict[str, Any]] = []
    for bucket in buckets.values():
        hashes = bucket["hashes"]
        bucket["count"] = len(hashes)
        bucket["step_count"] = sum(int(h.get("step_count") or 0) for h in hashes)
        bucket["run_count"] = sum(int(h.get("run_count") or 0) for h in hashes)
        out.append(bucket)
    out.sort(key=lambda x: (x.get("kind") or "", x.get("model") or ""))
    return out



def _scan_step_dir(step_dir: Path, *, include_metrics: bool = False) -> dict[str, Any] | None:
    runs: list[dict[str, Any]] = []
    for child in sorted(step_dir.iterdir()):
        if not child.is_dir():
            continue
        summary = child / "summary.json"
        fp = child / "fingerprint.json"
        if not summary.is_file():
            continue
        fp_data = _read_json(fp)
        name = str(fp_data.get("name") or "")
        if not name:
            name = suggest_run_name(
                fp_data.get("generate_overrides"),
                sampling=fp_data.get("sampling"),
            )
        runs.append(
            {
                "generate_hash": child.name,
                "name": name,
                "has_samples": (child / "samples.txt").is_file(),
                "has_per_sample_csv": (child / "per_sample.csv").is_file(),
            },
        )
    metrics_rows: list[dict[str, Any]] = []
    if include_metrics:
        csv_path = step_dir / "results.csv"
        if csv_path.is_file():
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row:
                        metrics_rows.append(dict(row))
    if not runs and not metrics_rows:
        return None
    return {
        "step": int(step_dir.name),
        "runs": runs,
        "has_chart": (step_dir / "results.png").is_file(),
        "has_table": (step_dir / "results_table.png").is_file(),
        "metrics_rows": metrics_rows if include_metrics else [],
        "run_count": len(runs),
    }


def get_eval_step(
    eval_root: Path,
    model: str,
    model_hash: str,
    step: int,
) -> dict[str, Any] | None:
    step_dir = eval_root / model / model_hash / str(step)
    if not step_dir.is_dir():
        return None
    info = _scan_step_dir(step_dir, include_metrics=True)
    if not info:
        return None
    return {
        "model": model,
        "hash": model_hash,
        "step": step,
        **info,
        "chart_url": f"/media/eval/{model}/{model_hash}/{step}/results.png"
        if (step_dir / "results.png").is_file()
        else None,
        "table_url": f"/media/eval/{model}/{model_hash}/{step}/results_table.png"
        if (step_dir / "results_table.png").is_file()
        else None,
    }


def get_eval_run_detail(
    eval_root: Path,
    model: str,
    model_hash: str,
    step: int,
    generate_hash: str,
) -> dict[str, Any] | None:
    run_dir = eval_root / model / model_hash / str(step) / generate_hash
    if not run_dir.is_dir():
        return None
    from monitor.samples_parse import load_offline_samples

    fp = _read_json(run_dir / "fingerprint.json")
    summary = _read_json(run_dir / "summary.json")
    samples = load_offline_samples(run_dir)
    return {
        "model": model,
        "hash": model_hash,
        "step": step,
        "generate_hash": generate_hash,
        "name": fp.get("name") or suggest_run_name(
            fp.get("generate_overrides"),
            sampling=fp.get("sampling"),
        ),
        "fingerprint": fp,
        "summary": summary,
        "samples": samples,
    }
