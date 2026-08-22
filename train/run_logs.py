"""Run 目录日志布局：迁移旧宽表、resume 截断、路径约定。"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from train.metrics import (
    EVAL_OFFICIAL_FIELDS,
    TRAIN_CSV_FIELDS_LM,
    TRAIN_OFFICIAL_FIELDS_LM,
    _train_log,
    ensure_csv_schema,
    eval_csv_fields,
    init_csv_header,
    train_csv_fields,
    truncate_csv_for_resume,
)
from models import kind_of

# 旧宽表扩展列 → 拆出目标
_OLD_TRAIN_EXT = {
    "loss_branch",
    "denoise_mse",
    "decode_ce",
    "late_ce",
    "lex_ce",
    "attr",
    "attr_rho",
    "chart_ce",
    "commit",
    "kl",
    "mask",
}
_OLD_EVAL_EXT = {
    "gen_loss",
    "gen_ppl",
    "gen_uniq_mean",
    "gen_nonempty_frac",
    "entropy",
    "dist1",
    "mean_entropy",
}


def train_metrics_dir(run_dir: Path) -> Path:
    return run_dir / "train_metrics"


def eval_components_dir(run_dir: Path) -> Path:
    return run_dir / "eval_components"


def eval_samples_dir(run_dir: Path) -> Path:
    return run_dir / "eval_samples"


def train_official_csv(run_dir: Path) -> Path:
    return train_metrics_dir(run_dir) / "official.csv"


def train_external_csv(run_dir: Path) -> Path:
    return train_metrics_dir(run_dir) / "external.csv"


def eval_official_csv(run_dir: Path) -> Path:
    return eval_components_dir(run_dir) / "official.csv"


def eval_external_csv(run_dir: Path) -> Path:
    return eval_components_dir(run_dir) / "external.csv"


def eval_tick_path(run_dir: Path) -> Path:
    return run_dir / "eval_tick.json"


def sample_step_dir(run_dir: Path, step: int) -> Path:
    return eval_samples_dir(run_dir) / f"step_{step:07d}"


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _backup(path: Path) -> Path:
    bak = path.with_name(path.name + ".pre_split")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def _is_old_train_wide(fields: list[str]) -> bool:
    return bool(set(fields) & _OLD_TRAIN_EXT)


def _is_old_eval_wide(fields: list[str]) -> bool:
    return bool(set(fields) & _OLD_EVAL_EXT)


def migrate_run_logs(run_dir: Path, *, model: str) -> None:
    """若仍为旧宽表则拆成主表 + 官方卫星；备份为 ``*.pre_split``（仅 lm）。"""
    if kind_of(model) == "latent":
        return
    train_csv = run_dir / "train_log.csv"
    eval_csv = run_dir / "eval_log.csv"

    if train_csv.exists():
        fields, rows = _read_rows(train_csv)
        if _is_old_train_wide(fields):
            _backup(train_csv)
            core_rows = []
            off_rows = []
            for row in rows:
                core = {k: row.get(k, "") for k in TRAIN_CSV_FIELDS_LM}
                core_rows.append(core)
                if any(row.get(k) not in (None, "") for k in _OLD_TRAIN_EXT):
                    off = {k: row.get(k, "") for k in TRAIN_OFFICIAL_FIELDS_LM}
                    off["step"] = row.get("step", "")
                    off_rows.append(off)
            _write_rows(train_csv, TRAIN_CSV_FIELDS_LM, core_rows)
            if off_rows:
                out = train_official_csv(run_dir)
                if not out.exists():
                    _write_rows(out, TRAIN_OFFICIAL_FIELDS_LM, off_rows)
            _train_log(
                f"Migrated train_log.csv → core + train_metrics/official.csv "
                f"({len(core_rows)} rows)"
            )

    if eval_csv.exists():
        fields, rows = _read_rows(eval_csv)
        if _is_old_eval_wide(fields):
            _backup(eval_csv)
            core_rows = []
            off_rows = []
            for row in rows:
                core = {k: row.get(k, "") for k in EVAL_CSV_FIELDS_LM}
                # 旧表可能无 lr
                core_rows.append(core)
                if any(row.get(k) not in (None, "") for k in _OLD_EVAL_EXT):
                    off = {k: "" for k in EVAL_OFFICIAL_FIELDS}
                    off["step"] = row.get("step", "")
                    for k in EVAL_OFFICIAL_FIELDS:
                        if k == "step":
                            continue
                        if k == "entropy" and not row.get("entropy"):
                            off[k] = row.get("mean_entropy", "")
                        else:
                            off[k] = row.get(k, "")
                    off_rows.append(off)
            _write_rows(eval_csv, EVAL_CSV_FIELDS_LM, core_rows)
            if off_rows:
                out = eval_official_csv(run_dir)
                if not out.exists():
                    _write_rows(out, EVAL_OFFICIAL_FIELDS, off_rows)
            _train_log(
                f"Migrated eval_log.csv → core + eval_components/official.csv "
                f"({len(core_rows)} rows)"
            )


def align_run_log_schemas(run_dir: Path, *, model: str) -> None:
    """已是新布局时：主表/官方表加列留空。"""
    train_fields = train_csv_fields(model)
    eval_fields = eval_csv_fields(model)
    train_csv = run_dir / "train_log.csv"
    eval_csv = run_dir / "eval_log.csv"
    if train_csv.exists():
        ensure_csv_schema(train_csv, train_fields)
    if eval_csv.exists():
        ensure_csv_schema(eval_csv, eval_fields)
    if kind_of(model) != "lm":
        return
    toff = train_official_csv(run_dir)
    if toff.exists():
        ensure_csv_schema(toff, TRAIN_OFFICIAL_FIELDS_LM)
    eoff = eval_official_csv(run_dir)
    if eoff.exists():
        ensure_csv_schema(eoff, EVAL_OFFICIAL_FIELDS)


def truncate_eval_samples(run_dir: Path, start_step: int) -> int:
    """删除 step >= start_step 的样本目录。"""
    root = eval_samples_dir(run_dir)
    if not root.is_dir():
        return 0
    removed = 0
    for child in list(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        try:
            step = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        if step >= start_step:
            shutil.rmtree(child)
            removed += 1
    return removed


def prepare_run_logs(
    run_dir: Path,
    *,
    model: str,
    start_step: int | None = None,
) -> dict[str, int]:
    """迁移 → schema 对齐 →（可选）按 step 截断。返回 kept 计数。"""
    migrate_run_logs(run_dir, model=model)
    align_run_log_schemas(run_dir, model=model)

    train_fields = train_csv_fields(model)
    eval_fields = eval_csv_fields(model)
    train_csv = run_dir / "train_log.csv"
    eval_csv = run_dir / "eval_log.csv"
    init_csv_header(train_csv, train_fields)
    init_csv_header(eval_csv, eval_fields)

    kept: dict[str, int] = {}
    if start_step is None:
        return kept

    kept["train_log"] = truncate_csv_for_resume(
        train_csv, start_step, train_fields,
    )
    kept["eval_log"] = truncate_csv_for_resume(
        eval_csv, start_step, eval_fields,
    )
    if kind_of(model) == "lm":
        toff = train_official_csv(run_dir)
        if toff.exists() or start_step is not None:
            if toff.exists():
                kept["train_official"] = truncate_csv_for_resume(
                    toff, start_step, TRAIN_OFFICIAL_FIELDS_LM,
                )
        eoff = eval_official_csv(run_dir)
        if eoff.exists():
            kept["eval_official"] = truncate_csv_for_resume(
                eoff, start_step, EVAL_OFFICIAL_FIELDS,
            )
        for label, path in (
            ("train_external", train_external_csv(run_dir)),
            ("eval_external", eval_external_csv(run_dir)),
        ):
            if not path.exists():
                continue
            fields, _ = _read_rows(path)
            if not fields:
                fields = ["step"]
            kept[label] = truncate_csv_for_resume(path, start_step, fields)
        kept["eval_samples_dirs"] = truncate_eval_samples(run_dir, start_step)
    return kept


def load_eval_tick(run_dir: Path) -> int:
    path = eval_tick_path(run_dir)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("tick", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def save_eval_tick(run_dir: Path, tick: int) -> None:
    path = eval_tick_path(run_dir)
    path.write_text(
        json.dumps({"tick": int(tick)}, indent=2) + "\n",
        encoding="utf-8",
    )
