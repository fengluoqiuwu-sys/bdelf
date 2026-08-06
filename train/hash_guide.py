"""给人看的 checkpoint 哈希指引 CSV（``cache/checkpoints/hash_guide.csv``）。

列：model, config, dataset, Preprocess, generate, hash, set
按前六列字典序排序；``set`` 为 overrides 的单行 JSON。
仅收录 ``full`` 变体；``fast`` 冒烟不写入。
本文件仅本地维护，不随 sync push/pull。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from train.train import CHECKPOINT_ROOT

HASH_GUIDE_FILENAME = "hash_guide.csv"
COLUMNS = (
    "model",
    "config",
    "dataset",
    "Preprocess",
    "generate",
    "hash",
    "set",
)


def hash_guide_path(root: Path | None = None) -> Path:
    return Path(root or CHECKPOINT_ROOT) / HASH_GUIDE_FILENAME


def _set_json(overrides: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(overrides or {}), ensure_ascii=False, separators=(",", ":"))


def _is_full_train(train: Mapping[str, Any]) -> bool:
    return str(train.get("variant") or "") == "full"


def row_from_train(train: Mapping[str, Any]) -> dict[str, str]:
    extra = dict(train.get("extra") or {})
    refs = dict(extra.get("config_refs") or {})
    cfg_hash = str(extra.get("config_hash") or train.get("name") or "")
    return {
        "model": str(train["model"]),
        "config": f"{train['model_config']}-{train['variant']}",
        "dataset": str(train["dataset"]),
        "Preprocess": str(train["preprocess"]),
        "generate": str(train["generate"]),
        "hash": cfg_hash,
        "set": _set_json(refs.get("overrides")),
    }


def _sort_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(row[c] for c in COLUMNS[:-1])


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for raw in reader:
            if not raw:
                continue
            rows.append({c: str(raw.get(c) or "") for c in COLUMNS})
        return rows


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=_sort_key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
    tmp.replace(path)


def upsert_hash_guide_row(
    train: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> Path | None:
    """按 hash 覆盖/插入一行，再按约定列排序写回。``fast`` 变体跳过。"""
    path = hash_guide_path(root)
    if not _is_full_train(train):
        return None
    new_row = row_from_train(train)
    rows = [r for r in _read_rows(path) if r.get("hash") != new_row["hash"]]
    rows.append(new_row)
    _write_rows(path, rows)
    return path


def rebuild_hash_guide(*, root: Path | None = None) -> Path:
    """扫描 ``full/{model}/{hash}/config.json`` 重建指引表（不含 fast）。"""
    root = Path(root or CHECKPOINT_ROOT)
    rows: list[dict[str, str]] = []
    for cfg_path in sorted(root.glob("full/*/*/config.json")):
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        train = payload.get("train")
        if not isinstance(train, dict):
            continue
        if not train.get("model") or not train.get("model_config"):
            continue
        if not _is_full_train(train):
            continue
        rows.append(row_from_train(train))
    # 同 hash 去重（后写覆盖）
    by_hash: dict[str, dict[str, str]] = {}
    for row in rows:
        by_hash[row["hash"]] = row
    path = hash_guide_path(root)
    _write_rows(path, list(by_hash.values()))
    return path
