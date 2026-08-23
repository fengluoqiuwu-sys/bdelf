"""读取 hash_guide.csv（不依赖 models）。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from monitor.config import HASH_GUIDE_NAME

COLUMNS = (
    "kind",
    "model",
    "config",
    "dataset",
    "Preprocess",
    "generate",
    "hash",
    "set",
)


def load_hash_guide(checkpoint_root: Path) -> dict[str, dict[str, str]]:
    path = checkpoint_root / HASH_GUIDE_NAME
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if not raw:
                continue
            row = {c: str(raw.get(c) or "") for c in COLUMNS}
            h = row.get("hash") or ""
            if h:
                out[h] = row
    return out


def guide_label(row: dict[str, str] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "config_label": row.get("config") or "",
        "dataset": row.get("dataset") or "",
        "preprocess": row.get("Preprocess") or "",
        "generate": row.get("generate") or "",
        "set": row.get("set") or "",
    }
