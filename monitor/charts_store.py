"""图表配置落盘（cache/monitor，gitignore）。"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monitor.config import MONITOR_STORE

_LOCK = threading.Lock()


def _path(repo_root: Path) -> Path:
    return repo_root / MONITOR_STORE


def _empty_store() -> dict[str, Any]:
    return {"version": 1, "charts": {}, "export_prefs": {"invert": False, "width": 960, "height": 360}}


def _chart_key(kind: str, model: str) -> str:
    kind = (kind or "").strip() or "_"
    model = (model or "").strip() or "_"
    if "/" in kind or "\\" in kind or "/" in model or "\\" in model:
        raise ValueError("kind/model 不能含路径分隔符")
    return f"{kind}/{model}"


def _read_store(repo_root: Path) -> dict[str, Any]:
    path = _path(repo_root)
    if not path.is_file():
        return _empty_store()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_store()
    if not isinstance(data, dict):
        return _empty_store()
    data.setdefault("version", 1)
    data.setdefault("charts", {})
    data.setdefault("export_prefs", {"invert": False, "width": 960, "height": 360})
    if not isinstance(data["charts"], dict):
        data["charts"] = {}
    return data


def _write_store(repo_root: Path, data: dict[str, Any]) -> None:
    path = _path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def get_chart_entry(
    repo_root: Path, kind: str, model: str
) -> tuple[bool, list[Any], dict[str, list[str]], dict[str, list[str]]]:
    key = _chart_key(kind, model)
    with _LOCK:
        store = _read_store(repo_root)
        charts = store.get("charts", {})
        if key not in charts:
            return False, [], {}, {}
        bucket = charts.get(key) or {}
        if not isinstance(bucket, dict):
            return True, [], {}, {}
        panels = bucket.get("panels")
        dismissed = _clean_dismissed(bucket.get("dismissed"))
        order = _clean_order(bucket.get("order"))
        return True, list(panels) if isinstance(panels, list) else [], dismissed, order


def get_panels(repo_root: Path, kind: str, model: str) -> list[Any]:
    _, panels, _, _ = get_chart_entry(repo_root, kind, model)
    return panels


def _clean_dismissed(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for hash_key, ids in raw.items():
        if not isinstance(hash_key, str) or "/" in hash_key or "\\" in hash_key:
            continue
        if not isinstance(ids, list):
            continue
        cleaned = [str(x) for x in ids if x]
        if cleaned:
            out[hash_key] = cleaned
    return out


def _clean_order_key(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    key = raw.strip()
    if not (key.startswith("d:") or key.startswith("c:")):
        return None
    rest = key[2:]
    if not rest or "/" in rest or "\\" in rest:
        return None
    return key


def _clean_order(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for hash_key, keys in raw.items():
        if not isinstance(hash_key, str) or "/" in hash_key or "\\" in hash_key:
            continue
        if not isinstance(keys, list):
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in keys:
            key = _clean_order_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
        if cleaned:
            out[hash_key] = cleaned
    return out


def put_chart_state(
    repo_root: Path,
    kind: str,
    model: str,
    *,
    panels: list[Any] | None = None,
    dismissed: dict[str, Any] | None = None,
    order: dict[str, Any] | None = None,
) -> None:
    key = _chart_key(kind, model)
    with _LOCK:
        store = _read_store(repo_root)
        charts = store.setdefault("charts", {})
        bucket = charts.get(key) if isinstance(charts.get(key), dict) else {}
        if panels is not None:
            bucket["panels"] = panels
        if dismissed is not None:
            bucket["dismissed"] = _clean_dismissed(dismissed)
        if order is not None:
            bucket["order"] = _clean_order(order)
        charts[key] = bucket
        _write_store(repo_root, store)


def put_panels(repo_root: Path, kind: str, model: str, panels: list[Any]) -> None:
    put_chart_state(repo_root, kind, model, panels=panels)


def _clamp_export_prefs(prefs: dict[str, Any] | None) -> dict[str, Any]:
    raw = prefs if isinstance(prefs, dict) else {}
    invert = bool(raw.get("invert"))
    try:
        height = int(raw.get("height") or 360)
    except (TypeError, ValueError):
        height = 360
    try:
        width = int(raw.get("width") or 960)
    except (TypeError, ValueError):
        width = 960
    return {
        "invert": invert,
        "width": min(4096, max(240, width)),
        "height": min(2400, max(120, height)),
    }


def get_export_prefs(repo_root: Path) -> dict[str, Any]:
    with _LOCK:
        prefs = _read_store(repo_root).get("export_prefs") or {}
    return _clamp_export_prefs(prefs)


def put_export_prefs(repo_root: Path, prefs: dict[str, Any]) -> dict[str, Any]:
    out = _clamp_export_prefs(prefs)
    with _LOCK:
        store = _read_store(repo_root)
        store["export_prefs"] = out
        _write_store(repo_root, store)
    return out
