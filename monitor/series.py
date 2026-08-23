"""CSV 时序读取、tokens 过滤与下采样。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator

from monitor.config import (
    DEFAULT_MAX_POINTS,
    EVAL_SERIES_CAP,
    MAX_POINTS_CAP,
    SOURCE_FILES,
    TRAIN_SOURCES,
)


def _parse_num(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _csv_header(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
            return list(header) if header else []
    except OSError:
        return []


def _iter_csv_rows(path: Path) -> tuple[list[str], Iterator[dict[str, str]]]:
    if not path.is_file():
        return [], iter(())
    f = open(path, newline="", encoding="utf-8")
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        f.close()
        return [], iter(())

    fields = list(reader.fieldnames)

    def _gen() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                if row:
                    yield {k: (row.get(k) or "") for k in fields}
        finally:
            f.close()

    return fields, _gen()


def _tail_last_x(path: Path, x_key: str) -> float | None:
    try:
        with open(path, "rb") as fb:
            fb.seek(0, 2)
            size = fb.tell()
            chunk = min(size, 65536)
            fb.seek(max(0, size - chunk))
            raw = fb.read().decode("utf-8", errors="replace")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if size > chunk and lines:
            lines = lines[1:]
        if len(lines) < 2:
            return None
        header = lines[0].split(",")
        if x_key not in header:
            return None
        xi = header.index(x_key)
        vals = lines[-1].split(",")
        if xi >= len(vals):
            return None
        return _parse_num(vals[xi])
    except OSError:
        return None


def _row_in_range(
    x: float,
    *,
    lo: float | None,
    hi: float | None,
) -> bool:
    if lo is not None and x < lo:
        return False
    if hi is not None and x > hi:
        return False
    return True


def _resolve_range(
    *,
    x_key: str,
    path: Path,
    tokens_from: int | None,
    tokens_to: int | None,
    last: int | None,
) -> tuple[float | None, float | None]:
    lo = float(tokens_from) if tokens_from is not None else None
    hi = float(tokens_to) if tokens_to is not None else None
    if last is not None:
        max_x = _tail_last_x(path, x_key)
        if max_x is not None:
            lo = max_x - float(last)
            hi = None
    return lo, hi


def _downsample_indices(n: int, max_points: int) -> list[int]:
    if n <= max_points:
        return list(range(n))
    if max_points <= 2:
        return [0, n - 1][:max_points]
    stride = (n - 1) / (max_points - 1)
    indices = {0, n - 1}
    for i in range(1, max_points - 1):
        indices.add(int(round(i * stride)))
    return sorted(indices)


def _make_point(
    row: dict[str, str],
    *,
    x_key: str,
    xv: float,
    metrics: list[str],
) -> dict[str, Any]:
    pt: dict[str, Any] = {"x": xv}
    for m in metrics:
        v = row.get(m)
        num = _parse_num(v)
        pt[m] = num if num is not None else v
    if "step" in row:
        pt["step"] = _parse_num(row.get("step"))
    return pt


def _collect_after_from_tail(
    path: Path,
    *,
    x_key: str,
    after: float,
    hi: float | None,
    fields: list[str],
) -> list[dict[str, str]]:
    """从文件尾往前读，只收集 x > after 的行（假定 tokens 单调递增）。"""
    chunk = 256 * 1024
    header_raw = b""
    lines: list[str] = []
    try:
        with open(path, "rb") as fb:
            header_raw = fb.readline()
            header_off = fb.tell()
            fb.seek(0, 2)
            end = fb.tell()
            if end <= header_off:
                return []
            pieces: list[bytes] = []
            pos = end
            while pos > header_off:
                start = max(header_off, pos - chunk)
                fb.seek(start)
                pieces.insert(0, fb.read(pos - start))
                pos = start
                raw = b"".join(pieces)
                text = raw.decode("utf-8", errors="replace")
                lines = [ln for ln in text.splitlines() if ln.strip()]
                if start > header_off and not raw.startswith(b"\n") and b"\n" in raw:
                    lines = lines[1:]
                if not lines:
                    continue
                try:
                    xi = fields.index(x_key)
                except ValueError:
                    return []
                first = next(csv.reader([lines[0]]), [])
                xv = _parse_num(first[xi] if xi < len(first) else None)
                if (xv is not None and xv <= after) or start == header_off:
                    break
    except OSError:
        return []

    header = header_raw.decode("utf-8", errors="replace").strip()
    body = "\n".join(lines)
    out: list[dict[str, str]] = []
    reader = csv.DictReader([header, *body.splitlines()] if body else [header])
    for row in reader:
        if not row:
            continue
        xv = _parse_num(row.get(x_key))
        if xv is None or xv <= after:
            continue
        if hi is not None and xv > hi:
            continue
        out.append({k: (row.get(k) or "") for k in fields})
    return out


_TOKEN_JOIN_CORE = {
    "eval_official": ("eval", "train"),
    "eval": ("train",),
    "train_official": ("train",),
}


def _core_step_tokens(run_dir: Path, source: str) -> dict[int, float]:
    cores = _TOKEN_JOIN_CORE.get(source) or ()
    out: dict[int, float] = {}
    for core in cores:
        rel = SOURCE_FILES.get(core)
        if not rel:
            continue
        path = run_dir / rel
        _, rows = _iter_csv_rows(path)
        for row in rows:
            step = _parse_num(row.get("step"))
            tokens = _parse_num(row.get("tokens"))
            if step is None or tokens is None:
                continue
            out[int(step)] = tokens
        if out:
            return out
    return out


def _rows_with_joined_tokens(
    path: Path,
    fields: list[str],
    token_map: dict[int, float],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    _, rows = _iter_csv_rows(path)
    for row in rows:
        step = _parse_num(row.get("step"))
        if step is None:
            continue
        tokens = token_map.get(int(step))
        if tokens is None:
            continue
        item = {k: (row.get(k) or "") for k in fields}
        item["tokens"] = str(int(tokens) if tokens == int(tokens) else tokens)
        out.append(item)
    return out


def _collect_filtered(
    path: Path,
    fields: list[str],
    rows: Iterator[dict[str, str]],
    *,
    x_key: str,
    lo: float | None,
    hi: float | None,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        xv = _parse_num(row.get(x_key))
        if xv is None:
            continue
        if not _row_in_range(xv, lo=lo, hi=hi):
            continue
        out.append(row)
    return out


def load_series(
    run_dir: Path,
    *,
    source: str,
    metrics: list[str],
    x_key: str = "tokens",
    tokens_from: int | None = None,
    tokens_to: int | None = None,
    last: int | None = None,
    after: float | None = None,
    max_points: int | None = None,
) -> dict[str, Any]:
    rel = SOURCE_FILES.get(source)
    if rel is None:
        raise ValueError(f"unknown source: {source!r}")

    path = run_dir / rel
    fields = _csv_header(path)
    empty = {
        "source": source,
        "path": str(path),
        "x_key": x_key,
        "metrics": [],
        "n_raw": 0,
        "n_returned": 0,
        "downsampled": False,
        "points": [],
    }
    if not fields:
        return empty

    joined_rows: list[dict[str, str]] | None = None
    range_path = path
    if (
        x_key == "tokens"
        and "tokens" not in fields
        and "step" in fields
        and source in _TOKEN_JOIN_CORE
    ):
        token_map = _core_step_tokens(run_dir, source)
        if token_map:
            joined_rows = _rows_with_joined_tokens(path, fields, token_map)
            fields = [*fields, "tokens"]
            for core in _TOKEN_JOIN_CORE[source]:
                rel = SOURCE_FILES.get(core)
                if rel and (run_dir / rel).is_file():
                    range_path = run_dir / rel
                    break

    if x_key not in fields:
        x_key = fields[0]

    lo, hi = _resolve_range(
        x_key=x_key,
        path=range_path,
        tokens_from=tokens_from,
        tokens_to=tokens_to,
        last=last,
    )

    cap = DEFAULT_MAX_POINTS
    if source in TRAIN_SOURCES:
        cap = min(max_points or DEFAULT_MAX_POINTS, MAX_POINTS_CAP)
    else:
        cap = EVAL_SERIES_CAP

    valid_metrics = [m for m in metrics if m in fields]

    if joined_rows is not None:
        filtered = []
        for row in joined_rows:
            xv = _parse_num(row.get(x_key))
            if xv is None:
                continue
            if after is not None and xv <= float(after):
                continue
            if not _row_in_range(xv, lo=lo, hi=hi):
                continue
            filtered.append(row)
        n_raw = len(filtered)
        indices = _downsample_indices(n_raw, cap) if n_raw > cap else list(range(n_raw))
        points = []
        for i in indices:
            row = filtered[i]
            xv = _parse_num(row.get(x_key))
            if xv is None:
                continue
            points.append(_make_point(row, x_key=x_key, xv=xv, metrics=valid_metrics))
        return {
            "source": source,
            "path": str(path),
            "x_key": x_key,
            "metrics": valid_metrics,
            "n_raw": n_raw,
            "n_returned": len(points),
            "downsampled": n_raw > len(points),
            "incremental": after is not None,
            "after": after,
            "points": points,
        }

    if after is not None:
        filtered = _collect_after_from_tail(
            path, x_key=x_key, after=float(after), hi=hi, fields=fields,
        )
        if lo is not None:
            kept = []
            for row in filtered:
                xv = _parse_num(row.get(x_key))
                if xv is not None and xv >= lo:
                    kept.append(row)
            filtered = kept
        n_raw = len(filtered)
        indices = _downsample_indices(n_raw, cap) if n_raw > cap else list(range(n_raw))
        points = []
        for i in indices:
            row = filtered[i]
            xv = _parse_num(row.get(x_key))
            if xv is None:
                continue
            points.append(_make_point(row, x_key=x_key, xv=xv, metrics=valid_metrics))
        return {
            "source": source,
            "path": str(path),
            "x_key": x_key,
            "metrics": valid_metrics,
            "n_raw": n_raw,
            "n_returned": len(points),
            "downsampled": n_raw > len(points),
            "incremental": True,
            "after": after,
            "points": points,
        }

    # 大 train 文件：两遍扫描，第二遍按 stride 取样，避免整表驻内存
    large_train = source in TRAIN_SOURCES and path.is_file() and path.stat().st_size > 2_000_000

    if large_train:
        n_raw = 0
        for row in _iter_csv_rows(path)[1]:
            xv = _parse_num(row.get(x_key))
            if xv is None:
                continue
            if _row_in_range(xv, lo=lo, hi=hi):
                n_raw += 1
        indices = set(_downsample_indices(n_raw, cap)) if n_raw > cap else None
        points: list[dict[str, Any]] = []
        valid_metrics = [m for m in metrics if m in fields]
        idx = 0
        for row in _iter_csv_rows(path)[1]:
            xv = _parse_num(row.get(x_key))
            if xv is None or not _row_in_range(xv, lo=lo, hi=hi):
                continue
            take = indices is None or idx in indices
            idx += 1
            if not take:
                continue
            pt: dict[str, Any] = {"x": xv}
            for m in valid_metrics:
                v = row.get(m)
                num = _parse_num(v)
                pt[m] = num if num is not None else v
            if "step" in row:
                pt["step"] = _parse_num(row.get("step"))
            points.append(pt)
        return {
            "source": source,
            "path": str(path),
            "x_key": x_key,
            "metrics": valid_metrics,
            "n_raw": n_raw,
            "n_returned": len(points),
            "downsampled": n_raw > len(points),
            "points": points,
        }

    filtered = _collect_filtered(path, fields, _iter_csv_rows(path)[1], x_key=x_key, lo=lo, hi=hi)
    n_raw = len(filtered)
    indices = _downsample_indices(n_raw, cap) if n_raw > cap else list(range(n_raw))
    downsampled = n_raw > len(indices)

    valid_metrics = [m for m in metrics if m in fields]
    points = []
    for i in indices:
        row = filtered[i]
        pt: dict[str, Any] = {"x": _parse_num(row.get(x_key))}
        for m in valid_metrics:
            v = row.get(m)
            num = _parse_num(v)
            pt[m] = num if num is not None else v
        if "step" in row:
            pt["step"] = _parse_num(row.get("step"))
        points.append(pt)

    return {
        "source": source,
        "path": str(path),
        "x_key": x_key,
        "metrics": valid_metrics,
        "n_raw": n_raw,
        "n_returned": len(points),
        "downsampled": downsampled,
        "points": points,
    }
