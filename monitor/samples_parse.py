"""样本解析（离线 samples.txt / 在线 samples.csv）。"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

_SAMPLE_HDR_RE = re.compile(
    r"^###\s*sample\s+(\d+)/(\d+)"
    r"(?:\s+gen_ppl=(\S+))?"
    r"(?:\s+entropy=(\S+))?"
    r"(?:\s+seq_rep_4=(\S+))?"
    r"(?:\s+accept=(\S+))?"
    r"(?:\s+nonword_frac=(\S+))?"
    r"(?:\s+nonword_n=(\S+))?"
    r"(?:\s+glue_n=(\S+))?"
    r"(?:\s+cola_g=(\S+))?",
)


def _pf(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _pi(s: str | None) -> int | None:
    if s is None:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_samples_txt(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8")
    if "### sample" not in raw:
        chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]
        return [{"index": i + 1, "text": t} for i, t in enumerate(chunks)]

    lines = raw.splitlines()
    samples: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("===") and i + 1 < len(lines):
            hdr = lines[i + 1]
            sm = _SAMPLE_HDR_RE.match(hdr.strip())
            if sm:
                i += 3
                buf: list[str] = []
                while i < len(lines) and not lines[i].startswith("==="):
                    buf.append(lines[i])
                    i += 1
                samples.append(
                    {
                        "index": _pi(sm.group(1)),
                        "total": _pi(sm.group(2)),
                        "gen_ppl": _pf(sm.group(3)),
                        "entropy": _pf(sm.group(4)),
                        "seq_rep_4": _pf(sm.group(5)),
                        "accept": _pi(sm.group(6)),
                        "nonword_frac": _pf(sm.group(7)),
                        "nonword_n": _pi(sm.group(8)),
                        "glue_n": _pi(sm.group(9)),
                        "cola_g": _pf(sm.group(10)),
                        "text": "\n".join(buf).strip(),
                    },
                )
                continue
        i += 1
    return samples


def load_per_sample_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader if row]


def load_offline_samples(run_dir: Path) -> list[dict[str, Any]]:
    per = run_dir / "per_sample.csv"
    if per.is_file():
        return load_per_sample_csv(per)
    return parse_samples_txt(run_dir / "samples.txt")


def load_online_lm_samples(samples_dir: Path) -> list[dict[str, Any]]:
    path = samples_dir / "samples.csv"
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: list[dict[str, Any]] = []
        for row in reader:
            if not row:
                continue
            item = dict(row)
            for key in ("gen_ppl", "entropy"):
                if key in item:
                    item[key] = _pf(str(item[key]))
            if "gen_len" in item:
                item["gen_len"] = _pi(str(item["gen_len"]))
            out.append(item)
        return out
