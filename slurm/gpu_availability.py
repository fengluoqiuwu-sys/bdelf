#!/usr/bin/env python3
"""查询 cls1 四个计算节点各有几张 GPU 可用。

本机直接跑 ``scontrol show node -o``（轻量只读，不占 GPU）。
按节点汇总：配置卡数 / 已分配 / 空闲 / 状态 / GPU 型号。

脚本不发起 SSH。本机查远端时，先 push，再：

    ssh ovan-server 'cd ~/source/bdelf && .venv/bin/python slurm/gpu_availability.py'

示例（在目标机仓库根执行）::

    .venv/bin/python slurm/gpu_availability.py
    .venv/bin/python slurm/gpu_availability.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_REPO_ROOT)

DEFAULT_NODES = ("cls1-srv1", "cls1-srv2", "cls1-srv3", "cls1-srv4")

_RE_NODE = re.compile(r"NodeName=(\S+)")
_RE_GRES = re.compile(r"\bGres=(\S+)")
_RE_STATE = re.compile(r"\bState=(\S+)")
_RE_CFG_GPU = re.compile(r"\bCfgTRES=[^\s]*gres/gpu=(\d+)")
_RE_ALLOC_GPU = re.compile(r"\bAllocTRES=(?:[^\s]*gres/gpu=(\d+))?")
_RE_REASON = re.compile(r"\bReason=(.+?)(?:\s+[A-Z][a-zA-Z]*=|\s*$)")


def _parse_gres_type(gres: str) -> str:
    # e.g. gpu:nvidia_geforce_rtx_4090:8 -> nvidia_geforce_rtx_4090
    parts = gres.split(":")
    if len(parts) >= 3 and parts[0] == "gpu":
        return parts[1]
    return gres or "?"


def _parse_node_line(line: str) -> dict | None:
    line = line.strip()
    if not line or "NodeName=" not in line:
        return None
    m_node = _RE_NODE.search(line)
    if not m_node:
        return None

    gres_m = _RE_GRES.search(line)
    gres = gres_m.group(1) if gres_m else ""
    state_m = _RE_STATE.search(line)
    state = state_m.group(1) if state_m else "?"
    cfg_m = _RE_CFG_GPU.search(line)
    total = int(cfg_m.group(1)) if cfg_m else 0
    alloc_m = _RE_ALLOC_GPU.search(line)
    used = int(alloc_m.group(1)) if alloc_m and alloc_m.group(1) else 0
    reason_m = _RE_REASON.search(line)
    reason = reason_m.group(1).strip() if reason_m else ""

    free = max(0, total - used)
    # DOWN / DRAIN 等不可调度时，空闲按 0 报告（仍保留 total/used）
    schedulable = not any(
        flag in state.upper().split("+")
        for flag in ("DOWN", "DRAIN", "NOT_RESPONDING", "FAIL", "UNKNOWN")
    )
    available = free if schedulable else 0

    return {
        "node": m_node.group(1),
        "gpu_type": _parse_gres_type(gres),
        "total": total,
        "used": used,
        "free": free,
        "available": available,
        "state": state,
        "reason": reason,
        "schedulable": schedulable,
    }


def fetch_nodes(nodes: tuple[str, ...]) -> tuple[list[dict], str]:
    proc = subprocess.run(
        ["scontrol", "show", "node", *nodes, "-o"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or f"scontrol failed (exit {proc.returncode})").strip()
        raise RuntimeError(msg)

    rows: list[dict] = []
    for line in proc.stdout.splitlines():
        row = _parse_node_line(line)
        if row:
            rows.append(row)
    return rows, proc.stderr.strip()


def _print_table(rows: list[dict]) -> None:
    headers = ("NODE", "GPU_TYPE", "TOTAL", "USED", "FREE", "AVAIL", "STATE")
    table = [
        (
            r["node"],
            r["gpu_type"],
            str(r["total"]),
            str(r["used"]),
            str(r["free"]),
            str(r["available"]),
            r["state"],
        )
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cols: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print(fmt(headers))
    print(fmt(tuple("-" * w for w in widths)))
    for row in table:
        print(fmt(row))

    avail_sum = sum(r["available"] for r in rows)
    free_sum = sum(r["free"] for r in rows)
    print()
    print(f"sum available (schedulable idle): {avail_sum}")
    print(f"sum free (total-used, ignore state): {free_sum}")
    for r in rows:
        if r["reason"] and not r["schedulable"]:
            print(f"note: {r['node']}: {r['reason']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="查询 cls1-srv[1-4] 空闲 GPU（本机 scontrol；不发起 SSH）。",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出（含 available / free / state 等字段）",
    )
    p.add_argument(
        "--nodes",
        default=",".join(DEFAULT_NODES),
        help="逗号分隔节点名（默认 cls1-srv1..4）",
    )
    args = p.parse_args(argv)

    nodes = tuple(n.strip() for n in args.nodes.split(",") if n.strip())
    if not nodes:
        p.error("--nodes 不能为空")

    try:
        rows, warn = fetch_nodes(nodes)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    except FileNotFoundError:
        print("scontrol 不在 PATH（请在 Slurm 登录/计算节点上执行本脚本）", file=sys.stderr)
        return 1

    missing = [n for n in nodes if not any(r["node"] == n for r in rows)]
    if missing:
        print(f"warning: no scontrol data for: {', '.join(missing)}", file=sys.stderr)
    if warn:
        print(warn, file=sys.stderr)

    by_name = {r["node"]: r for r in rows}
    ordered = [by_name[n] for n in nodes if n in by_name]

    if args.json:
        print(json.dumps(ordered, ensure_ascii=False, indent=2))
    else:
        _print_table(ordered)
    return 0 if ordered else 1


if __name__ == "__main__":
    sys.exit(main())
