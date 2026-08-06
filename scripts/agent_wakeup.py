#!/usr/bin/env python3
"""延时后向 stdout 打印唤醒哨兵，把下一步指令交回 Cursor agent。

Cursor 无自主闹钟：在后台跑本脚本，turn 结束后由终端输出通知唤醒。
输出格式（一行 JSON）::

    AGENT_WAKEUP {"tag":"...","after_sec":300,"prompt":"<给 agent 的下一步>"}

用法（仓库根；通常 ``block_until_ms: 0`` 后台跑）::

    .venv/bin/python scripts/agent_wakeup.py --after 5m -- \\
      '跑 bash slurm/remote_status.sh；若 RUNNING 则 pull fast 并判继续/调整'

    .venv/bin/python scripts/agent_wakeup.py --nth 1 -- \\
      'auto-train 首次唤醒：查队列与日志'

    .venv/bin/python scripts/agent_wakeup.py --after 60m --tag resource-wait -- \\
      '额度/排队等待结束：再 remote_status'
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone

# auto-train 递进间隔（秒）：第 n 次唤醒前等待
NTH_SECONDS = {
    1: 5 * 60,
    2: 15 * 60,
    3: 30 * 60,
}
NTH_DEFAULT = 60 * 60  # 第 4 次起


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_duration(text: str) -> int:
    """解析 ``300`` / ``5m`` / ``1h`` / ``30s`` / ``1d`` → 秒。"""
    raw = text.strip().lower()
    if not raw:
        raise ValueError("空时长")
    if raw.isdigit():
        return int(raw)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m|h|d)", raw)
    if not m:
        raise ValueError(f"无法解析时长 {text!r}（例: 300 / 5m / 1h）")
    val = float(m.group(1))
    unit = m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    sec = int(val * mult)
    if sec < 1:
        raise ValueError("时长须 >= 1 秒")
    return sec


def nth_seconds(n: int) -> int:
    if n < 1:
        raise ValueError("--nth 须 >= 1")
    return NTH_SECONDS.get(n, NTH_DEFAULT)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="睡指定时间后打印 AGENT_WAKEUP，把 prompt 交回 agent",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--after",
        metavar="DUR",
        help="等待时长：秒数或 30s/5m/1h/1d",
    )
    g.add_argument(
        "--nth",
        type=int,
        metavar="N",
        help="auto-train 第 N 次唤醒间隔（1→5m, 2→15m, 3→30m, ≥4→60m）",
    )
    p.add_argument(
        "--tag",
        default="",
        help="可选标签（写入 JSON；默认 after 用 custom，nth 用 auto-train-N）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将等待的秒数与将输出的行，不 sleep",
    )
    p.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="唤醒后给 agent 的指令（也可用 -- 后的参数）",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 支持: script --after 5m -- 'prompt here'
    if "--" in argv:
        i = argv.index("--")
        pre, post = argv[:i], argv[i + 1 :]
        argv = pre
        prompt_from_dash = " ".join(post).strip()
    else:
        prompt_from_dash = ""

    parser = build_parser()
    args = parser.parse_args(argv)
    prompt = (prompt_from_dash or args.prompt or "").strip()
    if not prompt:
        print("须提供唤醒后给 agent 的指令（位置参数或 -- 之后）", file=sys.stderr)
        return 2

    try:
        if args.after is not None:
            after_sec = parse_duration(args.after)
            tag = args.tag.strip() or "custom"
        else:
            after_sec = nth_seconds(args.nth)
            tag = args.tag.strip() or f"auto-train-{args.nth}"
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = {
        "tag": tag,
        "after_sec": after_sec,
        "prompt": prompt,
        "scheduled_at": _now_iso(),
    }
    line = "AGENT_WAKEUP " + json.dumps(payload, ensure_ascii=False)

    if args.dry_run:
        print(f"dry-run: sleep {after_sec}s then:", file=sys.stderr)
        print(line)
        return 0

    print(
        f"[agent_wakeup] sleeping {after_sec}s tag={tag!r} until wakeup…",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(after_sec)
    payload["woke_at"] = _now_iso()
    line = "AGENT_WAKEUP " + json.dumps(payload, ensure_ascii=False)
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
