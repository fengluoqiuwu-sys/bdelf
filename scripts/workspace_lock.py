#!/usr/bin/env python3
"""本机工作区互斥锁：抢占 / 释放 / 查看。

锁目录：``temp/local-workspace.lock/``（目录存在即占锁）；``meta.json`` 记 holder。
抢锁：mkdir → 写 meta → 睡 1s → 读回校验 holder。

用法（仓库根）::

    .venv/bin/python scripts/workspace_lock.py acquire --holder auto-train:elf-cfg --purpose "edit code"
    .venv/bin/python scripts/workspace_lock.py release --holder auto-train:elf-cfg
    .venv/bin/python scripts/workspace_lock.py status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import repo_env

ROOT = repo_env.ensure_repo_root()
LOCK_DIR = ROOT / "temp" / "local-workspace.lock"
META_PATH = LOCK_DIR / "meta.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_meta() -> dict | None:
    if not META_PATH.is_file():
        return None
    try:
        data = json.loads(META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def cmd_status(_: argparse.Namespace) -> int:
    if not LOCK_DIR.exists():
        print("unlocked")
        return 0
    meta = _read_meta()
    if meta is None:
        print(f"locked (无有效 meta): {LOCK_DIR}")
        return 0
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


def cmd_acquire(args: argparse.Namespace) -> int:
    holder = args.holder.strip()
    if not holder:
        print("holder 不能为空", file=sys.stderr)
        return 2
    purpose = (args.purpose or "").strip() or "edit"
    LOCK_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        LOCK_DIR.mkdir(exist_ok=False)
    except FileExistsError:
        meta = _read_meta() or {}
        print(
            f"锁被占用: holder={meta.get('holder', '?')!r} "
            f"purpose={meta.get('purpose', '')!r} path={LOCK_DIR}",
            file=sys.stderr,
        )
        return 1

    meta = {
        "holder": holder,
        "purpose": purpose,
        "acquired_at": _now_iso(),
    }
    META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    time.sleep(1.0)
    got = _read_meta()
    if not got or got.get("holder") != holder:
        print(
            f"校验失败: holder={None if not got else got.get('holder')!r}（期望 {holder!r}）",
            file=sys.stderr,
        )
        # 他人已写入 meta：勿删；仅 meta 异常时尝试清理自己 mkdir 的目录
        if got is not None and got.get("holder") != holder:
            return 1
        try:
            META_PATH.unlink(missing_ok=True)
            LOCK_DIR.rmdir()
        except OSError:
            pass
        return 1
    print(f"acquired holder={holder!r} purpose={purpose!r}")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    holder = args.holder.strip()
    if not holder:
        print("holder 不能为空", file=sys.stderr)
        return 2
    if not LOCK_DIR.exists():
        print("unlocked（无需释放）")
        return 0
    meta = _read_meta()
    if meta is None:
        print(f"锁目录存在但无有效 meta，拒绝自动删除: {LOCK_DIR}", file=sys.stderr)
        return 1
    got = meta.get("holder")
    if got != holder:
        print(
            f"拒绝释放: 当前 holder={got!r}，不是 {holder!r}",
            file=sys.stderr,
        )
        return 1
    try:
        META_PATH.unlink(missing_ok=True)
        LOCK_DIR.rmdir()
    except OSError as exc:
        print(f"释放失败: {exc}", file=sys.stderr)
        return 1
    print(f"released holder={holder!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="本机 temp/local-workspace.lock 抢锁/释锁")
    sub = p.add_subparsers(dest="cmd", required=True)

    ac = sub.add_parser("acquire", help="抢锁（失败 exit 1）")
    ac.add_argument("--holder", required=True, help="持有者，如 auto-train:elf-cfg / human")
    ac.add_argument("--purpose", default="edit", help="用途简述")
    ac.set_defaults(func=cmd_acquire)

    rel = sub.add_parser("release", help="释锁（仅 holder 匹配时）")
    rel.add_argument("--holder", required=True, help="必须与抢锁时一致")
    rel.set_defaults(func=cmd_release)

    st = sub.add_parser("status", help="查看锁状态")
    st.set_defaults(func=cmd_status)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
