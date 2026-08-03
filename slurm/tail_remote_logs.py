#!/usr/bin/env python3
"""读取本机（通常为远端仓库）Slurm 训练 .out / .err 末 N 行。

日志命名与 ``#SBATCH --output/--error`` 一致：``<job-name>-<job-id>.{out,err}``，
默认目录：仓库根下 ``slurm/logs``。

脚本不发起 SSH。本机查远端日志时，先 push，再：

    ssh ovan-server 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py 1234567'

示例（在目标机仓库根执行）::

    .venv/bin/python slurm/tail_remote_logs.py 1234567
    .venv/bin/python slurm/tail_remote_logs.py 1234567 -n 120
    .venv/bin/python slurm/tail_remote_logs.py 1234567 --which err
    .venv/bin/python slurm/tail_remote_logs.py --list
    .venv/bin/python slurm/tail_remote_logs.py --job-name ar-100m-full -n 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_REPO_ROOT)

DEFAULT_LOG_DIR = _REPO_ROOT / "slurm" / "logs"
DEFAULT_N = 80


def cmd_list(log_dir: Path, limit: int) -> int:
    if not log_dir.is_dir():
        print(f"missing log dir: {log_dir}", file=sys.stderr)
        return 1

    files = sorted(
        list(log_dir.glob("*.out")) + list(log_dir.glob("*.err")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        print(f"note: no .out/.err yet under {log_dir}; listing all files:", file=sys.stderr)
        files = sorted(log_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

    for path in files[:limit]:
        st = path.stat()
        print(f"{st.st_mtime:10.0f}  {st.st_size:12d}  {path}")
    return 0


def _existing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.is_file()]


def _pick_latest_by_name(log_dir: Path, job_name: str) -> Path:
    cands = sorted(
        log_dir.glob(f"{job_name}-*.out"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"no .out for job-name={job_name} under {log_dir}")
    return cands[0]


def cmd_tail(
    log_dir: Path,
    n: int,
    which: str,
    job_id: str | None,
    job_name: str | None,
) -> int:
    if not log_dir.is_dir():
        print(f"missing log dir: {log_dir}", file=sys.stderr)
        return 1

    outs: list[Path] = []
    errs: list[Path] = []
    if job_id and job_name:
        outs = [log_dir / f"{job_name}-{job_id}.out"]
        errs = [log_dir / f"{job_name}-{job_id}.err"]
    elif job_id:
        outs = sorted(log_dir.glob(f"*-{job_id}.out"))
        errs = sorted(log_dir.glob(f"*-{job_id}.err"))
    else:
        assert job_name is not None
        try:
            latest = _pick_latest_by_name(log_dir, job_name)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        outs = [latest]
        errs = [latest.with_suffix(".err")]

    real_outs = _existing(outs)
    real_errs = _existing(errs)

    if len(real_outs) > 1 or len(real_errs) > 1:
        print(f"multiple matches for job_id={job_id}; pass --job-name", file=sys.stderr)
        for p in real_outs + real_errs:
            print(p, file=sys.stderr)
        return 2

    found = False
    targets: list[Path] = []
    if which in ("out", "both"):
        targets.extend(real_outs)
    if which in ("err", "both"):
        targets.extend(real_errs)

    for path in targets:
        found = True
        print(f"===== {path} (last {n} lines) =====")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"read failed: {exc}", file=sys.stderr)
            return 1
        lines = text.splitlines()
        print("\n".join(lines[-n:]))
        print()

    if not found:
        print(f"no matching log files under {log_dir}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="读取本机 Slurm .out/.err 末 N 行（不发起 SSH；远端请先 push 再 ssh 执行）。",
    )
    p.add_argument(
        "job_id",
        nargs="?",
        help="Slurm job id（匹配 *-<job_id>.{out,err}）",
    )
    p.add_argument(
        "-n",
        "--lines",
        type=int,
        default=DEFAULT_N,
        metavar="N",
        help=f"末行数（默认 {DEFAULT_N}）",
    )
    p.add_argument(
        "--which",
        choices=("both", "out", "err"),
        default="both",
        help="读 .out / .err / 两者（默认 both）",
    )
    p.add_argument(
        "--job-name",
        help="作业名（#SBATCH --job-name）；与 job_id 合用可精确定位，"
        "单独使用则取该名前缀最新的 .out/.err",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="列出 logs 目录最近文件后退出",
    )
    p.add_argument(
        "--list-limit",
        type=int,
        default=30,
        help="--list 时最多显示行数（默认 30）",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"日志目录（默认 {DEFAULT_LOG_DIR}）",
    )
    args = p.parse_args(argv)

    if args.lines < 1:
        p.error("-n/--lines must be >= 1")
    if args.job_id is not None and not args.job_id.isdigit():
        p.error("job_id must be numeric")

    log_dir = args.log_dir.expanduser()
    if not log_dir.is_absolute():
        log_dir = (_REPO_ROOT / log_dir).resolve()

    if args.list:
        return cmd_list(log_dir, args.list_limit)

    if args.job_id is None and not args.job_name:
        p.error("需要 job_id，或 --job-name，或 --list")

    return cmd_tail(
        log_dir=log_dir,
        n=args.lines,
        which=args.which,
        job_id=args.job_id,
        job_name=args.job_name,
    )


if __name__ == "__main__":
    sys.exit(main())
