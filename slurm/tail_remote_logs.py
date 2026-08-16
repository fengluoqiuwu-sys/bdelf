#!/usr/bin/env python3
"""读取本机（通常为远端仓库）训练作业 .out / .err 末 N 行。

新布局（Slurm / common 统一）::

    logs/<server-name>/<时间戳>/<job-name>-<job-id>.{out,err}
    logs/<server-name>/<时间戳>/gpu-<job-id>.log

旧布局（兼容）:: ``slurm/logs/<job-name>-<job-id>.{out,err}``。

脚本不发起 SSH。本机查远端日志时，先 push，再（工作目录见 servers.csv；本仓库通常 ``~/source/bdelf``）：

    ssh ovan-server 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py 1234567'
    ssh train-server-1 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py pid12345'

示例（在目标机仓库根执行）::

    .venv/bin/python slurm/tail_remote_logs.py 1234567
    .venv/bin/python slurm/tail_remote_logs.py 1234567 -n 120
    .venv/bin/python slurm/tail_remote_logs.py 1234567 --which err
    .venv/bin/python slurm/tail_remote_logs.py --list
    .venv/bin/python slurm/tail_remote_logs.py --job-name ar-100m-full -n 50
    .venv/bin/python slurm/tail_remote_logs.py --server ovan-server --list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_REPO_ROOT)

DEFAULT_LOGS_ROOT = _REPO_ROOT / "logs"
LEGACY_LOG_DIR = _REPO_ROOT / "slurm" / "logs"
DEFAULT_N = 80


def _iter_log_dirs(logs_root: Path, server: str | None) -> list[Path]:
    """返回要扫描的叶子日志目录（时间戳目录 + 旧 slurm/logs）。"""
    dirs: list[Path] = []
    if logs_root.is_dir():
        servers = [logs_root / server] if server else sorted(logs_root.iterdir())
        for srv in servers:
            if not srv.is_dir():
                continue
            for child in sorted(srv.iterdir()):
                if child.is_dir() and child.name != "pending" and not child.name.startswith("_"):
                    dirs.append(child)
                elif child.name == "_sbatch-default" and child.is_dir():
                    dirs.append(child)
    if LEGACY_LOG_DIR.is_dir() and server in (None, "ovan-server"):
        dirs.append(LEGACY_LOG_DIR)
    return dirs


def _all_out_err(logs_root: Path, server: str | None) -> list[Path]:
    files: list[Path] = []
    for d in _iter_log_dirs(logs_root, server):
        files.extend(d.glob("*.out"))
        files.extend(d.glob("*.err"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def cmd_list(logs_root: Path, server: str | None, limit: int) -> int:
    files = _all_out_err(logs_root, server)
    if not files:
        print(
            f"note: no .out/.err under {logs_root}"
            + (f"/{server}" if server else "")
            + f" or {LEGACY_LOG_DIR}; listing other files:",
            file=sys.stderr,
        )
        extras: list[Path] = []
        for d in _iter_log_dirs(logs_root, server):
            extras.extend(p for p in d.iterdir() if p.is_file())
        files = sorted(extras, key=lambda p: p.stat().st_mtime, reverse=True)

    for path in files[:limit]:
        st = path.stat()
        print(f"{st.st_mtime:10.0f}  {st.st_size:12d}  {path}")
    return 0


def _existing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.is_file()]


def _pick_latest_by_name(
    logs_root: Path, server: str | None, job_name: str
) -> Path:
    cands: list[Path] = []
    for d in _iter_log_dirs(logs_root, server):
        cands.extend(d.glob(f"{job_name}-*.out"))
    cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(
            f"no .out for job-name={job_name} under {logs_root}"
            + (f"/{server}" if server else "")
        )
    return cands[0]


def _match_job_id(
    logs_root: Path, server: str | None, job_id: str
) -> tuple[list[Path], list[Path]]:
    """job_id 可为纯数字（Slurm）或 pid12345 / 12345（common）。"""
    keys = {job_id}
    if job_id.startswith("pid") and job_id[3:].isdigit():
        keys.add(job_id[3:])
    elif job_id.isdigit():
        keys.add(f"pid{job_id}")

    outs: list[Path] = []
    errs: list[Path] = []
    for d in _iter_log_dirs(logs_root, server):
        for key in keys:
            outs.extend(d.glob(f"*-{key}.out"))
            errs.extend(d.glob(f"*-{key}.err"))
    # 去重保序
    def uniq(paths: list[Path]) -> list[Path]:
        seen: set[Path] = set()
        out: list[Path] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    return uniq(outs), uniq(errs)


def cmd_tail(
    logs_root: Path,
    server: str | None,
    n: int,
    which: str,
    job_id: str | None,
    job_name: str | None,
) -> int:
    outs: list[Path] = []
    errs: list[Path] = []
    if job_id and job_name:
        for d in _iter_log_dirs(logs_root, server):
            outs.append(d / f"{job_name}-{job_id}.out")
            errs.append(d / f"{job_name}-{job_id}.err")
            if job_id.isdigit():
                outs.append(d / f"{job_name}-pid{job_id}.out")
                errs.append(d / f"{job_name}-pid{job_id}.err")
            elif job_id.startswith("pid"):
                outs.append(d / f"{job_name}-{job_id[3:]}.out")
                errs.append(d / f"{job_name}-{job_id[3:]}.err")
    elif job_id:
        outs, errs = _match_job_id(logs_root, server, job_id)
    else:
        assert job_name is not None
        try:
            latest = _pick_latest_by_name(logs_root, server, job_name)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        outs = [latest]
        errs = [latest.with_suffix(".err")]

    real_outs = _existing(outs)
    real_errs = _existing(errs)

    if len(real_outs) > 1 or len(real_errs) > 1:
        print(f"multiple matches for job_id={job_id}; pass --job-name / --server", file=sys.stderr)
        for p in real_outs + real_errs:
            print(p, file=sys.stderr)
        return 2

    found = False
    targets: list[Path] = []
    if which in ("out", "both"):
        targets.extend(real_outs)
    if which in ("err", "both"):
        targets.extend(real_errs)
    if which == "gpu" and job_id:
        keys = {job_id}
        if job_id.startswith("pid") and job_id[3:].isdigit():
            keys.add(job_id[3:])
        elif job_id.isdigit():
            keys.add(f"pid{job_id}")
        for d in _iter_log_dirs(logs_root, server):
            for key in keys:
                gp = d / f"gpu-{key}.log"
                if gp.is_file():
                    targets.append(gp)
                    break

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
        print(f"no matching log files under {logs_root}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="读取作业 .out/.err（及可选 gpu.log）末 N 行；不发起 SSH。",
    )
    p.add_argument(
        "job_id",
        nargs="?",
        help="Slurm job id 或 common pid / pid<PID>（匹配 *-<id>.{out,err}）",
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
        choices=("both", "out", "err", "gpu"),
        default="both",
        help="读 .out / .err / 两者 / gpu.log（默认 both）",
    )
    p.add_argument(
        "--job-name",
        help="作业名；与 job_id 合用可精确定位，单独使用则取该名前缀最新的 .out/.err",
    )
    p.add_argument(
        "--server",
        help="限制在 logs/<server>/ 下查找（如 ovan-server、train-server-1）",
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
        default=DEFAULT_LOGS_ROOT,
        help=f"日志根目录（默认 {DEFAULT_LOGS_ROOT}；其下为 <server>/<时间戳>/）",
    )
    args = p.parse_args(argv)

    if args.lines < 1:
        p.error("-n/--lines must be >= 1")
    if args.job_id is not None:
        jid = args.job_id
        ok = jid.isdigit() or (jid.startswith("pid") and jid[3:].isdigit())
        if not ok:
            p.error("job_id 须为数字（Slurm）或 pid<PID>（common）")

    logs_root = args.log_dir.expanduser()
    if not logs_root.is_absolute():
        logs_root = (_REPO_ROOT / logs_root).resolve()

    if args.list:
        return cmd_list(logs_root, args.server, args.list_limit)

    if args.job_id is None and not args.job_name:
        p.error("需要 job_id，或 --job-name，或 --list")

    return cmd_tail(
        logs_root=logs_root,
        server=args.server,
        n=args.lines,
        which=args.which,
        job_id=args.job_id,
        job_name=args.job_name,
    )


if __name__ == "__main__":
    sys.exit(main())
