#!/usr/bin/env python3
"""本机经 SSH 读取远端 Slurm 训练 .out / .err 末 N 行（轻量只读，不占用 GPU）。

日志命名与 ``#SBATCH --output/--error`` 一致：``<job-name>-<job-id>.{out,err}``，
默认目录：``~/source/bdelf/slurm/logs``（与 sync-ovan-server 远端树一致）。

远端只执行 ``ls`` / ``tail`` 等轻量命令，不在登录节点启 Python。

示例::

    .venv/bin/python slurm/tail_remote_logs.py 1234567
    .venv/bin/python slurm/tail_remote_logs.py 1234567 -n 120
    .venv/bin/python slurm/tail_remote_logs.py 1234567 --which err
    .venv/bin/python slurm/tail_remote_logs.py --list
    .venv/bin/python slurm/tail_remote_logs.py --job-name ar-100m-full -n 50
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import textwrap


DEFAULT_HOST = "ovan-server"
DEFAULT_LOG_DIR = "~/source/bdelf/slurm/logs"
DEFAULT_N = 80


def _ssh(host: str, remote_script: str) -> int:
    return subprocess.run(["ssh", host, "bash", "-s"], input=remote_script, text=True).returncode


def _quote_log_dir(path: str) -> str:
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    if path == "~":
        return "~"
    return shlex.quote(path)


def cmd_list(host: str, log_dir: str, limit: int) -> int:
    d = _quote_log_dir(log_dir)
    script = textwrap.dedent(
        f"""\
        set -euo pipefail
        d={d}
        if [[ ! -d "$d" ]]; then
          echo "missing log dir: $d" >&2
          exit 1
        fi
        shopt -s nullglob
        files=("$d"/*.out "$d"/*.err)
        if (( ${{#files[@]}} > 0 )); then
          ls -lt -- "${{files[@]}}" | head -n {int(limit)}
        else
          echo "note: no .out/.err yet under $d; listing all files:" >&2
          ls -lt -- "$d" | head -n {int(limit)}
        fi
        """
    )
    return _ssh(host, script)


def cmd_tail(
    host: str,
    log_dir: str,
    n: int,
    which: str,
    job_id: str | None,
    job_name: str | None,
) -> int:
    d = _quote_log_dir(log_dir)
    jid_q = shlex.quote(job_id) if job_id else "''"
    name_q = shlex.quote(job_name) if job_name else "''"
    want_out = 1 if which in ("out", "both") else 0
    want_err = 1 if which in ("err", "both") else 0

    script = textwrap.dedent(
        f"""\
        set -euo pipefail
        d={d}
        jid={jid_q}
        name={name_q}
        n={int(n)}
        want_out={want_out}
        want_err={want_err}

        if [[ ! -d "$d" ]]; then
          echo "missing log dir: $d" >&2
          exit 1
        fi
        shopt -s nullglob

        outs=()
        errs=()
        if [[ -n "$jid" && -n "$name" ]]; then
          outs=("$d/${{name}}-${{jid}}.out")
          errs=("$d/${{name}}-${{jid}}.err")
        elif [[ -n "$jid" ]]; then
          outs=("$d"/*-"$jid".out)
          errs=("$d"/*-"$jid".err)
        else
          cands=("$d"/"$name"-*.out)
          if (( ${{#cands[@]}} == 0 )); then
            echo "no .out for job-name=$name under $d" >&2
            exit 1
          fi
          latest=""
          newest=0
          for f in "${{cands[@]}}"; do
            mt=$(stat -c %Y -- "$f" 2>/dev/null || stat -f %m -- "$f")
            if (( mt >= newest )); then
              newest=$mt
              latest=$f
            fi
          done
          outs=("$latest")
          errs=("${{latest%.out}}.err")
        fi

        # 过滤不存在的路径；job_id 多匹配时要求 --job-name
        real_outs=()
        for f in "${{outs[@]}}"; do
          [[ -e "$f" ]] && real_outs+=("$f")
        done
        real_errs=()
        for f in "${{errs[@]}}"; do
          [[ -e "$f" ]] && real_errs+=("$f")
        done

        if (( ${{#real_outs[@]}} > 1 || ${{#real_errs[@]}} > 1 )); then
          echo "multiple matches for job_id=$jid; pass --job-name" >&2
          printf '%s\\n' "${{real_outs[@]}}" "${{real_errs[@]}}" >&2
          exit 2
        fi

        found=0
        if (( want_out )); then
          for f in "${{real_outs[@]}}"; do
            found=1
            printf '===== %s (last %s lines) =====\\n' "$f" "$n"
            tail -n "$n" -- "$f"
            printf '\\n'
          done
        fi
        if (( want_err )); then
          for f in "${{real_errs[@]}}"; do
            found=1
            printf '===== %s (last %s lines) =====\\n' "$f" "$n"
            tail -n "$n" -- "$f"
            printf '\\n'
          done
        fi
        if (( found == 0 )); then
          echo "no matching log files under $d" >&2
          exit 1
        fi
        """
    )
    return _ssh(host, script)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Tail last N lines of remote Slurm .out/.err via SSH.",
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
        help="作业名（#SBATCH --job-name / %%x）；与 job_id 合用可精确定位，"
        "单独使用则取该名前缀最新的 .out/.err",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="列出远端 logs 目录最近文件后退出",
    )
    p.add_argument(
        "--list-limit",
        type=int,
        default=30,
        help="--list 时最多显示行数（默认 30）",
    )
    p.add_argument("--host", default=DEFAULT_HOST, help=f"SSH 主机（默认 {DEFAULT_HOST}）")
    p.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"远端日志目录（默认 {DEFAULT_LOG_DIR}）",
    )
    args = p.parse_args(argv)

    if args.lines < 1:
        p.error("-n/--lines must be >= 1")
    if args.job_id is not None and not args.job_id.isdigit():
        p.error("job_id must be numeric")

    if args.list:
        return cmd_list(args.host, args.log_dir, args.list_limit)

    if args.job_id is None and not args.job_name:
        p.error("需要 job_id，或 --job-name，或 --list")

    return cmd_tail(
        host=args.host,
        log_dir=args.log_dir,
        n=args.lines,
        which=args.which,
        job_id=args.job_id,
        job_name=args.job_name,
    )


if __name__ == "__main__":
    sys.exit(main())
