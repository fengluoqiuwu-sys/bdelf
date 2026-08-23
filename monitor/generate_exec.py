"""本机 generate 子进程调度（FastAPI 侧，不 import torch）。"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from monitor.agent_local import finish_local_gpu_job, register_local_gpu_job
from monitor.generate_meta import query_gpu_memory

_LOCK = threading.Lock()
_TIMEOUT_PER_SAMPLE = 20 * 60
_TIMEOUT_MAX = 180 * 60


def _timeout_sec(num_samples: int) -> int:
    n = max(1, int(num_samples or 1))
    return min(_TIMEOUT_MAX, max(_TIMEOUT_PER_SAMPLE, n * _TIMEOUT_PER_SAMPLE))


def _event_line(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def iter_local_generate(repo_root: Path, spec: dict[str, Any]) -> Iterator[str]:
    """子进程 stdout 为 NDJSON：sample →（全部生成完）eval → done。"""
    gpu = query_gpu_memory()
    if not gpu.get("ok"):
        yield _event_line({
            "type": "error",
            "error": gpu.get("reason") or "显存检查未通过",
            "gpu": gpu,
        })
        return

    if not _LOCK.acquire(blocking=False):
        yield _event_line({"type": "error", "error": "已有生成任务在跑，请等它结束"})
        return

    work = repo_root / "temp" / "monitor-generate"
    work.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    spec_path = work / f"spec-{stamp}.json"
    out_path = work / f"out-{stamp}.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "monitor.generate_run",
        "--spec",
        str(spec_path),
        "--out",
        str(out_path),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc: subprocess.Popen[str] | None = None
    stderr_buf: list[str] = []
    finished = False
    saw_terminal = False
    timeout = _timeout_sec(int(spec.get("num_samples") or 1))

    def _finish(pid: int, state: str) -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        finish_local_gpu_job(repo_root, pid, state=state)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        register_local_gpu_job(
            repo_root,
            pid=proc.pid,
            job_name="monitor-generate",
            cmdline=" ".join(cmd),
        )
        assert proc.stdout is not None and proc.stderr is not None

        def _drain_err() -> None:
            try:
                stderr_buf.append(proc.stderr.read() or "")
            except Exception:
                pass

        err_thread = threading.Thread(target=_drain_err, daemon=True)
        err_thread.start()

        deadline = time.monotonic() + timeout
        timed_out = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([proc.stdout], [], [], min(1.0, remaining))
            if ready:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                stripped = line.strip()
                if not stripped.startswith("{"):
                    continue
                try:
                    ev = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict) and ev.get("type") in ("error", "done"):
                    saw_terminal = True
                yield stripped + "\n"
            elif proc.poll() is not None:
                rest = proc.stdout.read() or ""
                for extra in rest.splitlines():
                    extra = extra.strip()
                    if not extra.startswith("{"):
                        continue
                    try:
                        ev = json.loads(extra)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(ev, dict) and ev.get("type") in ("error", "done"):
                        saw_terminal = True
                    yield extra + "\n"
                break

        if timed_out:
            proc.kill()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pass
            _finish(proc.pid, "TIMEOUT")
            yield _event_line({"type": "error", "error": f"生成超时（>{timeout // 60} 分钟）"})
            return

        rc = proc.wait(timeout=30)
        err_thread.join(timeout=2)
        stderr = "".join(stderr_buf)
        _finish(proc.pid, "DONE" if rc == 0 else "FAILED")
        if rc != 0 and not saw_terminal:
            msg = stderr.strip()[-2000:] if stderr.strip() else f"exit {rc}"
            yield _event_line({
                "type": "error",
                "error": msg,
                "log_tail": stderr[-2000:] if stderr else "",
            })
    except GeneratorExit:
        if proc is not None and proc.poll() is None:
            proc.kill()
        if proc is not None:
            _finish(proc.pid, "FAILED")
        raise
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            proc.kill()
        if proc is not None:
            _finish(proc.pid, "FAILED")
        yield _event_line({"type": "error", "error": str(exc)})
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=8)
            except Exception:
                pass
            _finish(proc.pid, "FAILED")
        _LOCK.release()
