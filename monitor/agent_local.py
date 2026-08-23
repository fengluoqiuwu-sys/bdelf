"""本机 GPU 作业登记（temp/agent，scheduler=local）。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

_TZ = timezone(timedelta(hours=8))


def _agent_dirs(repo_root: Path) -> tuple[Path, Path]:
    active = repo_root / "temp" / "agent" / "active"
    launched = repo_root / "temp" / "agent" / "launched"
    active.mkdir(parents=True, exist_ok=True)
    launched.mkdir(parents=True, exist_ok=True)
    return active, launched


def _now_iso() -> str:
    return datetime.now(_TZ).isoformat(timespec="seconds")


def _dead_pid(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


def sweep_dead_local_jobs(repo_root: Path) -> None:
    active, _ = _agent_dirs(repo_root)
    for path in list(active.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("scheduler") != "local":
            continue
        pid = data.get("pid")
        if not isinstance(pid, int) or _dead_pid(pid):
            path.unlink(missing_ok=True)


def register_local_gpu_job(
    repo_root: Path,
    *,
    pid: int,
    job_name: str,
    cmdline: str,
    holder: str = "monitor:generate",
) -> dict[str, Any]:
    sweep_dead_local_jobs(repo_root)
    active, launched = _agent_dirs(repo_root)
    job_id = f"pid{pid}"
    rec = {
        "job_id": job_id,
        "pid": pid,
        "job_name": job_name,
        "script": "monitor/generate_run.py",
        "cmdline": cmdline,
        "cpus": 1,
        "gpus": 1,
        "gpu_ids": [0],
        "started_at": _now_iso(),
        "state": "RUNNING",
        "holder": holder,
        "scheduler": "local",
    }
    text = json.dumps(rec, ensure_ascii=False, indent=2) + "\n"
    (active / f"{job_id}.json").write_text(text, encoding="utf-8")
    (launched / f"{job_id}.json").write_text(text, encoding="utf-8")
    return rec


def finish_local_gpu_job(repo_root: Path, pid: int, *, state: str = "DONE") -> None:
    active, launched = _agent_dirs(repo_root)
    job_id = f"pid{pid}"
    active_path = active / f"{job_id}.json"
    launched_path = launched / f"{job_id}.json"
    rec: dict[str, Any] = {}
    if launched_path.is_file():
        try:
            rec = json.loads(launched_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rec = {}
    rec.update({"state": state, "finished_at": _now_iso()})
    launched_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    active_path.unlink(missing_ok=True)
