"""监控实例标识：本机 local / 远端 remote。落在 cache/monitor（gitignore），不进 git。"""

from __future__ import annotations

import json
from pathlib import Path

from monitor.config import MONITOR_INSTANCE

VALID_ROLES = frozenset({"local", "remote"})


def instance_path(repo_root: Path) -> Path:
    return repo_root / MONITOR_INSTANCE


def _write_role(path: Path, role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"role": role}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_instance_role(repo_root: Path) -> str:
    path = instance_path(repo_root)
    if not path.is_file():
        return "local"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "local"
    role = str((data or {}).get("role") or "").strip()
    return role if role in VALID_ROLES else "local"


def ensure_instance_role(repo_root: Path, override: str | None = None) -> str:
    """首次无文件则写成 local。override 为 local|remote 时强制写入。"""
    path = instance_path(repo_root)
    if override:
        role = str(override).strip()
        if role not in VALID_ROLES:
            raise ValueError(f"instance role 必须是 local 或 remote，收到 {override!r}")
        _write_role(path, role)
        return role
    if path.is_file():
        return read_instance_role(repo_root)
    _write_role(path, "local")
    return "local"
