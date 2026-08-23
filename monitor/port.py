"""随机高端口选择；优先复用上次成功绑定的端口。"""

from __future__ import annotations

import random
import socket
from pathlib import Path

from monitor.config import PORT_BIND_TRIES, PORT_MAX, PORT_MIN

_LAST_PORT_DIR = Path("temp") / "web"


def port_listening(host: str, port: int) -> bool:
    """是否已有进程在听这个端口（TIME_WAIT 不算占用）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            sock.connect((host, port))
        except OSError:
            return False
    return True


def port_available(host: str, port: int) -> bool:
    if port_listening(host, port):
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def last_port_path(repo_root: Path, name: str) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(name))
    if not safe:
        safe = "default"
    return Path(repo_root) / _LAST_PORT_DIR / f"{safe}.last_port"


def read_last_port(repo_root: Path, name: str) -> int | None:
    path = last_port_path(repo_root, name)
    try:
        text = path.read_text(encoding="utf-8").strip()
        port = int(text)
    except (OSError, ValueError):
        return None
    if PORT_MIN <= port <= PORT_MAX:
        return port
    return None


def write_last_port(repo_root: Path, name: str, port: int) -> None:
    path = last_port_path(repo_root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{int(port)}\n", encoding="utf-8")


def pick_port(host: str = "127.0.0.1", prefer: int | None = None) -> int:
    """上次端口仍空闲则复用，否则在范围内随机。"""
    if prefer is not None:
        try:
            port = int(prefer)
        except (TypeError, ValueError):
            port = None
        else:
            if PORT_MIN <= port <= PORT_MAX and not port_listening(host, port):
                return port
    for _ in range(PORT_BIND_TRIES):
        port = random.randint(PORT_MIN, PORT_MAX)
        if port_available(host, port):
            return port
    raise RuntimeError(
        f"无法在 [{PORT_MIN}, {PORT_MAX}] 内找到可用端口（已试 {PORT_BIND_TRIES} 次）",
    )
