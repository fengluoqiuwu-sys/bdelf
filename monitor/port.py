"""随机高端口选择。"""

from __future__ import annotations

import random
import socket

from monitor.config import PORT_BIND_TRIES, PORT_MAX, PORT_MIN


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(host: str = "127.0.0.1") -> int:
    for _ in range(PORT_BIND_TRIES):
        port = random.randint(PORT_MIN, PORT_MAX)
        if port_available(host, port):
            return port
    raise RuntimeError(
        f"无法在 [{PORT_MIN}, {PORT_MAX}] 内找到可用端口（已试 {PORT_BIND_TRIES} 次）",
    )
