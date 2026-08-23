#!/usr/bin/env python3
"""本地训练 / 评测监控站入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(description="bdelf 本机训练 / 评测 / 生成监控站（只读 cache/；生成为本机 GPU）")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址（默认 127.0.0.1）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="监听端口（默认复用上次成功端口，不可用再随机 16385–65535）",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=root,
        help="仓库根目录（默认本脚本所在目录）",
    )
    parser.add_argument(
        "--instance",
        choices=("local", "remote"),
        default=None,
        help="写入 cache/monitor/instance.json（默认：文件已有则沿用，没有则生成本机 local）",
    )
    args = parser.parse_args(argv)

    from monitor.app import create_app
    from monitor.port import pick_port, read_last_port, write_last_port

    repo_root = args.root.resolve()
    slot = args.instance or "local"
    if args.port is not None:
        port = args.port
    else:
        port = pick_port(args.host, prefer=read_last_port(repo_root, slot))
    write_last_port(repo_root, slot, port)
    if port < 16385:
        print(f"警告：端口 {port} < 16385，建议使用高端口", file=sys.stderr)

    app = create_app(repo_root, instance_role=args.instance)
    url = f"http://{args.host}:{port}"
    print(url, flush=True)

    import uvicorn

    uvicorn.run(app, host=args.host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
