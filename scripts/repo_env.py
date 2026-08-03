"""把进程的工作目录与 ``sys.path`` 固定到仓库根（``scripts/`` 的上一级）。

所有 ``scripts/`` 下的辅助脚本应在导入项目包之前调用 ``ensure_repo_root()``。
约定：从仓库根执行，例如 ``.venv/bin/python scripts/<name>.py``。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_root() -> Path:
    """``chdir`` 到仓库根，并把根目录插入 ``sys.path[0]``。"""
    root = REPO_ROOT
    if Path.cwd().resolve() != root:
        os.chdir(root)
    root_s = str(root)
    if sys.path[:1] != [root_s]:
        if root_s in sys.path:
            sys.path.remove(root_s)
        sys.path.insert(0, root_s)
    return root
