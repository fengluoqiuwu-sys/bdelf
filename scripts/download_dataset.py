"""按名称下载一个或多个数据集。

用法（工作目录为仓库根）::

    .venv/bin/python scripts/download_dataset.py <name>
    .venv/bin/python scripts/download_dataset.py <name1> <name2> ...
    .venv/bin/python scripts/download_dataset.py '<name1>|<name2>|...'

对每个名字查找 ``config/datasets/<name>.yaml``；存在且非 ``prototype`` 则下载。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import repo_env

repo_env.ensure_repo_root()

from dataset import get_dataset

CONFIG_DIR = repo_env.REPO_ROOT / "config" / "datasets"


def download_by_name(name: str) -> bool:
    """下载名为 ``name`` 的数据集；配置不存在时返回 ``False``。"""
    if name == "prototype":
        print(f"[download] Config {name}.yaml does not exist")
        return False

    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        print(f"[download] Config {name}.yaml does not exist")
        return False

    dataset = get_dataset(name)
    dataset.download()
    return True


def _expand_names(raw_names: list[str]) -> list[str]:
    """展开 CLI 参数；单个参数内可用 ``|`` 分隔多个名字。"""
    names: list[str] = []
    for raw in raw_names:
        for part in raw.split("|"):
            name = part.strip()
            if name:
                names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="按名称下载一个或多个数据集。")
    parser.add_argument(
        "names",
        nargs="+",
        help='数据集名，如 owt arxiv wikitext（或 "owt|arxiv|wikitext"）',
    )
    args = parser.parse_args()

    for name in _expand_names(args.names):
        download_by_name(name)


if __name__ == "__main__":
    main()
