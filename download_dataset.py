"""Download one or more datasets by name.

Usage:
    python download_dataset.py <name>
    python download_dataset.py <name1>|<name2>|...

Multiple names can be separated by "|" (no spaces). For each name, looks up
``config/datasets/<name>.yaml``. If it exists (and is not the ``prototype``
template), the dataset is downloaded; otherwise it reports that the config
does not exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import get_dataset

CONFIG_DIR = Path(__file__).resolve().parent / "config" / "datasets"


def download_by_name(name: str) -> bool:
    """Download the dataset identified by ``name``.

    Returns ``True`` on success, ``False`` if the config does not exist
    (or is the ``prototype`` template).
    """
    if name == "prototype":
        print(f"配置 {name}.yaml 不存在")
        return False

    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        print(f"配置 {name}.yaml 不存在")
        return False

    dataset = get_dataset(name)
    dataset.download()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one or more datasets by name.")
    parser.add_argument(
        "names",
        help='dataset name(s), separated by "|" without spaces (e.g. owt|wikitext)',
    )
    args = parser.parse_args()

    for name in (part.strip() for part in args.names.split("|")):
        if not name:
            continue
        download_by_name(name)


if __name__ == "__main__":
    main()
