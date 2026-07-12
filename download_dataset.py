"""Download one or more datasets by name.

Usage:
    python download_dataset.py <name>
    python download_dataset.py <name1> <name2> ...
    python download_dataset.py <name1>|<name2>|...   # quote when using |

Multiple names can be passed as separate arguments or joined with "|" in one
quoted argument. For each name, looks up ``config/datasets/<name>.yaml``. If it
exists (and is not the ``prototype`` template), the dataset is downloaded;
otherwise it reports that the config does not exist.
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
        print(f"Config {name}.yaml does not exist")
        return False

    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        print(f"Config {name}.yaml does not exist")
        return False

    dataset = get_dataset(name)
    dataset.download()
    return True


def _expand_names(raw_names: list[str]) -> list[str]:
    """Flatten CLI names; each argument may use ``|`` as an inner separator."""
    names: list[str] = []
    for raw in raw_names:
        for part in raw.split("|"):
            name = part.strip()
            if name:
                names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one or more datasets by name.")
    parser.add_argument(
        "names",
        nargs="+",
        help='dataset name(s), e.g. owt arxiv wikitext (or "owt|arxiv|wikitext")',
    )
    args = parser.parse_args()

    for name in _expand_names(args.names):
        download_by_name(name)


if __name__ == "__main__":
    main()
