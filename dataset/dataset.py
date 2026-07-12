"""Generic dataset template.

Contains:
- ``FL_DatasetConfig``: dataset config, loadable from a yaml under ``config/datasets``.
- ``FL_Dataset``: a generic template inheriting ``torch.utils.data.Dataset``.
  The underlying data comes from the HuggingFace Hub, with a ``download`` method
  that fetches the data locally.
- ``register_dataset`` / ``get_dataset``: registry-based factory. Datasets are
  never instantiated directly; use ``get_dataset(name)`` to obtain an instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Type

import yaml
from torch.utils.data import Dataset

# Directory holding the per-dataset yaml configs.
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "datasets"

# Registry mapping dataset name -> concrete FL_Dataset subclass.
_DATASET_REGISTRY: Dict[str, Type["FL_Dataset"]] = {}


def register_dataset(name: str, cls: Type["FL_Dataset"]) -> None:
    """Bind ``name`` to a concrete ``FL_Dataset`` subclass in the registry."""
    if not (isinstance(cls, type) and issubclass(cls, FL_Dataset)):
        raise TypeError(f"{cls!r} is not an FL_Dataset subclass.")
    _DATASET_REGISTRY[name] = cls


def list_datasets() -> List[str]:
    """Return the names of all registered datasets."""
    return sorted(_DATASET_REGISTRY)


def get_dataset(name: str) -> "FL_Dataset":
    """Get a dataset instance by name.

    Loads ``config/datasets/<name>.yaml`` and instantiates the registered
    subclass. This is the only supported way to create a dataset instance.
    """
    cls = _DATASET_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(list_datasets()) or "<none>"
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")

    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置 {name}.yaml 不存在")

    config = FL_DatasetConfig.from_yaml(config_path)
    return cls(config)


def _parse_splits(value: str | List[str] | None) -> List[str]:
    """Parse a split spec into a list of split names.

    Accepts a list, or a string using "|" as separator, e.g. "train | test".
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split("|") if part.strip()]


@dataclass
class FL_DatasetConfig:
    """Generic dataset config.

    Fields map one-to-one with the yaml template under ``config/datasets``.
    """

    name: str = "prototype"
    repo_id: str = ""
    revision: str = "main"
    # Optional subset / config name within the repo (e.g. "wikitext-103-v1").
    subset: Optional[str] = None
    # Splits to download (use "|" for multiple, e.g. "train | test").
    download_split: str = "train"
    # Supported splits (may be produced after processing).
    split: str = "train"
    # Local download path. Defaults to "cache/datasets/{name}" when empty.
    download_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.download_path:
            self.download_path = f"cache/datasets/{self.name}"

    @property
    def download_splits(self) -> List[str]:
        """Splits to download, parsed into a list."""
        return _parse_splits(self.download_split)

    @property
    def splits(self) -> List[str]:
        """Supported splits, parsed into a list."""
        return _parse_splits(self.split)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "FL_DatasetConfig":
        """Load config from a yaml file; unknown fields are ignored."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        valid_keys = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in valid_keys}
        return cls(**kwargs)


class FL_Dataset(Dataset):
    """Generic dataset template.

    Meant to be subclassed. The underlying dataset comes from the HuggingFace
    Hub. Subclass this, register the subclass with :func:`register_dataset`,
    and obtain instances via :func:`get_dataset`.
    """

    def __init__(self, config: FL_DatasetConfig) -> None:
        super().__init__()
        self.config = config

    def _is_prototype(self) -> bool:
        """Check if the dataset is the prototype dataset."""
        return self.config and self.config.name == "prototype"

    def is_downloaded(self) -> bool:
        """Check whether the local download path exists and is non-empty."""
        if self._is_prototype():
            return False
        path = Path(self.config.download_path)
        if not path.exists():
            return False
        # An existing but empty directory is treated as not downloaded.
        if path.is_dir():
            return any(path.iterdir())
        return True

    def get_splits(self) -> List[str]:
        """Return the list of supported splits."""
        if self._is_prototype():
            return []
        return self.config.splits

    def download(self) -> bool:
        """Download the dataset locally.

        - If a local download already exists, print "已下载" and return ``True``.
        - Otherwise download from the HuggingFace Hub into ``config.download_path``.
          Global HuggingFace settings (disable XET, use the hf-mirror endpoint)
          are applied by importing ``hf_config`` before importing ``huggingface_hub``.
        """
        if self._is_prototype():
            raise ValueError("Prototype dataset cannot be downloaded.")
        if self.is_downloaded():
            print("已下载")
            return True

        # Import hf_config first to apply global HF settings, then import the
        # huggingface_hub package so the mirror and XET settings take effect.
        import hf_config  # noqa: F401
        from huggingface_hub import snapshot_download

        download_path = Path(self.config.download_path)
        download_path.mkdir(parents=True, exist_ok=True)

        # When a subset is set, only download files under that subset folder.
        allow_patterns = None
        if self.config.subset:
            allow_patterns = [f"{self.config.subset}/*"]

        snapshot_download(
            repo_id=self.config.repo_id,
            repo_type="dataset",
            revision=self.config.revision,
            local_dir=str(download_path),
            allow_patterns=allow_patterns,
        )
        return True
