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
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Type

from torch.utils.data import Dataset

from config_util import load_yaml_config

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
        raise FileNotFoundError(f"Config {name}.yaml does not exist")

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
    Non-attribute YAML keys are stored in ``extra``.
    """

    _YAML_REQUIRED = frozenset(
        {"name", "repo_id", "revision", "download_split", "download_path", "split"}
    )

    name: str = "prototype"
    repo_id: str = ""
    revision: str = "main"
    # Optional subset / config name within the repo (e.g. "wikitext-103-v1").
    subset: Optional[str] = None
    # Splits to download (use "|" for multiple, e.g. "train | test").
    download_split: str = "train"
    # Supported splits (may be produced after processing).
    split: str = "train"
    # Local download path. Leave empty to use the HuggingFace Hub default cache.
    download_path: Optional[str] = None
    # Random eval holdout size (arxiv / owt): sample this many rows as eval.
    eval_count: Optional[int] = None
    # Random seed for eval holdout sampling (arxiv / owt).
    eval_seed: Optional[int] = None
    # YAML keys that are not config attributes (e.g. _doc).
    extra: Dict[str, Any] = field(default_factory=dict)

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
        """Load config from a yaml file with strict key validation."""
        return load_yaml_config(cls, path, required=cls._YAML_REQUIRED)


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

    def _snapshot_allow_patterns(self) -> Optional[List[str]]:
        if self.config.subset:
            return [f"{self.config.subset}/*"]
        return None

    def _hf_snapshot_download(self, *, local_files_only: bool):
        import hf_config  # noqa: F401
        from huggingface_hub import snapshot_download

        kwargs = {
            "repo_id": self.config.repo_id,
            "repo_type": "dataset",
            "revision": self.config.revision,
            "allow_patterns": self._snapshot_allow_patterns(),
        }
        if local_files_only:
            kwargs["local_files_only"] = True
        elif self.config.download_path:
            download_path = Path(self.config.download_path)
            download_path.mkdir(parents=True, exist_ok=True)
            kwargs["local_dir"] = str(download_path)

        return snapshot_download(**kwargs)

    def is_downloaded(self) -> bool:
        """Check whether the dataset is available locally."""
        if self._is_prototype():
            return False
        if not self.config.download_path:
            from huggingface_hub.errors import LocalEntryNotFoundError

            try:
                self._hf_snapshot_download(local_files_only=True)
                return True
            except LocalEntryNotFoundError:
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

    def load_split(self, split: str):
        """Load one logical split (e.g. ``train`` / ``eval``) as a HF Dataset."""
        if self._is_prototype():
            raise ValueError("Prototype dataset cannot be loaded.")
        if split not in self.config.splits:
            raise ValueError(
                f"Unknown split '{split}'. Supported: {self.config.splits}"
            )
        return self._build_split(split)

    def _build_split(self, split: str):
        """Build a logical split. Subclasses may override for custom mapping."""
        if self.config.eval_count is not None:
            return self._build_random_holdout_split(split)
        raise NotImplementedError(
            f"{type(self).__name__} does not implement split mapping."
        )

    def _build_random_holdout_split(self, split: str):
        """Sample ``eval_count`` rows as eval; the rest become train."""
        if self.config.eval_count is None or self.config.eval_seed is None:
            raise ValueError(
                f"{self.config.name}: eval_count and eval_seed are required "
                "for random holdout splits."
            )
        if self.config.eval_count < 0:
            raise ValueError(
                f"{self.config.name}: eval_count must be non-negative."
            )

        parts = self._get_holdout_parts()
        if split == "train":
            return parts["train"]
        if split == "eval":
            return parts["test"]
        raise ValueError(f"Unknown split '{split}'.")

    def _get_holdout_parts(self):
        """Return cached train/test split from the raw train split."""
        cached = getattr(self, "_holdout_parts", None)
        if cached is not None:
            return cached

        full = self._load_raw_split("train")
        eval_count = min(self.config.eval_count, len(full))
        if eval_count <= 0:
            parts = {"train": full, "test": full.select([])}
        else:
            parts = full.train_test_split(
                test_size=eval_count,
                seed=self.config.eval_seed,
                shuffle=True,
            )
        self._holdout_parts = parts
        return parts

    def _local_parquet_files(self, data_root: Path, hf_split: str) -> List[Path]:
        plain_text = data_root / "plain_text"
        if plain_text.is_dir():
            files = sorted(plain_text.glob(f"{hf_split}-*.parquet"))
            if files:
                return files
        return sorted(data_root.rglob(f"{hf_split}-*.parquet"))

    def _load_raw_split(self, hf_split: str):
        """Load one HuggingFace split name without remapping."""
        import hf_config  # noqa: F401
        from datasets import load_dataset

        if self.config.download_path:
            data_root = Path(self.config.download_path)
            if data_root.is_dir():
                files = self._local_parquet_files(data_root, hf_split)
                if files:
                    workers = max(1, os.cpu_count() or 1)
                    return load_dataset(
                        "parquet",
                        data_files={hf_split: [str(path) for path in files]},
                        split=hf_split,
                        num_proc=workers,
                    )

        kwargs: Dict[str, Any] = {
            "path": self.config.repo_id,
            "split": hf_split,
            "revision": self.config.revision,
        }
        if self.config.subset:
            kwargs["name"] = self.config.subset
        if self.config.download_path:
            kwargs["data_dir"] = str(self.config.download_path)

        return load_dataset(**kwargs)

    def can_stream_parquet(self, hf_split: str = "train") -> bool:
        if not self.config.download_path:
            return False
        data_root = Path(self.config.download_path)
        return bool(self._local_parquet_files(data_root, hf_split))

    def count_raw_rows(self, hf_split: str = "train") -> int:
        import pyarrow.parquet as pq

        if not self.config.download_path:
            raise FileNotFoundError(
                f"{self.config.name}: download_path is required to count raw rows."
            )
        files = self._local_parquet_files(Path(self.config.download_path), hf_split)
        if not files:
            raise FileNotFoundError(
                f"{self.config.name}: no local parquet files for split {hf_split!r}."
            )
        return sum(pq.ParquetFile(path).metadata.num_rows for path in files)

    def holdout_eval_indices(self) -> frozenset[int]:
        if self.config.eval_count is None or self.config.eval_seed is None:
            return frozenset()
        cached = getattr(self, "_holdout_eval_indices", None)
        if cached is not None:
            return cached

        total = self.count_raw_rows("train")
        eval_count = min(self.config.eval_count, total)
        rng = random.Random(self.config.eval_seed)
        eval_set = frozenset(rng.sample(range(total), eval_count))
        self._holdout_eval_indices = eval_set
        return eval_set

    def iter_parquet_rows(
        self,
        *,
        text_column: str = "text",
        read_row_batch: int = 2048,
        hf_split: str = "train",
    ) -> Iterator[tuple[int, str | None]]:
        """Yield ``(row_index, stripped_text_or_none)`` in parquet file order."""
        import pyarrow.parquet as pq

        if not self.config.download_path:
            raise FileNotFoundError(
                f"{self.config.name}: download_path is required for parquet streaming."
            )
        files = self._local_parquet_files(Path(self.config.download_path), hf_split)
        if not files:
            raise FileNotFoundError(
                f"{self.config.name}: no local parquet files for split {hf_split!r}."
            )

        row_index = 0
        for parquet_path in files:
            parquet_file = pq.ParquetFile(parquet_path)
            for record_batch in parquet_file.iter_batches(
                batch_size=read_row_batch,
                columns=[text_column],
            ):
                for text in record_batch.column(text_column).to_pylist():
                    stripped = str(text).strip() if text else None
                    if not stripped:
                        stripped = None
                    yield row_index, stripped
                    row_index += 1

    def iter_split_texts(
        self,
        split: str,
        *,
        text_column: str = "text",
        read_row_batch: int = 2048,
        hf_split: str = "train",
    ) -> Iterator[str]:
        """Sequentially read texts from local parquet files for one logical split."""
        eval_indices = self.holdout_eval_indices()
        for row_index, text in self.iter_parquet_rows(
            text_column=text_column,
            read_row_batch=read_row_batch,
            hf_split=hf_split,
        ):
            in_eval = row_index in eval_indices
            if split == "eval":
                if not in_eval:
                    continue
            elif split == "train":
                if in_eval:
                    continue
            else:
                raise ValueError(f"Unknown split '{split}'.")
            if text:
                yield text

    def ensure_downloaded(self) -> None:
        """Ensure the dataset exists locally, downloading from the Hub if needed."""
        if self._is_prototype():
            raise ValueError("Prototype dataset cannot be downloaded.")
        if self.is_downloaded():
            return
        print(f"Dataset '{self.config.name}' not found locally; downloading...")
        self._hf_snapshot_download(local_files_only=False)

    def download(self) -> bool:
        """Download the dataset locally.

        - If a local download already exists, print "Already downloaded" and return ``True``.
        - Otherwise download from the HuggingFace Hub. When ``download_path`` is
          empty, files go to the HuggingFace Hub default cache; otherwise into
          ``config.download_path``.
          Global HuggingFace settings (disable XET, use the hf-mirror endpoint)
          are applied by importing ``hf_config`` before importing ``huggingface_hub``.
        """
        if self._is_prototype():
            raise ValueError("Prototype dataset cannot be downloaded.")
        if self.is_downloaded():
            print("Already downloaded")
            return True

        self._hf_snapshot_download(local_files_only=False)
        return True
