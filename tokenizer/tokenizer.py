"""Generic tokenizer template.

Contains:
- ``FL_TokenizerConfig``: tokenizer config, loadable from a yaml under ``config/tokenizers``.
- ``FL_Tokenizer``: inherits the HuggingFace base tokenizer class (e.g. ``GPT2TokenizerFast``)
  for ``base_tokenizer``, with optional special tokens added on top.
- ``get_tokenizer``: factory that loads ``config/tokenizers/<name>.yaml`` and returns
  an ``FL_Tokenizer`` instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from config_util import load_yaml_config

# Directory holding the per-tokenizer yaml configs.
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "tokenizers"

# Cached FL_Tokenizer class per HuggingFace base tokenizer class.
_FL_TOKENIZER_CLASSES: Dict[str, Type["FL_Tokenizer"]] = {}


def list_tokenizers() -> List[str]:
    """Return tokenizer names discovered from ``config/tokenizers/*.yaml``."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(
        path.stem
        for path in CONFIG_DIR.glob("*.yaml")
        if path.stem != "prototype"
    )


def get_tokenizer(name: str) -> "FL_Tokenizer":
    """Get a tokenizer instance by name.

    Loads ``config/tokenizers/<name>.yaml`` and returns an ``FL_Tokenizer``.
    """
    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        available = ", ".join(list_tokenizers()) or "<none>"
        raise FileNotFoundError(
            f"Config {name}.yaml does not exist. Available: {available}"
        )

    config = FL_TokenizerConfig.from_yaml(config_path)
    return FL_Tokenizer(config)


def _parse_special_tokens(value: Union[str, List[str], None]) -> List[str]:
    """Parse a special-token spec into a list of token strings.

    Accepts a YAML list, or a comma-separated string, e.g. "<|pad|>, <|bos|>".
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _infer_embed_size(model_config: Any) -> Optional[int]:
    """Try to read embedding / hidden size from a HuggingFace model config."""
    for attr in ("hidden_size", "n_embd", "d_model", "embed_dim"):
        size = getattr(model_config, attr, None)
        if size is not None:
            return int(size)
    return None


@dataclass
class FL_TokenizerConfig:
    """Generic tokenizer config.

    Fields map one-to-one with the yaml template under ``config/tokenizers``.
    Non-attribute YAML keys are stored in ``extra``.
    """

    _YAML_REQUIRED = frozenset(
        {"name", "base_tokenizer", "base_tokenizer_cache_path", "special_tokens"}
    )

    name: str = "prototype"
    # HuggingFace tokenizer name or local path (default: gpt2).
    base_tokenizer: str = "gpt2"
    # Local cache path for the base tokenizer. Defaults to "cache/tokenizers/{base_tokenizer}" when empty.
    base_tokenizer_cache_path: Optional[str] = None
    # Extra special tokens to add on top of the base tokenizer.
    special_tokens: List[str] = field(default_factory=list)
    # YAML keys that are not config attributes (e.g. _doc).
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.special_tokens = _parse_special_tokens(self.special_tokens)
        if not self.base_tokenizer_cache_path:
            self.base_tokenizer_cache_path = f"cache/tokenizers/{self.base_tokenizer}"

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "FL_TokenizerConfig":
        """Load config from a yaml file with strict key validation."""
        return load_yaml_config(cls, path, required=cls._YAML_REQUIRED)


class _FLTokenizerMixin:
    """Shared methods mixed into the HF tokenizer subclass."""

    config: FL_TokenizerConfig
    _embed_size: Optional[int] = None

    def get_vocab_size(self) -> int:
        """Return the vocabulary size (after special tokens are added)."""
        return len(self)

    def get_embed_size(self) -> int:
        """Return the model embedding / hidden size derived from the base model."""
        if self._embed_size is None:
            raise ValueError(
                f"Cannot infer embed_size from base_tokenizer "
                f"'{self.config.base_tokenizer}'."
            )
        return self._embed_size

    def save(self, path: Optional[str] = None) -> None:
        """Save the tokenizer to a local directory."""
        save_path = path or self.config.base_tokenizer_cache_path
        if save_path is None:
            raise ValueError("No save path specified.")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.save_pretrained(save_path)

    def is_saved(self) -> bool:
        """Check whether the tokenizer has been saved locally."""
        path = Path(self.config.base_tokenizer_cache_path)
        return path.exists() and (path / "tokenizer_config.json").exists()


def _get_fl_tokenizer_class(base_cls: type) -> Type["FL_Tokenizer"]:
    """Return a cached ``FL_Tokenizer`` class inheriting from ``base_cls``."""
    key = f"{base_cls.__module__}.{base_cls.__name__}"
    if key in _FL_TOKENIZER_CLASSES:
        return _FL_TOKENIZER_CLASSES[key]

    class FL_Tokenizer(base_cls, _FLTokenizerMixin):
        """HuggingFace tokenizer subclass with bdelf config and helpers."""

    FL_Tokenizer.__name__ = "FL_Tokenizer"
    FL_Tokenizer.__qualname__ = "FL_Tokenizer"
    _FL_TOKENIZER_CLASSES[key] = FL_Tokenizer
    return FL_Tokenizer


class FL_Tokenizer:
    """Factory for HuggingFace tokenizer subclasses.

    The returned instance inherits from the HuggingFace tokenizer class that
    corresponds to ``config.base_tokenizer`` (e.g. ``GPT2TokenizerFast``).
    Obtain instances via :func:`get_tokenizer` or ``FL_Tokenizer(config)``.
    """

    def __new__(cls, config: FL_TokenizerConfig) -> "FL_Tokenizer":
        if config.name == "prototype":
            raise ValueError("Prototype tokenizer cannot be instantiated.")
        return _build_fl_tokenizer(config)


def _build_fl_tokenizer(config: FL_TokenizerConfig) -> "FL_Tokenizer":
    """Build an ``FL_Tokenizer`` instance from config."""
    import hf_config  # noqa: F401
    from transformers import AutoConfig, AutoTokenizer

    cache_dir = config.base_tokenizer_cache_path
    probe = AutoTokenizer.from_pretrained(
        config.base_tokenizer,
        cache_dir=cache_dir,
    )
    fl_cls = _get_fl_tokenizer_class(type(probe))

    tokenizer = fl_cls.from_pretrained(
        config.base_tokenizer,
        cache_dir=cache_dir,
    )
    tokenizer.config = config

    if config.special_tokens:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": config.special_tokens}
        )
        if num_added:
            print(f"Added {num_added} special token(s)")

    model_config = AutoConfig.from_pretrained(config.base_tokenizer)
    tokenizer._embed_size = _infer_embed_size(model_config)
    return tokenizer
