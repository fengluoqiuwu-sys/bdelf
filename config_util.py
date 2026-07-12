"""Shared YAML config loading for FL_*Config dataclasses."""

from __future__ import annotations

import os
from dataclasses import fields
from typing import Any, Dict, FrozenSet, Type, TypeVar

import yaml

T = TypeVar("T")

# Dataclass field that holds YAML keys not mapped to config attributes.
EXTRA_FIELD = "extra"


def load_yaml_config(
    cls: Type[T],
    path: str | os.PathLike,
    *,
    required: FrozenSet[str],
) -> T:
    """Load a dataclass config from YAML with strict validation.

    - Keys that are not config attributes are stored in ``extra`` as key-value pairs.
    - Missing ``required`` keys raise ``ValueError``.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")

    attr_keys = {f.name for f in fields(cls) if f.name != EXTRA_FIELD}

    missing = required - set(raw)
    if missing:
        raise ValueError(f"{path}: missing required fields {sorted(missing)}")

    extra: Dict[str, Any] = {k: v for k, v in raw.items() if k not in attr_keys}
    kwargs = {k: raw[k] for k in attr_keys if k in raw}
    kwargs[EXTRA_FIELD] = extra
    return cls(**kwargs)
