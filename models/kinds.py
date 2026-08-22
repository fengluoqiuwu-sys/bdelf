"""Model kind registry: ``lm`` vs ``latent`` (not in config hash)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Literal

ModelKind = Literal["lm", "latent"]

_KINDS: tuple[ModelKind, ...] = ("lm", "latent")
_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config" / "models"
_TRAIN_MODEL_ROOT = Path(__file__).resolve().parents[1] / "config" / "train" / "model"
_GENERATE_ROOT = Path(__file__).resolve().parents[1] / "config" / "generate"


def _scan_families(root: Path) -> dict[str, ModelKind]:
    out: dict[str, ModelKind] = {}
    for kind in _KINDS:
        kind_dir = root / kind
        if not kind_dir.is_dir():
            continue
        for path in sorted(kind_dir.iterdir()):
            if path.is_dir() and path.name != "prototype":
                out[path.name] = kind
    return out


def _family_kind_table() -> dict[str, ModelKind]:
    table = _scan_families(_CONFIG_ROOT)
    if not table:
        raise RuntimeError(f"no model families under {_CONFIG_ROOT}")
    return table


def kind_of(model: str) -> ModelKind:
    """Return ``lm`` or ``latent`` for a model family slug."""
    table = _family_kind_table()
    if model not in table:
        raise KeyError(f"unknown model family {model!r}")
    return table[model]


def list_kinds() -> list[ModelKind]:
    return list(_KINDS)


def list_models(*, kind: ModelKind | None = None) -> list[str]:
    """List registered model family names, optionally filtered by kind."""
    table = _family_kind_table()
    if kind is None:
        return sorted(table)
    return sorted(name for name, k in table.items() if k == kind)


def family_module_name(model: str) -> str:
    """Import path ``models.<kind>.<family>``."""
    return f"models.{kind_of(model)}.{model}"


def import_family_module(model: str):
    """Import the package for ``model`` (validates ``KIND`` when set)."""
    mod_name = family_module_name(model)
    module = importlib.import_module(mod_name)
    expected = kind_of(model)
    declared = getattr(module, "KIND", None)
    if declared is not None and declared != expected:
        raise ValueError(
            f"{mod_name}: KIND={declared!r} disagrees with directory kind {expected!r}"
        )
    return module


def resolve_model_config_dir(model: str) -> Path:
    return _CONFIG_ROOT / kind_of(model) / model


def resolve_train_model_dir(model: str) -> Path:
    return _TRAIN_MODEL_ROOT / kind_of(model) / model


def resolve_generate_dir(model: str) -> Path:
    return _GENERATE_ROOT / kind_of(model) / model
