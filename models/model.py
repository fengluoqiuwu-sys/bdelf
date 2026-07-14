"""Model factory, YAML config loading, and PreTrainedModel wrapper base."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import hf_config  # noqa: F401
from transformers import PretrainedConfig, PreTrainedModel

from models.tokens import apply_token_layout_to_config, token_layout_from_cfg


def ensure_token_layout(config: PretrainedConfig) -> None:
    """Fill token IDs on config when only ``tokenizer`` was set (e.g. yaml load)."""
    if getattr(config, "vocab_size", 0) == 0 and getattr(config, "tokenizer", None):
        apply_token_layout_to_config(
            config, token_layout_from_cfg({"tokenizer": config.tokenizer}),
        )


def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Temperature + optional top-k multinomial sampling over the last dim.

    Args:
        logits: ``(..., V)`` unnormalized scores.
    Returns:
        ``(...)`` long token ids.
    """
    logits = logits / max(float(temperature), 1e-8)
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        values, _ = torch.topk(logits, k)
        cutoff = values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(*probs.shape[:-1])


def merge_sampling_cfg(
    config: PretrainedConfig,
    sampling_cfg: Dict[str, Any] | None,
    *,
    use_fast_infer_override: bool | None = None,
) -> Dict[str, Any]:
    """Merge explicit ``sampling_cfg`` over ``config.sampling`` from yaml."""
    base = getattr(config, "sampling", None) or {}
    if not isinstance(base, dict):
        base = {}
    merged = dict(base)
    if sampling_cfg:
        merged.update(sampling_cfg)
    if use_fast_infer_override is not None:
        merged["use_fast_infer"] = use_fast_infer_override
    return merged

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "models"

TConfig = TypeVar("TConfig", bound=PretrainedConfig)


def list_models() -> List[str]:
    """Return model family names discovered from ``config/models/*/``."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(
        path.name
        for path in CONFIG_DIR.iterdir()
        if path.is_dir() and path.name != "prototype"
    )


def list_model_configs(model: str) -> List[str]:
    """Return config names for a model family (yaml stems under ``config/models/<model>/``)."""
    model_dir = CONFIG_DIR / model
    if not model_dir.is_dir():
        return []
    return sorted(
        path.stem
        for path in model_dir.glob("*.yaml")
        if path.stem != "prototype"
    )


def resolve_model_config_path(model: str, config_arg: str) -> Path:
    """Resolve a config name or explicit yaml path."""
    as_path = Path(config_arg)
    if as_path.suffix in (".yaml", ".yml") and as_path.is_file():
        return as_path
    path = CONFIG_DIR / model / f"{config_arg}.yaml"
    if not path.is_file():
        available = ", ".join(list_model_configs(model)) or "<none>"
        raise FileNotFoundError(
            f"Config {path} does not exist. Available for '{model}': {available}"
        )
    return path


def load_model_yaml(
    path: str | os.PathLike,
    *,
    required: frozenset[str],
) -> Dict[str, Any]:
    """Load a model YAML mapping with strict required-key validation."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")

    missing = required - set(raw)
    if missing:
        raise ValueError(f"{path}: missing required fields {sorted(missing)}")

    return raw


def split_model_cfg(raw: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    """Split YAML dict into backbone kwargs and optional nested ``sampling`` block."""
    data = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    sampling = data.pop("sampling", None)
    if sampling is not None and not isinstance(sampling, dict):
        raise ValueError("sampling must be a mapping when present")
    return data, sampling


def config_from_yaml(config_cls: Type[TConfig], path: str | os.PathLike) -> TConfig:
    """Load a ``PretrainedConfig`` subclass from yaml."""
    required = getattr(config_cls, "_YAML_REQUIRED", frozenset())
    raw = load_model_yaml(path, required=required)
    data, sampling = split_model_cfg(raw)
    config = config_cls(**data)
    if sampling is not None:
        config.sampling = sampling
    return config


class FL_PreTrainedModel(PreTrainedModel):
    """Wraps a plain ``nn.Module`` backbone; training forward delegates to it."""

    backbone: nn.Module
    main_input_name = "input_ids"
    base_model_prefix = "backbone"

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        idx: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens = input_ids if input_ids is not None else idx
        if tokens is None:
            raise ValueError("forward requires input_ids or idx")
        label_tensor = labels if labels is not None else targets
        return self.backbone(tokens, label_tensor, **kwargs)

    @property
    def full_sequence_training(self) -> bool:
        return getattr(self.backbone, "full_sequence_training", False)

    @property
    def dual_branch_logging(self) -> bool:
        return getattr(self.backbone, "dual_branch_logging", False)

    @property
    def last_loss_branch(self) -> str:
        return getattr(self.backbone, "last_loss_branch", "")

    def generate(
        self,
        *args: Any,
        sampling_cfg: Dict[str, Any] | None = None,
        for_eval: bool = False,
        **kwargs: Any,
    ) -> Any:
        infer_override = None
        if for_eval:
            infer_override = False
        return self.backbone.generate(
            *args,
            sampling_cfg=merge_sampling_cfg(
                self.config,
                sampling_cfg,
                use_fast_infer_override=infer_override,
            ),
            **kwargs,
        )

    def count_parameters(self) -> int:
        """Return total parameter count (tokenizer is not part of the model)."""
        return sum(p.numel() for p in self.parameters())

    def print_parameter_count(self) -> None:
        """Print total parameter count (tokenizer is not part of the model)."""
        n = self.count_parameters()
        if n >= 1_000_000_000:
            scale, unit = 1e9, "B"
        elif n >= 1_000_000:
            scale, unit = 1e6, "M"
        elif n >= 1_000:
            scale, unit = 1e3, "K"
        else:
            scale, unit = 1.0, ""
        human = f"{n / scale:.2f}{unit}" if unit else str(n)
        print(f"[model] Total parameters: {n:,} ({human})")


def build_model(model_name: str, model_cfg: dict) -> FL_PreTrainedModel:
    """Build a model by family name and config dict."""
    try:
        module = importlib.import_module(f"models.{model_name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Model package not found: models/{model_name}/") from exc

    if not hasattr(module, "build_model"):
        raise ValueError(f"models/{model_name}/ is missing build_model(cfg)")
    return module.build_model(model_cfg)


def get_model(model: str, config_name: str) -> FL_PreTrainedModel:
    """Load ``config/models/<model>/<config_name>.yaml`` and instantiate the model."""
    path = resolve_model_config_path(model, config_name)
    try:
        module = importlib.import_module(f"models.{model}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Model package not found: models/{model}/") from exc

    if not hasattr(module, "CONFIG_CLS"):
        raise ValueError(f"models/{model}/ is missing CONFIG_CLS")
    config = config_from_yaml(module.CONFIG_CLS, path)
    ensure_token_layout(config)
    return module.build_model_from_config(config)
