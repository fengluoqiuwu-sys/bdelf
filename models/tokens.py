"""Model-side helpers for the shared ``FL_Tokenizer`` token layout."""

from __future__ import annotations

from typing import Any, Mapping

from tokenizer import FL_TokenLayout, get_token_layout

__all__ = [
    "FL_TokenLayout",
    "apply_token_layout_to_config",
    "get_token_layout",
    "token_layout_from_cfg",
]


def token_layout_from_cfg(cfg: Mapping[str, Any]) -> FL_TokenLayout:
    """Resolve layout from model yaml via ``tokenizer`` (same name as preprocess)."""
    tokenizer_name = cfg.get("tokenizer")
    if not tokenizer_name:
        raise ValueError("模型配置必须指定 tokenizer（与 preprocess 使用同一配置名）")
    return get_token_layout(str(tokenizer_name))


def apply_token_layout_to_config(config: Any, layout: FL_TokenLayout) -> None:
    """Write resolved token IDs onto a model ``PretrainedConfig``."""
    config.vocab_size = layout.vocab_size
    config.bos_token_id = layout.bos_token_id
    config.eos_token_id = layout.eos_token_id
    config.pad_token_id = layout.pad_token_id
    config.ignore_index = layout.ignore_index
