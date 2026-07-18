"""Frozen T5 encoder used as ELF's continuous embedding source (training only)."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn

import hf_config  # noqa: F401


_T5_DEFAULTS = {
    "t5-small": dict(
        vocab_size=32128, d_model=512, d_kv=64, d_ff=2048,
        num_layers=6, num_heads=8,
    ),
    "t5-base": dict(
        vocab_size=32128, d_model=768, d_kv=64, d_ff=3072,
        num_layers=12, num_heads=12,
    ),
    "t5-large": dict(
        vocab_size=32128, d_model=1024, d_kv=64, d_ff=4096,
        num_layers=24, num_heads=16,
    ),
}


class T5EncoderConfig:
    def __init__(self, model_name: str, dtype: Any = torch.float32) -> None:
        self.model_name = model_name
        self.dtype = dtype
        self.vocab_size = 0
        self.d_model = 0
        defaults = _T5_DEFAULTS.get(model_name, {})
        for key, value in defaults.items():
            setattr(self, key, value)

    @classmethod
    def from_pretrained(
        cls, model_name: str, dtype: Any = torch.float32,
    ) -> "T5EncoderConfig":
        return cls(model_name, dtype)


class T5Encoder(nn.Module):
    """Frozen ``T5EncoderModel`` wrapper."""

    def __init__(self, config: T5EncoderConfig, *, pretrained: bool = True) -> None:
        super().__init__()
        from transformers import T5Config, T5EncoderModel

        cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
        load_kwargs: dict[str, Any] = {}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        if pretrained:
            try:
                self.model = T5EncoderModel.from_pretrained(
                    config.model_name, local_files_only=True, **load_kwargs,
                )
            except (OSError, ValueError):
                self.model = T5EncoderModel.from_pretrained(
                    config.model_name, **load_kwargs,
                )
        else:
            hf_config_obj = T5Config.from_pretrained(config.model_name, **load_kwargs)
            self.model = T5EncoderModel(hf_config_obj)

        hf = self.model.config
        config.vocab_size = int(hf.vocab_size)
        config.d_model = int(hf.d_model)
        self.config = config
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


def load_t5_encoder(
    model_name: str = "t5-small",
    dtype: Any = torch.float32,
) -> tuple[T5EncoderConfig, T5Encoder]:
    config = T5EncoderConfig.from_pretrained(model_name, dtype=dtype)
    encoder = T5Encoder(config, pretrained=True)
    if dtype is not None:
        encoder = encoder.to(dtype)
    return config, encoder


def encode_text(
    input_ids: torch.Tensor,
    encoder: T5Encoder,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    latent_mean: float = 0.0,
    latent_std: float = 0.2,
) -> torch.Tensor:
    """Encode tokens and channel-normalize to the ELF latent scale."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    hidden = encoder(input_ids, attention_mask=attention_mask)
    return (hidden - latent_mean) / max(float(latent_std), 1e-8)
