"""ODAR T5 encoder facade: shared core with ``[odar]`` log prefix."""

from __future__ import annotations

from typing import Any

import torch

from models.elf_core.t5_encoder import *  # noqa: F403
from models.elf_core import t5_encoder as _core

_LOG_TAG = "odar"


def ensure_t5_encoder_cached(
    model_name: str = "t5-small",
    *,
    log_tag: str = _LOG_TAG,
) -> str:
    return _core.ensure_t5_encoder_cached(model_name, log_tag=log_tag)


class T5Encoder(_core.T5Encoder):
    def __init__(
        self,
        config: _core.T5EncoderConfig,
        *,
        pretrained: bool = True,
        log_tag: str = _LOG_TAG,
    ) -> None:
        super().__init__(config, pretrained=pretrained, log_tag=log_tag)


def load_t5_encoder(
    model_name: str = "t5-small",
    dtype: Any = torch.float32,
    *,
    log_tag: str = _LOG_TAG,
) -> tuple[_core.T5EncoderConfig, T5Encoder]:
    config = _core.T5EncoderConfig.from_pretrained(model_name, dtype=dtype)
    encoder = T5Encoder(config, pretrained=True, log_tag=log_tag)
    if dtype is not None:
        encoder = encoder.to(dtype)
    return config, encoder
