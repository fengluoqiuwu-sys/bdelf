"""ResidW — ELF 薄变体：SC-CFG 老师只蒸馏位置残差。

骨干与推理复用 ``models.elf.model._ELFBackbone``。训练期 SC-CFG 把
``v_target = v_FM + (1-1/w) * HP(v_cond - v_uncond)``，HP 为长度均值高通
（有效位 ``loss_mask``）；``v_FM`` 不动。推理仍是带 ``w`` 的单次前向。

References:
  - ELF: https://arxiv.org/abs/2605.10938
  - Official PyTorch: https://github.com/lillian039/ELF/tree/pytorch_elf
"""

from __future__ import annotations

import torch

from models.elf.model import _ELFBackbone
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    split_model_cfg,
)
from models.residw.config import FL_ResidWConfig
from models.residw.t5_encoder import ensure_t5_encoder_cached
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg


class _ResidWBackbone(_ELFBackbone):
    """ResidW：仅覆盖训练期 SC-CFG 的 Δv（长度维高通）。"""

    def _highpass_seq(
        self, delta: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """对长度维去直流：``delta_i - mean_{j in mask} delta_j``。

        ``delta`` 为 (B, L, C)，``mask`` 为 (B, L)。pad / 无效位不进均值。
        """
        mask_f = mask.to(dtype=delta.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (delta * mask_f).sum(dim=1, keepdim=True) / denom
        return delta - mean

    def _sc_cfg_velocity_delta(
        self,
        v_cond: torch.Tensor,
        v_uncond: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        # ResidW：只蒸馏 HP(v_c - v_∅)，直流不进老师。
        return self._highpass_seq(v_cond - v_uncond, loss_mask)


class FL_ResidWModel(FL_PreTrainedModel):
    config_class = FL_ResidWConfig

    def __init__(self, config: FL_ResidWConfig) -> None:
        super().__init__(config)
        self.backbone = _ResidWBackbone(**config.backbone_kwargs())

    def count_parameters(self) -> int:
        """Trainable params only (exclude frozen T5 encoder)."""
        return self.backbone.trainable_parameter_count()


def build_model_from_config(config: FL_ResidWConfig) -> FL_ResidWModel:
    ensure_token_layout(config)
    # Populate cache before training starts (auto-download if missing).
    ensure_t5_encoder_cached(config.encoder_model_name)
    return FL_ResidWModel(config)


def build_model(cfg: dict) -> FL_ResidWModel:
    data, sampling = split_model_cfg(cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    config = FL_ResidWConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
