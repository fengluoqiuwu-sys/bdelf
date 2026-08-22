"""Posβ（posbeta）— ELF 薄变体：位置相关插值 β_t^(ℓ)。

骨干与推理复用 ``models.lm.elf.model._ELFBackbone``。在冻结 encoder 连续空间
实现 DFM §4.1 写出但未实验的 ``z[ℓ] = β_t^(ℓ) x[ℓ] + (1-β_t^(ℓ)) ε[ℓ]``，
``β_t^(ℓ) = t^{1+κ u(ℓ)}``，前缀更干净、后缀更噪。κ=0 时与 ELF 各向同性
路径逐公式一致。推理仍共享全局 Euler 时钟；``x̂→v`` 与 SDE 回噪声按位置用 β。

References:
  - ELF: https://arxiv.org/abs/2605.10938
  - Official PyTorch: https://github.com/lillian039/ELF/tree/pytorch_elf
"""

from __future__ import annotations

import torch

from models.lm.elf.model import _ELFBackbone
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    split_model_cfg,
)
from models.lm.posbeta.config import FL_PosBetaConfig
from models.lm.posbeta.t5_encoder import ensure_t5_encoder_cached
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg


class _PosBetaBackbone(_ELFBackbone):
    """Posβ：覆盖插值路径 / ``x→v`` / SDE ``z_back``；κ=0 退回 ELF。"""

    def __init__(self, *args: object, pos_beta_kappa: float = 1.0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.pos_beta_kappa = float(pos_beta_kappa)

    def _pos_beta(
        self, t: torch.Tensor, seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """位置相关干净系数 β 与 ∂_t β，形状 (B, L, 1)。

        ``β_t^(ℓ) = t^{1+κ u(ℓ)}``，``u=(ℓ-1)/(L-1)``。κ=0 时不要走这里
        （``_x_to_v`` / ``_denoise_path`` / ``_sde_z_back`` 已短路到 ELF）。
        """
        device, dtype = t.device, t.dtype
        if seq_len <= 1:
            u = torch.zeros(seq_len, device=device, dtype=dtype)
        else:
            u = torch.arange(seq_len, device=device, dtype=dtype) / float(seq_len - 1)
        exp = 1.0 + self.pos_beta_kappa * u
        t_col = t.reshape(-1, 1).clamp(0.0, 1.0)
        beta = t_col.pow(exp.unsqueeze(0))
        # β̇ = (1+κu) t^{κu}；t→0 用 t_eps 避免 0^{负}。
        t_dot = t_col.clamp(min=self.t_eps)
        beta_dot = exp.unsqueeze(0) * t_dot.pow(exp.unsqueeze(0) - 1.0)
        return beta.unsqueeze(-1), beta_dot.unsqueeze(-1)

    def _x_to_v(
        self, x_pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        if abs(self.pos_beta_kappa) < 1e-12:
            return super()._x_to_v(x_pred, z, t)
        beta, beta_dot = self._pos_beta(t, z.shape[1])
        return beta_dot / torch.clamp(1.0 - beta, min=self.t_eps) * (x_pred - z)

    def _denoise_path(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if abs(self.pos_beta_kappa) < 1e-12:
            return super()._denoise_path(x0, t, noise)
        beta, beta_dot = self._pos_beta(t, x0.shape[1])
        z = beta * x0 + (1.0 - beta) * noise
        v_target = beta_dot / torch.clamp(1.0 - beta, min=self.t_eps) * (x0 - z)
        return z, v_target

    def _sde_z_back(
        self,
        z: torch.Tensor,
        alpha: float,
        eps: torch.Tensor,
        t_back: float,
    ) -> torch.Tensor:
        if abs(self.pos_beta_kappa) < 1e-12:
            return super()._sde_z_back(z, alpha, eps, t_back)
        # 剩余噪声预算按 1-β 缩放，κ=0 时与全局 1-t 一致。
        t_back_b = torch.full(
            (z.shape[0],), t_back, dtype=z.dtype, device=z.device,
        )
        beta, _ = self._pos_beta(t_back_b, z.shape[1])
        remain_iso = torch.clamp(
            1.0 - t_back_b.reshape(-1, 1, 1), min=self.t_eps,
        )
        remain_pos = torch.clamp(1.0 - beta, min=self.t_eps)
        return alpha * z + (1.0 - alpha) * eps * (remain_pos / remain_iso)

    def describe_training(self) -> str:
        decoder_prob = float(self.decoder_prob)
        return (
            f"POSBETA: per-example denoise:decode ≈ "
            f"{max(0.0, 1.0 - decoder_prob):g}:{decoder_prob:g} "
            f"+ Posβ interpolant (κ={self.pos_beta_kappa}); "
            "κ=0 退回各向同性 ELF；metrics: mse / ce"
        )


class FL_PosBetaModel(FL_PreTrainedModel):
    config_class = FL_PosBetaConfig

    def __init__(self, config: FL_PosBetaConfig) -> None:
        super().__init__(config)
        self.backbone = _PosBetaBackbone(**config.backbone_kwargs())

    def count_parameters(self) -> int:
        """Trainable params only (exclude frozen T5 encoder)."""
        return self.backbone.trainable_parameter_count()


def build_model_from_config(config: FL_PosBetaConfig) -> FL_PosBetaModel:
    ensure_token_layout(config)
    ensure_t5_encoder_cached(config.encoder_model_name)
    return FL_PosBetaModel(config)


def build_model(cfg: dict) -> FL_PosBetaModel:
    data, sampling = split_model_cfg(cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    config = FL_PosBetaConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
