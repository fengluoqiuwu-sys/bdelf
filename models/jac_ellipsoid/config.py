"""JacEllipsoid 配置：ELF 双分支 + QDA 观测读出（不改 models/elf）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_JacEllipsoidConfig(PretrainedConfig):
    """ELF 复制体：decode / 采样走各向异性 QDA，中段 FM 仍欧氏 MSE。"""

    model_type = "fl_jac_ellipsoid"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "encoder_model_name",
            "text_encoder_dim",
            "bottleneck_dim",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "num_time_tokens",
            "num_model_mode_tokens",
            "self_cond_prob",
            "latent_mean",
            "latent_std",
            "denoiser_p_mean",
            "denoiser_p_std",
            "denoiser_noise_scale",
            "decoder_prob",
            "decoder_p_mean",
            "decoder_p_std",
            "decoder_noise_scale",
            "t_eps",
            "time_schedule",
            "qda_mode",
            "qda_top_m",
            "qda_rank",
            "qda_lambda_scale",
            "qda_prior",
            "qda_table_path",
            "qda_decode_only",
            "freeze_dit",
            "qda_diag",
        }
    )

    def __init__(
        self,
        name: str = "jac_ellipsoid",
        tokenizer: str = "t5-small",
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        max_seq_len: int = 1024,
        encoder_model_name: str = "t5-small",
        text_encoder_dim: int = 512,
        bottleneck_dim: int = 128,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 0,
        num_model_mode_tokens: int = 4,
        self_cond_prob: float = 0.5,
        self_cond_cfg_min: float = 0.5,
        self_cond_cfg_max: float = 5.0,
        latent_mean: float = 0.0,
        latent_std: float = 0.2,
        denoiser_p_mean: float = -1.5,
        denoiser_p_std: float = 0.8,
        denoiser_noise_scale: float = 2.0,
        decoder_prob: float = 0.2,
        decoder_p_mean: float = 0.8,
        decoder_p_std: float = 0.8,
        decoder_noise_scale: float = 5.0,
        t_eps: float = 0.05,
        time_schedule: str = "logit_normal",
        qda_mode: str = "jacobian",
        qda_top_m: int = 16,
        qda_rank: int = 16,
        qda_lambda_scale: float = 0.01,
        qda_prior: str = "uniform",
        qda_table_path: str = "",
        qda_decode_only: bool = False,
        freeze_dit: bool = False,
        qda_diag: bool = False,
        sampling: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if "block_size" in kwargs:
            max_seq_len = int(kwargs.pop("block_size"))
        super().__init__(**kwargs)
        self.name = name
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_seq_len = max_seq_len
        self.encoder_model_name = encoder_model_name
        self.text_encoder_dim = text_encoder_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.self_cond_prob = self_cond_prob
        self.self_cond_cfg_min = self_cond_cfg_min
        self.self_cond_cfg_max = self_cond_cfg_max
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.denoiser_p_mean = denoiser_p_mean
        self.denoiser_p_std = denoiser_p_std
        self.denoiser_noise_scale = denoiser_noise_scale
        self.decoder_prob = decoder_prob
        self.decoder_p_mean = decoder_p_mean
        self.decoder_p_std = decoder_p_std
        self.decoder_noise_scale = decoder_noise_scale
        self.t_eps = t_eps
        self.time_schedule = time_schedule
        self.qda_mode = str(qda_mode).lower()
        self.qda_top_m = int(qda_top_m)
        self.qda_rank = int(qda_rank)
        self.qda_lambda_scale = float(qda_lambda_scale)
        self.qda_prior = str(qda_prior).lower()
        self.qda_table_path = "" if qda_table_path is None else str(qda_table_path)
        self.qda_decode_only = bool(qda_decode_only)
        self.freeze_dit = bool(freeze_dit)
        self.qda_diag = bool(qda_diag)
        self.sampling = sampling or {}

    def token_layout(self) -> FL_TokenLayout:
        return FL_TokenLayout(
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            ignore_index=self.ignore_index,
        )

    def backbone_kwargs(self) -> Dict[str, Any]:
        return {
            "token_layout": self.token_layout(),
            "max_seq_len": self.max_seq_len,
            "encoder_model_name": self.encoder_model_name,
            "text_encoder_dim": self.text_encoder_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "mlp_ratio": self.mlp_ratio,
            "num_time_tokens": self.num_time_tokens,
            "num_self_cond_cfg_tokens": self.num_self_cond_cfg_tokens,
            "num_model_mode_tokens": self.num_model_mode_tokens,
            "self_cond_prob": self.self_cond_prob,
            "self_cond_cfg_min": self.self_cond_cfg_min,
            "self_cond_cfg_max": self.self_cond_cfg_max,
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "denoiser_p_mean": self.denoiser_p_mean,
            "denoiser_p_std": self.denoiser_p_std,
            "denoiser_noise_scale": self.denoiser_noise_scale,
            "decoder_prob": self.decoder_prob,
            "decoder_p_mean": self.decoder_p_mean,
            "decoder_p_std": self.decoder_p_std,
            "decoder_noise_scale": self.decoder_noise_scale,
            "t_eps": self.t_eps,
            "time_schedule": self.time_schedule,
            "qda_mode": self.qda_mode,
            "qda_top_m": self.qda_top_m,
            "qda_rank": self.qda_rank,
            "qda_lambda_scale": self.qda_lambda_scale,
            "qda_prior": self.qda_prior,
            "qda_table_path": self.qda_table_path,
            "qda_decode_only": self.qda_decode_only,
            "freeze_dit": self.freeze_dit,
            "qda_diag": self.qda_diag,
        }


CONFIG_CLS = FL_JacEllipsoidConfig


@dataclass
class JacEllipsoidSamplingConfig:
    """推理采样（末步 QDA 或 softmax 对照）。"""

    sampling_method: str = "sde"
    num_sampling_steps: int = 32
    sde_gamma: float = 1.5
    time_schedule: str | None = None
    temperature: float = 1.0
    top_k: int | None = None
    self_cond_cfg_scale: float = 3.0
    ace: bool | float | str = False
    ace_direction: str | None = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "JacEllipsoidSamplingConfig":
        raw = cfg.get("sampling", cfg)
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})
