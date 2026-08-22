"""TrACE（Tracked ACE）配置：ELF 骨架 + 训练期抗吸引子正则。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_TrACEConfig(PretrainedConfig):
    """TrACE：ELF 双分支 + 抗吸引子正则 L_attr；长训慢跟踪 d。"""

    model_type = "fl_trace"
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
            "attr_weight",
            "attr_warmup_steps",
            "attr_d_every",
            "attr_d_ema_beta",
            "attr_estimate_n",
            "attr_estimate_bs",
            "attr_estimate_seed",
            "attr_min_rep_gap",
            "attr_freeze_d",
            "attr_d_init",
        }
    )

    def __init__(
        self,
        name: str = "trace",
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
        # 0 = pre-SC-CFG ELF (old checkpoints / YAML without these fields).
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
        attr_weight: float = 7.0,
        attr_warmup_steps: int = 1000,
        attr_d_every: int = 1000,
        attr_d_ema_beta: float = 0.9,
        attr_estimate_n: int = 128,
        attr_estimate_bs: int = 8,
        attr_estimate_seed: int = 0,
        attr_min_rep_gap: float = 0.01,
        attr_freeze_d: bool = False,
        attr_d_init: str | None = None,
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
        self.attr_weight = float(attr_weight)
        self.attr_warmup_steps = int(attr_warmup_steps)
        self.attr_d_every = int(attr_d_every)
        self.attr_d_ema_beta = float(attr_d_ema_beta)
        self.attr_estimate_n = int(attr_estimate_n)
        self.attr_estimate_bs = int(attr_estimate_bs)
        self.attr_estimate_seed = int(attr_estimate_seed)
        self.attr_min_rep_gap = float(attr_min_rep_gap)
        self.attr_freeze_d = bool(attr_freeze_d)
        if attr_d_init is None or str(attr_d_init).strip() in ("", "null", "None"):
            self.attr_d_init = None
        else:
            self.attr_d_init = str(attr_d_init).strip()
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
            "attr_weight": self.attr_weight,
            "attr_warmup_steps": self.attr_warmup_steps,
            "attr_d_every": self.attr_d_every,
            "attr_d_ema_beta": self.attr_d_ema_beta,
            "attr_estimate_n": self.attr_estimate_n,
            "attr_estimate_bs": self.attr_estimate_bs,
            "attr_estimate_seed": self.attr_estimate_seed,
            "attr_min_rep_gap": self.attr_min_rep_gap,
            "attr_freeze_d": self.attr_freeze_d,
            "attr_d_init": self.attr_d_init,
        }


CONFIG_CLS = FL_TrACEConfig


@dataclass
class TrACESamplingConfig:
    """TrACE 推理采样（默认 ACE=off；支持 SC-CFG）。"""

    sampling_method: str = "sde"
    num_sampling_steps: int = 32
    sde_gamma: float = 1.5
    time_schedule: str | None = None
    temperature: float = 1.0
    top_k: int | None = None
    self_cond_cfg_scale: float = 3.0  # training-time SC-CFG scale (paper default)
    ace: bool | float | str = False  # ACE λ / 路径；false 关闭
    ace_direction: str | None = None  # 显式方向 .pt；None 则 artifacts/ace/{hash}/{step}/

    @classmethod
    def from_dict(cls, cfg: dict) -> "TrACESamplingConfig":
        raw = cfg.get("sampling", cfg)
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})
