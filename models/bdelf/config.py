"""Block Diffusion + Embedded Language Flow configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_BDELFConfig(PretrainedConfig):
    """Configuration for Block Diffusion + Embedded Language Flow (token emb latent)."""

    model_type = "fl_bdelf"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "text_encoder_dim",
            "bottleneck_dim",
            "diffusion_block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "attn_backend",
            "num_time_tokens",
            "self_cond_prob",
            "latent_mean",
            "latent_std",
            "denoiser_p_mean",
            "denoiser_p_std",
            "denoiser_noise_scale",
            "n_dec_layer",
            "n_dec_head",
            "n_dec_embd",
            "dec_dropout",
            "lambda_dec",
            "t_eps",
            "time_schedule",
            "fix_bos",
            "freeze_wte",
            "wte_init",
        }
    )

    def __init__(
        self,
        name: str = "bdelf",
        tokenizer: str = "t5-small",
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        max_seq_len: int = 1024,
        text_encoder_dim: int = 512,
        bottleneck_dim: int = 128,
        diffusion_block_size: int = 16,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        attn_backend: str = "flex",
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 0,
        self_cond_prob: float = 0.5,
        self_cond_cfg_min: float = 0.5,
        self_cond_cfg_max: float = 5.0,
        latent_mean: float = 0.0,
        latent_std: float = 0.2,
        denoiser_p_mean: float = -1.5,
        denoiser_p_std: float = 0.8,
        denoiser_noise_scale: float = 2.0,
        n_dec_layer: int = 4,
        n_dec_head: int = 6,
        n_dec_embd: int = 384,
        dec_dropout: float = 0.1,
        lambda_dec: float = 1.0,
        t_eps: float = 0.05,
        time_schedule: str = "logit_normal",
        fix_bos: bool = True,
        freeze_wte: bool = True,
        wte_init: str = "gpt2",
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
        self.text_encoder_dim = text_encoder_dim
        self.bottleneck_dim = bottleneck_dim
        self.diffusion_block_size = diffusion_block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.attn_backend = attn_backend
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.self_cond_prob = self_cond_prob
        self.self_cond_cfg_min = self_cond_cfg_min
        self.self_cond_cfg_max = self_cond_cfg_max
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.denoiser_p_mean = denoiser_p_mean
        self.denoiser_p_std = denoiser_p_std
        self.denoiser_noise_scale = denoiser_noise_scale
        self.n_dec_layer = n_dec_layer
        self.n_dec_head = n_dec_head
        self.n_dec_embd = n_dec_embd
        self.dec_dropout = dec_dropout
        self.lambda_dec = lambda_dec
        self.t_eps = t_eps
        self.time_schedule = time_schedule
        self.fix_bos = fix_bos
        self.freeze_wte = freeze_wte
        self.wte_init = wte_init
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
            "text_encoder_dim": self.text_encoder_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "diffusion_block_size": self.diffusion_block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "attn_backend": self.attn_backend,
            "num_time_tokens": self.num_time_tokens,
            "num_self_cond_cfg_tokens": self.num_self_cond_cfg_tokens,
            "self_cond_prob": self.self_cond_prob,
            "self_cond_cfg_min": self.self_cond_cfg_min,
            "self_cond_cfg_max": self.self_cond_cfg_max,
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "denoiser_p_mean": self.denoiser_p_mean,
            "denoiser_p_std": self.denoiser_p_std,
            "denoiser_noise_scale": self.denoiser_noise_scale,
            "n_dec_layer": self.n_dec_layer,
            "n_dec_head": self.n_dec_head,
            "n_dec_embd": self.n_dec_embd,
            "dec_dropout": self.dec_dropout,
            "lambda_dec": self.lambda_dec,
            "t_eps": self.t_eps,
            "time_schedule": self.time_schedule,
            "fix_bos": self.fix_bos,
            "freeze_wte": self.freeze_wte,
            "wte_init": self.wte_init,
        }


CONFIG_CLS = FL_BDELFConfig


@dataclass
class FlowSamplingConfig:
    """BDELF inference configuration (supports SC-CFG guidance)."""

    num_ode_steps: int = 32
    time_schedule: str | None = None
    use_fast_infer: bool = True
    temperature: float = 1.0
    top_k: int | None = None
    self_cond_cfg_scale: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict) -> FlowSamplingConfig:
        raw = cfg.get("sampling", cfg)
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})
