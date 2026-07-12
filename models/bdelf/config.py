"""Block Diffusion + Embedded Language Flow configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_BDELFConfig(PretrainedConfig):
    """Configuration for Block Diffusion + Embedded Language Flow."""

    model_type = "fl_bdelf"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "diffusion_block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "attn_backend",
            "denoiser_p_mean",
            "denoiser_p_std",
            "denoiser_noise_scale",
            "decoder_prob",
            "decoder_p_mean",
            "decoder_p_std",
            "decoder_noise_scale",
            "t_eps",
            "time_schedule",
            "fix_bos",
        }
    )

    def __init__(
        self,
        name: str = "bdelf",
        tokenizer: str = "gpt2",
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        max_seq_len: int = 4096,
        diffusion_block_size: int = 32,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 672,
        dropout: float = 0.1,
        attn_backend: str = "flex",
        denoiser_p_mean: float = -1.5,
        denoiser_p_std: float = 0.8,
        denoiser_noise_scale: float = 2.0,
        decoder_prob: float = 0.2,
        decoder_p_mean: float = 0.8,
        decoder_p_std: float = 0.8,
        decoder_noise_scale: float = 5.0,
        t_eps: float = 0.05,
        time_schedule: str = "logit_normal",
        fix_bos: bool = True,
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
        self.diffusion_block_size = diffusion_block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.attn_backend = attn_backend
        self.denoiser_p_mean = denoiser_p_mean
        self.denoiser_p_std = denoiser_p_std
        self.denoiser_noise_scale = denoiser_noise_scale
        self.decoder_prob = decoder_prob
        self.decoder_p_mean = decoder_p_mean
        self.decoder_p_std = decoder_p_std
        self.decoder_noise_scale = decoder_noise_scale
        self.t_eps = t_eps
        self.time_schedule = time_schedule
        self.fix_bos = fix_bos
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
            "diffusion_block_size": self.diffusion_block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "attn_backend": self.attn_backend,
            "denoiser_p_mean": self.denoiser_p_mean,
            "denoiser_p_std": self.denoiser_p_std,
            "denoiser_noise_scale": self.denoiser_noise_scale,
            "decoder_prob": self.decoder_prob,
            "decoder_p_mean": self.decoder_p_mean,
            "decoder_p_std": self.decoder_p_std,
            "decoder_noise_scale": self.decoder_noise_scale,
            "t_eps": self.t_eps,
            "time_schedule": self.time_schedule,
            "fix_bos": self.fix_bos,
        }


CONFIG_CLS = FL_BDELFConfig


@dataclass
class FlowSamplingConfig:
    """BDELF inference configuration."""

    num_ode_steps: int = 8
    time_schedule: str | None = None
    use_fast_infer: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> FlowSamplingConfig:
        raw = cfg.get("sampling", cfg)
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})
