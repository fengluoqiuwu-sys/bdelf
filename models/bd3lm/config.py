"""Block Diffusion Language Model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_BD3LMConfig(PretrainedConfig):
    """Configuration for Block Diffusion Language Model."""

    model_type = "fl_bd3lm"
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
            "sampling_eps_min",
            "sampling_eps_max",
            "fix_bos",
        }
    )

    def __init__(
        self,
        name: str = "bd3lm",
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
        sampling_eps_min: float = 1e-3,
        sampling_eps_max: float = 1.0,
        fix_bos: bool = True,
        sampling: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if "ignore_bos" in kwargs:
            fix_bos = bool(kwargs.pop("ignore_bos"))
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
        self.sampling_eps_min = sampling_eps_min
        self.sampling_eps_max = sampling_eps_max
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
            "sampling_eps_min": self.sampling_eps_min,
            "sampling_eps_max": self.sampling_eps_max,
            "fix_bos": self.fix_bos,
        }


CONFIG_CLS = FL_BD3LMConfig


@dataclass
class SamplingConfig:
    """推理超参，默认值对齐 bd3lms 官方配置。"""

    num_steps: int = 5000
    sampler: Literal["semi_ar"] = "semi_ar"
    first_hitting: bool = True
    nucleus_p: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict) -> SamplingConfig:
        raw = cfg.get("sampling", cfg)
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})
