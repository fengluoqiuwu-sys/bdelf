"""AR1.5: AR2-like anchors without long-range s KV / t window.

Differences vs AR2:
  - Historical anchors are never attended (train & infer). Only the current
    block's s is visible to that block's t (and sibling anchors).
  - No t_window: every s/t query sees all previous clean t tokens.
  - Inference discards s KV after each block finishes; cache_t grows unbounded.
"""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_AR15Config(PretrainedConfig):
    """Configuration for the AR1.5 semi-autoregressive anchor model."""

    model_type = "fl_ar1_5"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "block_size",
            "num_anchors",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "attn_backend",
            "mask_ratio_min",
            "num_noise_copies",
            "attn_type_bias",
            "fix_bos",
        }
    )

    def __init__(
        self,
        name: str = "ar1_5",
        tokenizer: str = "gpt2",
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        max_seq_len: int = 8192,
        block_size: int = 16,
        num_anchors: int = 1,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 672,
        dropout: float = 0.1,
        attn_backend: str = "flex",
        mask_ratio_min: float = 0.05,
        num_noise_copies: int = 2,
        attn_type_bias: bool = True,
        fix_bos: bool = True,
        sampling: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Drop AR2-only keys if a shared yaml fragment is reused.
        kwargs.pop("t_window", None)
        super().__init__(**kwargs)
        self.name = name
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.attn_backend = attn_backend
        self.mask_ratio_min = mask_ratio_min
        self.num_noise_copies = num_noise_copies
        self.attn_type_bias = attn_type_bias
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
            "block_size": self.block_size,
            "num_anchors": self.num_anchors,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "attn_backend": self.attn_backend,
            "mask_ratio_min": self.mask_ratio_min,
            "num_noise_copies": self.num_noise_copies,
            "attn_type_bias": self.attn_type_bias,
            "fix_bos": self.fix_bos,
        }


CONFIG_CLS = FL_AR15Config
