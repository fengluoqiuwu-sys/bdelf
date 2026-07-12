"""Autoregressive GPT configuration."""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_ARConfig(PretrainedConfig):
    """Configuration for autoregressive GPT backbone."""

    model_type = "fl_ar"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "use_flash",
        }
    )

    def __init__(
        self,
        name: str = "ar",
        tokenizer: str = "gpt2",
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        max_seq_len: int = 1024,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 672,
        dropout: float = 0.1,
        use_flash: bool = True,
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
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.use_flash = use_flash

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
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "use_flash": self.use_flash,
        }


CONFIG_CLS = FL_ARConfig
