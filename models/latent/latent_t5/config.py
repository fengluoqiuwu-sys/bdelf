"""latent_t5 配置（T5-small 维数；默认同模式 self-attn + span 辅助；可读出 none）。"""

from __future__ import annotations

from typing import Any, Dict, Literal

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout

ReadoutMode = Literal["e", "b", "none"]


class FL_LatentT5Config(PretrainedConfig):
    model_type = "fl_latent_t5"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "readout",
            "n_layer_enc",
            "n_layer_dec",
            "n_head",
            "n_embd",
            "d_kv",
            "d_ff",
            "latent_dim",
            "dropout",
            "beta_kl",
            "lambda_span",
            "span_mask_ratio",
            "num_sentinels",
            "bidirectional",
            "decoder_bidirectional",
        }
    )

    def __init__(
        self,
        name: str = "latent_t5",
        tokenizer: str = "gpt2",
        readout: ReadoutMode = "e",
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        max_seq_len: int = 4096,
        n_layer_enc: int = 6,
        n_layer_dec: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        latent_dim: int = 32,
        dropout: float = 0.0,
        beta_kl: float = 0.1,
        lambda_span: float = 1.0,
        span_mask_ratio: float = 0.15,
        span_mean_len: int = 3,
        num_sentinels: int = 100,
        bidirectional: bool = True,
        decoder_bidirectional: bool | None = None,
        use_flash: bool = True,
        sampling: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if readout not in ("e", "b", "none"):
            raise ValueError(f"readout must be 'e', 'b' or 'none', got {readout!r}")
        if decoder_bidirectional is not None and not isinstance(
            decoder_bidirectional, bool
        ):
            raise ValueError(
                f"decoder_bidirectional 须为 true/false/null，得到 {decoder_bidirectional!r}"
            )
        if readout == "none":
            if bidirectional is False or decoder_bidirectional is False:
                raise ValueError("readout=none（原 T5）只支持双向，禁止 unidirectional")
            bidirectional = True
            decoder_bidirectional = True
        self.name = name
        self.tokenizer = tokenizer
        self.readout = readout
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_seq_len = max_seq_len
        self.n_layer_enc = n_layer_enc
        self.n_layer_dec = n_layer_dec
        self.n_head = n_head
        self.n_embd = n_embd
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.beta_kl = beta_kl
        self.lambda_span = lambda_span
        self.span_mask_ratio = span_mask_ratio
        self.span_mean_len = span_mean_len
        self.num_sentinels = num_sentinels
        self.bidirectional = bidirectional
        self.decoder_bidirectional = decoder_bidirectional
        self.use_flash = use_flash
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
            "readout": self.readout,
            "n_layer_enc": self.n_layer_enc,
            "n_layer_dec": self.n_layer_dec,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "d_kv": self.d_kv,
            "d_ff": self.d_ff,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "beta_kl": self.beta_kl,
            "lambda_span": self.lambda_span,
            "span_mask_ratio": self.span_mask_ratio,
            "span_mean_len": self.span_mean_len,
            "num_sentinels": self.num_sentinels,
            "bidirectional": self.bidirectional,
            "decoder_bidirectional": self.decoder_bidirectional,
            "use_flash": self.use_flash,
        }


CONFIG_CLS = FL_LatentT5Config
