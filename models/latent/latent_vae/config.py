"""latent_vae 配置（T5-small 维数 + Cola 公式）。"""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_LatentVAEConfig(PretrainedConfig):
    model_type = "fl_latent_vae"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "n_layer_enc",
            "n_layer_dec",
            "n_head",
            "n_embd",
            "d_kv",
            "d_ff",
            "latent_dim",
            "dropout",
            "beta_kl",
            "lambda_mask",
            "mask_ratio",
            "attn_backend",
        }
    )

    def __init__(
        self,
        name: str = "latent_vae",
        tokenizer: str = "gpt2",
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        max_seq_len: int = 1024,
        n_layer_enc: int = 6,
        n_layer_dec: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        latent_dim: int = 64,
        dropout: float = 0.0,
        beta_kl: float = 0.1,
        lambda_mask: float = 1.0,
        mask_ratio: float = 0.15,
        use_flash: bool = True,
        attn_backend: str = "sdpa",
        block_size: int = 1,
        sampling: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.pop("bidirectional", False):
            raise ValueError(
                "latent_vae 不支持 encoder 双向注意力；请勿在 YAML 或 --set 中设置 "
                "bidirectional=true（仅 latent_t5 可选双向）"
            )
        super().__init__(**kwargs)
        self.name = name
        self.tokenizer = tokenizer
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
        self.lambda_mask = lambda_mask
        self.mask_ratio = mask_ratio
        self.use_flash = use_flash
        self.attn_backend = attn_backend
        self.block_size = block_size
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
            "n_layer_enc": self.n_layer_enc,
            "n_layer_dec": self.n_layer_dec,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "d_kv": self.d_kv,
            "d_ff": self.d_ff,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "beta_kl": self.beta_kl,
            "lambda_mask": self.lambda_mask,
            "mask_ratio": self.mask_ratio,
            "use_flash": self.use_flash,
            "attn_backend": self.attn_backend,
            "block_size": self.block_size,
        }


CONFIG_CLS = FL_LatentVAEConfig
