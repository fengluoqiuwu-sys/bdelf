"""Cola Text VAE configuration (Stage-1)."""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_ColaVAEConfig(PretrainedConfig):
    """Causal-block Text VAE: encoder/decoder + continuous latent sequence."""

    model_type = "fl_cola_vae"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "n_layer_enc",
            "n_layer_dec",
            "n_head",
            "n_embd",
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
        name: str = "cola_vae",
        tokenizer: str = "gpt2",
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        max_seq_len: int = 1024,
        n_layer_enc: int = 4,
        n_layer_dec: int = 4,
        n_head: int = 6,
        n_embd: int = 384,
        latent_dim: int = 16,
        dropout: float = 0.0,
        beta_kl: float = 0.1,
        lambda_mask: float = 1.0,
        mask_ratio: float = 0.15,
        use_flash: bool = True,
        attn_backend: str = "flex",
        block_size: int = 16,
        rope_theta: float = 500000.0,
        qk_norm: bool = True,
        post_norm: bool = True,
        patch_size: int = 1,
        ffn_mult: int = 4,
        scaling_factor: float = 1.0,
        shifting_factor: float = 0.0,
        sampling: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
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
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.beta_kl = beta_kl
        self.lambda_mask = lambda_mask
        self.mask_ratio = mask_ratio
        self.use_flash = use_flash
        self.attn_backend = attn_backend
        self.block_size = block_size
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.post_norm = post_norm
        self.patch_size = patch_size
        self.ffn_mult = ffn_mult
        self.scaling_factor = scaling_factor
        self.shifting_factor = shifting_factor
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
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "beta_kl": self.beta_kl,
            "lambda_mask": self.lambda_mask,
            "mask_ratio": self.mask_ratio,
            "use_flash": self.use_flash,
            "attn_backend": self.attn_backend,
            "block_size": self.block_size,
            "rope_theta": self.rope_theta,
            "qk_norm": self.qk_norm,
            "post_norm": self.post_norm,
            "patch_size": self.patch_size,
            "ffn_mult": self.ffn_mult,
            "scaling_factor": self.scaling_factor,
            "shifting_factor": self.shifting_factor,
        }


CONFIG_CLS = FL_ColaVAEConfig
