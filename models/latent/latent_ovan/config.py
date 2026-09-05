"""latent_ovan 配置（csh BlockVAE 结构 + 本机 gpt2 词表）。"""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_LatentOvanConfig(PretrainedConfig):
    model_type = "fl_latent_ovan"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "n_layer_enc",
            "n_layer_dec",
            "n_head",
            "n_embd",
            "d_kv",
            "mlp_mult",
            "latent_dim",
            "dropout",
            "beta_kl",
            "lambda_mask",
            "mask_ratio",
            "mask_rand_frac",
            "z_zero_ratio",
            "recon_t_mean",
            "recon_t_std",
            "noise_sigma",
            "qk_norm",
            "tie_wte_normalize",
            "sample_vocab_size",
            "use_flash",
            "attn_backend",
            "block_size",
        }
    )

    def __init__(
        self,
        name: str = "latent_ovan",
        tokenizer: str = "gpt2",
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
        mlp_mult: int = 4,
        latent_dim: int = 16,
        dropout: float = 0.1,
        beta_kl: float = 0.05,
        lambda_mask: float = 1.0,
        mask_ratio: float = 0.1,
        mask_rand_frac: float = 1.0,
        z_zero_ratio: float = 0.1,
        recon_t_mean: float = 0.5,
        recon_t_std: float = 1.0,
        noise_sigma: float = 1.0,
        sample_vocab_size: int = 50257,
        qk_norm: bool = True,
        tie_wte_normalize: bool = True,
        use_flash: bool = True,
        attn_backend: str = "sdpa",
        block_size: int = 16,
        sampling: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.pop("bidirectional", False):
            raise ValueError(
                "latent_ovan 不支持 encoder 全双向；块内双向由 block_size>1 提供"
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
        self.mlp_mult = mlp_mult
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.beta_kl = beta_kl
        self.lambda_mask = lambda_mask
        self.mask_ratio = mask_ratio
        self.mask_rand_frac = mask_rand_frac
        self.z_zero_ratio = z_zero_ratio
        self.recon_t_mean = recon_t_mean
        self.recon_t_std = recon_t_std
        self.noise_sigma = noise_sigma
        self.sample_vocab_size = sample_vocab_size
        self.qk_norm = qk_norm
        self.tie_wte_normalize = tie_wte_normalize
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
            "mlp_mult": self.mlp_mult,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "beta_kl": self.beta_kl,
            "lambda_mask": self.lambda_mask,
            "mask_ratio": self.mask_ratio,
            "mask_rand_frac": self.mask_rand_frac,
            "z_zero_ratio": self.z_zero_ratio,
            "recon_t_mean": self.recon_t_mean,
            "recon_t_std": self.recon_t_std,
            "noise_sigma": self.noise_sigma,
            "sample_vocab_size": self.sample_vocab_size,
            "qk_norm": self.qk_norm,
            "tie_wte_normalize": self.tie_wte_normalize,
            "use_flash": self.use_flash,
            "attn_backend": self.attn_backend,
            "block_size": self.block_size,
        }


CONFIG_CLS = FL_LatentOvanConfig
