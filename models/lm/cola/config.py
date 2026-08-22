"""Cola DLM Stage-2 configuration (block-causal DiT prior + Text VAE)."""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_ColaConfig(PretrainedConfig):
    """Block-causal latent DiT prior conditioned on a loaded Text VAE."""

    model_type = "fl_cola"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "n_layer",
            "n_head",
            "n_embd",
            "latent_dim",
            "diffusion_block_size",
            "dropout",
            "attn_backend",
            "vae_model",
            "vae_size",
        }
    )

    def __init__(
        self,
        name: str = "cola",
        tokenizer: str = "gpt2",
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        max_seq_len: int = 1024,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        latent_dim: int = 16,
        diffusion_block_size: int = 16,
        dropout: float = 0.0,
        denoiser_p_mean: float = 0.0,
        denoiser_p_std: float = 1.0,
        time_schedule: str = "logit_normal",
        schedule_loc: float = 1.0,
        t_eps: float = 0.05,
        ode_T: float = 1000.0,
        rope_dim: int | None = None,
        qk_norm: bool = True,
        expand_ratio: int = 4,
        attn_backend: str = "flex",
        vae_model: str = "cola_vae",
        vae_size: str = "100m",
        vae_run: str | None = None,
        vae_lr_ratio: float = 1.0,
        lambda_vae: float = 1.0,
        lambda_fm: float = 1.0,
        lambda_ref: float = 0.1,
        beta_kl: float = 0.1,
        lambda_mask: float = 1.0,
        mask_ratio: float = 0.15,
        freeze_vae: bool = False,
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
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.latent_dim = latent_dim
        self.diffusion_block_size = diffusion_block_size
        self.dropout = dropout
        self.denoiser_p_mean = denoiser_p_mean
        self.denoiser_p_std = denoiser_p_std
        self.time_schedule = time_schedule
        self.schedule_loc = schedule_loc
        self.t_eps = t_eps
        self.ode_T = ode_T
        self.rope_dim = rope_dim
        self.qk_norm = qk_norm
        self.expand_ratio = expand_ratio
        self.attn_backend = attn_backend
        self.vae_model = vae_model
        self.vae_size = vae_size
        self.vae_run = vae_run
        self.vae_lr_ratio = vae_lr_ratio
        self.lambda_vae = lambda_vae
        self.lambda_fm = lambda_fm
        self.lambda_ref = lambda_ref
        self.beta_kl = beta_kl
        self.lambda_mask = lambda_mask
        self.mask_ratio = mask_ratio
        self.freeze_vae = freeze_vae
        self.sampling = sampling or {
            "num_ode_steps": 16,
            "cfg_scale": 7.0,
            "temperature": 1.0,
            "ode_T": 1000.0,
        }

    def token_layout(self) -> FL_TokenLayout:
        return FL_TokenLayout(
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            ignore_index=self.ignore_index,
        )


CONFIG_CLS = FL_ColaConfig
