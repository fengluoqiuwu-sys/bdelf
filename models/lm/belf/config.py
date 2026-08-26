"""BELF 块条件流语言模型配置。"""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout


class FL_BelfConfig(PretrainedConfig):
    """块条件 rectified flow：2L 去噪块 + AdaLN-Zero G。"""

    model_type = "fl_belf"
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "tokenizer",
            "n_layer",
            "n_head",
            "n_embd",
            "max_seq_len",
            "dropout",
            "latent_model",
            "tag",
            "exit",
            "sc_cfg",
            "latent_tune",
            "time_step",
            "proj_type",
            "attn_backend",
            "block_size",
        }
    )

    def __init__(
        self,
        name: str = "belf",
        tokenizer: str = "gpt2",
        vocab_size: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        max_seq_len: int = 4096,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        latent_model: str = "latent_vae",
        tag: str = "100m-b32-d1",
        exit: str = "decoder",
        sc_cfg: bool = True,
        latent_tune: str = "mid",
        time_step: int = 16,
        proj_type: str = "linear",
        attn_backend: str = "sdpa",
        block_size: int = 16,
        n_layer_dec: int = 6,
        latent_thaw_tokens: int | float = 15e9,
        p_mean: float = -1.5,
        p_std: float = 0.8,
        t_eps: float = 0.05,
        vel_eps: float = 1e-5,
        ce_detach_g: bool = False,
        cond_mode: str = "clean",
        clean_block_prob: float = 0.05,
        lambda_mse: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_s1: float = 1.0,
        lambda_vae: float = 1.0,
        lambda_ref: float = 1.0,
        sc_p_mean: float = 0.0,
        sc_p_std: float = 1.0,
        w_sc_min: float = 0.5,
        w_sc_max: float = 5.0,
        sc_guided_prob: float = 0.5,
        ctx_drop_prob: float = 0.1,
        denoiser_noise_scale: float = 1.0,
        whiten: bool = True,
        rope_dim: int | None = None,
        qk_norm: bool = True,
        mlp_ratio: float = 4.0,
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
        self.dropout = dropout
        self.latent_model = latent_model
        self.tag = tag
        self.exit = exit
        self.sc_cfg = bool(sc_cfg)
        self.latent_tune = latent_tune
        self.time_step = int(time_step)
        self.proj_type = proj_type
        self.attn_backend = attn_backend
        self.block_size = int(block_size)
        self.n_layer_dec = int(n_layer_dec)
        self.latent_thaw_tokens = latent_thaw_tokens
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.t_eps = float(t_eps)
        self.vel_eps = float(vel_eps)
        self.ce_detach_g = bool(ce_detach_g)
        self.cond_mode = str(cond_mode)
        self.clean_block_prob = float(clean_block_prob)
        self.lambda_mse = float(lambda_mse)
        self.lambda_ce = float(lambda_ce)
        self.lambda_s1 = float(lambda_s1)
        self.lambda_vae = float(lambda_vae)
        self.lambda_ref = float(lambda_ref)
        self.sc_p_mean = float(sc_p_mean)
        self.sc_p_std = float(sc_p_std)
        self.w_sc_min = float(w_sc_min)
        self.w_sc_max = float(w_sc_max)
        self.sc_guided_prob = float(sc_guided_prob)
        self.ctx_drop_prob = float(ctx_drop_prob)
        self.denoiser_noise_scale = float(denoiser_noise_scale)
        self.whiten = bool(whiten)
        self.rope_dim = rope_dim
        self.qk_norm = bool(qk_norm)
        self.mlp_ratio = float(mlp_ratio)
        self.sampling = sampling or {
            "sampling_method": "sde",
            "sde_gamma": 1.5,
            "w_sc": 1.0,
            "w_ctx": 1.0,
            "temperature": 1.0,
            "commit_x0hat": True,
        }

    def token_layout(self) -> FL_TokenLayout:
        return FL_TokenLayout(
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            ignore_index=self.ignore_index,
        )


CONFIG_CLS = FL_BelfConfig
