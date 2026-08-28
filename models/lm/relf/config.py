"""RELF 配置：滚动窗局部时间场，无 ``block_size`` / ``gen_mode``。"""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig

from models.tokens import FL_TokenLayout

_LEGACY_IGNORED = frozenset({"exit", "lambda_ce", "ce_detach_g"})


class FL_RelfConfig(PretrainedConfig):
    """滚动窗 rectified flow LM；入口走 ``LatentBundle``。"""

    model_type = "fl_relf"
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
            "sc_cfg",
            "latent_tune",
            "time_step",
            "latent_dim",
            "attn_backend",
            "window_size",
            "step_size",
        }
    )

    def __init__(
        self,
        name: str = "relf",
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
        mlp_ratio: float = 4.0,
        latent_model: str = "latent_vae",
        tag: str = "100m-b32-d1",
        sc_cfg: bool = True,
        latent_tune: str = "mid",
        time_step: int = 16,
        latent_dim: int = 32,
        attn_backend: str = "sdpa",
        window_size: int = 32,
        step_size: int = 2,
        proj_bias: bool = True,
        proj_norm: str = "rmsnorm",
        whiten: bool = True,
        whiten_on: str = "mu",
        qk_norm: bool = True,
        rope_theta: float = 10000.0,
        t_freq_dim: int = 256,
        lm_head_bias: bool = False,
        noise_sigma: float = 1.0,
        latent_thaw_tokens: int = 15_000_000_000,
        lambda_vae: float = 1.0,
        lambda_ref: float = 1.0,
        denoiser_p_mean: float = -1.5,
        denoiser_p_std: float = 0.8,
        t_clean_eps: float = 0.05,
        vel_eps: float = 1e-3,
        lambda_mse: float = 1.0,
        lambda_s1: float = 1.0,
        ctx_source: str = "z",
        x0_source: str = "z",
        cond_mode: str = "clean",
        clean_block_prob: float = 0.05,
        train_t_schedule: str = "mixed",
        window_t: str = "ladder",
        self_left_prob: float = 0.25,
        self_cond_cfg_p_mean: float = -1.5,
        self_cond_cfg_p_std: float = 0.8,
        sc_guided_prob: float = 0.5,
        ctx_p_drop: float = 0.1,
        self_cond_cfg_min: float = 0.5,
        self_cond_cfg_max: float = 5.0,
        sampling: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if "block_size" in kwargs:
            raise ValueError("relf 无 block_size；请用 window_size / step_size")
        if "gen_mode" in kwargs:
            raise ValueError("relf 不设 gen_mode；生成锁死 rolling_generate")
        for banned in ("p_preroll", "p_freeroll", "p_postroll"):
            if banned in kwargs:
                raise ValueError(f"relf 不设 {banned}；截断由 BOS/EOS 给出")
        if "n_exit_layer" in kwargs or "n_layer_dec" in kwargs:
            raise ValueError("relf 出口锁死 VAE-dec，无层数键")
        if "proj_type" in kwargs or "bottleneck_dim" in kwargs:
            raise ValueError("relf 不设 proj_type/bottleneck_dim；流维是 latent_dim，G 隐层是 n_embd")
        for k in _LEGACY_IGNORED:
            kwargs.pop(k, None)
        super().__init__(**kwargs)
        self.name = name
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_seq_len = int(max_seq_len)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.n_embd = int(n_embd)
        self.dropout = float(dropout)
        self.mlp_ratio = float(mlp_ratio)
        self.latent_model = str(latent_model)
        self.tag = str(tag)
        self.sc_cfg = bool(sc_cfg)
        self.latent_tune = str(latent_tune).strip().lower()
        self.time_step = int(time_step)
        self.latent_dim = int(latent_dim)
        self.attn_backend = str(attn_backend).strip().lower()
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.proj_bias = bool(proj_bias)
        self.proj_norm = str(proj_norm).strip().lower()
        self.whiten = bool(whiten)
        self.whiten_on = str(whiten_on).strip().lower()
        self.qk_norm = bool(qk_norm)
        self.rope_theta = float(rope_theta)
        self.t_freq_dim = int(t_freq_dim)
        self.lm_head_bias = bool(lm_head_bias)
        self.noise_sigma = float(noise_sigma)
        self.latent_thaw_tokens = int(latent_thaw_tokens)
        self.lambda_vae = float(lambda_vae)
        self.lambda_ref = float(lambda_ref)
        self.denoiser_p_mean = float(denoiser_p_mean)
        self.denoiser_p_std = float(denoiser_p_std)
        self.t_clean_eps = float(t_clean_eps)
        self.vel_eps = float(vel_eps)
        self.lambda_mse = float(lambda_mse)
        self.lambda_s1 = float(lambda_s1)
        self.ctx_source = str(ctx_source).strip().lower()
        self.x0_source = str(x0_source).strip().lower()
        self.cond_mode = str(cond_mode).strip().lower()
        if self.cond_mode != "clean":
            raise ValueError(f"relf cond_mode 锁死 clean，收到 {cond_mode!r}")
        self.clean_block_prob = float(clean_block_prob)
        self.train_t_schedule = str(train_t_schedule).strip().lower()
        self.window_t = str(window_t).strip().lower()
        if self.train_t_schedule != "mixed":
            raise ValueError(
                f"relf train_t_schedule 锁死 mixed，收到 {train_t_schedule!r}"
            )
        if self.window_t != "ladder":
            raise ValueError(
                f"relf window_t 锁死 ladder，收到 {window_t!r}"
            )
        p_left = float(self_left_prob)
        if not (0.0 <= p_left <= 1.0):
            raise ValueError(f"self_left_prob 须在 [0,1]，收到 {self_left_prob}")
        self.self_left_prob = p_left
        self.self_cond_cfg_p_mean = float(self_cond_cfg_p_mean)
        self.self_cond_cfg_p_std = float(self_cond_cfg_p_std)
        self.sc_guided_prob = float(sc_guided_prob)
        self.ctx_p_drop = float(ctx_p_drop)
        self.self_cond_cfg_min = float(self_cond_cfg_min)
        self.self_cond_cfg_max = float(self_cond_cfg_max)
        self.sampling = sampling or {
            "sampling_method": "sde",
            "sde_gamma": 1.5,
            "w_sc": 3.0,
            "w_ctx": 1.0,
            "temperature": 0.0,
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


CONFIG_CLS = FL_RelfConfig
