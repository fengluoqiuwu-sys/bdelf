"""BELF 块条件流语言模型配置。"""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig

from models.latent.encdec.readout import ensure_sigma_tag, parse_kl_entropy
from models.tokens import FL_TokenLayout

_KEY_ALIASES = (
    ("p_mean", "denoiser_p_mean"),
    ("p_std", "denoiser_p_std"),
    ("t_eps", "t_clean_eps"),
    ("sc_p_mean", "self_cond_cfg_p_mean"),
    ("sc_p_std", "self_cond_cfg_p_std"),
    ("w_sc_min", "self_cond_cfg_min"),
    ("w_sc_max", "self_cond_cfg_max"),
    ("ctx_drop_prob", "ctx_p_drop"),
    ("denoiser_noise_scale", "noise_sigma"),
)

# 旧键：加载旧 config.json 时忽略，不进指纹语义。
_LEGACY_IGNORED = frozenset({"exit", "lambda_ce", "ce_detach_g", "time_step"})


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
            "sc_cfg",
            "latent_tune",
            "latent_dim",
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
        mlp_ratio: float = 4.0,
        latent_model: str = "latent_vae",
        tag: str = "100m-b32-d1",
        sc_cfg: bool = True,
        latent_tune: str = "mid",
        latent_dim: int = 32,
        attn_backend: str = "sdpa",
        block_size: int = 16,
        proj_bias: bool = True,
        proj_norm: str = "rmsnorm",
        whiten: bool = True,
        whiten_on: str = "mu",
        qk_norm: bool = True,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        t_freq_dim: int = 256,
        lm_head_bias: bool = False,
        noise_sigma: float = 1.0,
        latent_thaw_tokens: int | float = 5e9,
        denoiser_p_mean: float = -1.5,
        denoiser_p_std: float = 0.8,
        t_clean_eps: float = 0.05,
        vel_eps: float = 1e-3,
        cond_mode: str = "clean",
        clean_block_prob: float = 0.05,
        lambda_mse: float = 1.0,
        lambda_s1: float = 1.0,
        lambda_vae: float = 1.0,
        lambda_ref: float = 1.0,
        kl_entropy: bool = False,
        ctx_source: str = "mu",
        x0_source: str = "z",
        self_left_prob: float = 0.25,
        self_left_thaw_tokens: int | float = 5e9,
        self_cond_cfg_p_mean: float = -1.5,
        self_cond_cfg_p_std: float = 0.8,
        self_cond_cfg_min: float = 0.5,
        self_cond_cfg_max: float = 5.0,
        sc_guided_prob: float = 0.5,
        ctx_p_drop: float = 0.1,
        sampling: Dict[str, Any] | None = None,
        train_t_schedule: str = "independent",
        **kwargs: Any,
    ) -> None:
        if "gen_mode" in kwargs:
            raise ValueError("belf 不设 gen_mode；生成锁死 block_generate")
        if "n_layer_dec" in kwargs or "n_exit_layer" in kwargs:
            raise ValueError("belf 出口锁死 VAE-dec，无层数键")
        if "proj_type" in kwargs or "bottleneck_dim" in kwargs:
            raise ValueError("belf 不设 proj_type/bottleneck_dim；流维是 latent_dim，G 隐层是 n_embd")
        for old, new in _KEY_ALIASES:
            if old not in kwargs:
                continue
            mapped = kwargs.pop(old)
            if new == "denoiser_p_mean":
                denoiser_p_mean = mapped
            elif new == "denoiser_p_std":
                denoiser_p_std = mapped
            elif new == "t_clean_eps":
                t_clean_eps = mapped
            elif new == "self_cond_cfg_p_mean":
                self_cond_cfg_p_mean = mapped
            elif new == "self_cond_cfg_p_std":
                self_cond_cfg_p_std = mapped
            elif new == "self_cond_cfg_min":
                self_cond_cfg_min = mapped
            elif new == "self_cond_cfg_max":
                self_cond_cfg_max = mapped
            elif new == "ctx_p_drop":
                ctx_p_drop = mapped
            elif new == "noise_sigma":
                noise_sigma = mapped
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
        self.max_seq_len = max_seq_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.mlp_ratio = float(mlp_ratio)
        self.latent_model = latent_model
        self.kl_entropy = parse_kl_entropy(kl_entropy)
        self.tag = ensure_sigma_tag(str(tag), self.kl_entropy)
        self.sc_cfg = bool(sc_cfg)
        self.latent_tune = latent_tune
        self.latent_dim = int(latent_dim)
        self.attn_backend = attn_backend
        self.block_size = int(block_size)
        self.proj_bias = bool(proj_bias)
        self.proj_norm = str(proj_norm).strip().lower()
        self.whiten = bool(whiten)
        self.whiten_on = str(whiten_on).strip().lower()
        self.qk_norm = bool(qk_norm)
        self.rope_dim = rope_dim
        self.rope_theta = float(rope_theta)
        self.t_freq_dim = int(t_freq_dim)
        self.lm_head_bias = bool(lm_head_bias)
        self.noise_sigma = float(noise_sigma)
        self.latent_thaw_tokens = latent_thaw_tokens
        self.denoiser_p_mean = float(denoiser_p_mean)
        self.denoiser_p_std = float(denoiser_p_std)
        self.t_clean_eps = float(t_clean_eps)
        self.vel_eps = float(vel_eps)
        self.cond_mode = str(cond_mode).strip().lower()
        if self.cond_mode != "clean":
            raise ValueError(f"belf cond_mode 锁死 clean，收到 {cond_mode!r}")
        self.clean_block_prob = float(clean_block_prob)
        sched = str(train_t_schedule).strip().lower()
        # 旧 ckpt 的 block（离散 hop）视为 independent。
        if sched in ("block", "independent"):
            sched = "independent"
        if sched != "independent":
            raise ValueError(
                f"belf train_t_schedule 锁死 independent，收到 {train_t_schedule!r}"
            )
        self.train_t_schedule = sched
        self.lambda_mse = float(lambda_mse)
        self.lambda_s1 = float(lambda_s1)
        self.lambda_vae = float(lambda_vae)
        self.lambda_ref = float(lambda_ref)
        ctx = str(ctx_source).strip().lower()
        if ctx != "mu":
            raise ValueError(f"belf ctx_source 锁死 mu，不允许 {ctx_source!r}")
        self.ctx_source = "mu"
        self.x0_source = str(x0_source).strip().lower()
        p_left = float(self_left_prob)
        if not (0.0 <= p_left <= 1.0):
            raise ValueError(f"self_left_prob 须在 [0,1]，收到 {self_left_prob}")
        self.self_left_prob = p_left
        self.self_left_thaw_tokens = int(self_left_thaw_tokens)
        self.self_cond_cfg_p_mean = float(self_cond_cfg_p_mean)
        self.self_cond_cfg_p_std = float(self_cond_cfg_p_std)
        self.self_cond_cfg_min = float(self_cond_cfg_min)
        self.self_cond_cfg_max = float(self_cond_cfg_max)
        self.sc_guided_prob = float(sc_guided_prob)
        self.ctx_p_drop = float(ctx_p_drop)
        self.sampling = sampling or {
            "sampling_method": "sde",
            "sde_gamma": 1.5,
            "w_sc": 2.0,
            "w_ctx": 1.0,
            "temperature": 0.0,
            "commit_x0hat": False,
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
