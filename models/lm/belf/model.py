"""BELF：块条件 rectified flow + 2L AdaLN-Zero G。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lm.belf.config import FL_BelfConfig
from models.lm.belf.generate import (
    MODE_DECODE,
    MODE_DENOISE,
    MODE_NONE,
    block_generate as run_block_generate,
)
from models.lm.belf_relf_core import (
    AdaLNZeroStack,
    ExitMap,
    LatentBundle,
    as_sdpa_mask,
    blend_v_tgt,
    check_time_step,
    hide_left_keys,
    hide_right_pad_from_unknown,
    interpolate,
    keep_params_in_graph,
    ladder_levels,
    maybe_drop_left,
    pack_2l,
    pack_2l_mask,
    pack_2l_parallel_blocks_mask,
    sample_w_sc,
    v_star,
    validate_joint_tune,
    validate_loaded_block,
    x_to_v,
)
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    split_model_cfg,
)
from models.rope import pair_positions
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg


def _whiten_vec(mod: nn.Module, names: tuple[str, ...], dim: int) -> torch.Tensor | None:
    """从入口模块取长度为 ``dim`` 的仿射向量；没有则 ``None``。"""
    for name in names:
        val = getattr(mod, name, None)
        if not isinstance(val, torch.Tensor):
            continue
        flat = val.detach().reshape(-1).float()
        if flat.numel() == dim:
            return flat.contiguous()
        if flat.numel() == 1:
            return flat.expand(dim).contiguous().clone()
    return None


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(dtype=x.dtype)
        return self.weight.to(dtype=x.dtype) * x


class _BelfBackbone(nn.Module):
    """块条件流：训练 2L 并行，推理 ``block_generate``。"""

    full_sequence_training = True
    supports_prefix = True
    # 一次 forward 按槽拆 MSE/CE；不要 ELF decoder_prob 抽支。
    dual_branch_logging = False
    mixed_branch_training = False
    eval_ppl_from_ce = True

    def __init__(self, config: FL_BelfConfig, bundle: LatentBundle) -> None:
        super().__init__()
        self.config = config
        self.token_layout = config.token_layout()
        self.max_seq_len = int(config.max_seq_len)
        self.n_embd = int(config.n_embd)
        self.n_head = int(config.n_head)
        self.block_size = int(config.block_size)
        if self.block_size < 1:
            raise ValueError(f"block_size W 须为正整数，收到 {config.block_size}")
        self.time_step = int(config.time_step)
        check_time_step(self.time_step)
        validate_loaded_block(
            family="belf",
            loaded_block=bundle.block_size,
            W=self.block_size,
        )
        validate_joint_tune(loaded_block=bundle.block_size, tune=bundle.tune)
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd={self.n_embd} 须能被 n_head={self.n_head} 整除"
            )

        self.bundle = bundle
        self.latent_dim = int(bundle.latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("LatentBundle.latent_dim 无效")
        if int(config.latent_dim) != self.latent_dim:
            raise ValueError(
                f"belf latent_dim={config.latent_dim} != artifact.X={self.latent_dim}"
            )
        self.sc_cfg = bool(config.sc_cfg)
        self.exit_kind = str(config.exit).strip().lower()
        if self.exit_kind not in ("decoder", "linear"):
            raise ValueError(f"exit 须为 decoder|linear，收到 {config.exit!r}")
        if self.exit_kind == "decoder":
            bundle.require_vae_dec(reason="exit=decoder")
        self.ce_detach_g = bool(config.ce_detach_g)
        self.cond_mode = str(config.cond_mode)
        self.clean_block_prob = float(config.clean_block_prob)
        self.lambda_mse = float(config.lambda_mse)
        self.lambda_ce = float(config.lambda_ce)
        self.lambda_s1 = float(config.lambda_s1)
        self.denoiser_p_mean = float(config.denoiser_p_mean)
        self.denoiser_p_std = float(config.denoiser_p_std)
        self.t_clean_eps = float(config.t_clean_eps)
        self.vel_eps = float(config.vel_eps)
        self.sc_p_mean = float(config.self_cond_cfg_p_mean)
        self.sc_p_std = float(config.self_cond_cfg_p_std)
        self.w_sc_min = float(config.self_cond_cfg_min)
        self.w_sc_max = float(config.self_cond_cfg_max)
        self.sc_guided_prob = float(config.sc_guided_prob)
        self.ctx_drop_prob = float(config.ctx_p_drop)
        self.denoiser_noise_scale = float(config.noise_sigma)
        self.attn_backend = str(config.attn_backend)
        self.whiten = bool(config.whiten)
        self.whiten_on = str(config.whiten_on).strip().lower()
        self.x0_source = str(config.x0_source).strip().lower()
        self.ctx_source = str(config.ctx_source).strip().lower()

        x_dim = self.latent_dim
        mean = torch.zeros(x_dim)
        std = torch.ones(x_dim)
        src = getattr(bundle.latent, "backbone", bundle.latent)
        if self.whiten_on == "z":
            names_m = ("whitening_mean_z", "whitening_mean", "latent_mean")
            names_s = ("whitening_std_z", "whitening_std", "latent_std")
        else:
            names_m = ("whitening_mean", "latent_mean")
            names_s = ("whitening_std", "latent_std")
        found_m = _whiten_vec(src, names_m, x_dim)
        found_s = _whiten_vec(src, names_s, x_dim)
        if found_m is not None:
            mean = found_m
        if found_s is not None:
            std = found_s
        self.register_buffer("whiten_mean", mean, persistent=True)
        self.register_buffer("whiten_std", std, persistent=True)

        bias = bool(config.proj_bias)
        self.in_proj = nn.Linear(x_dim, self.n_embd, bias=bias)
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        self.proj_norm = (
            _RMSNorm(self.n_embd)
            if str(config.proj_norm).lower() == "rmsnorm"
            else nn.Identity()
        )

        head_dim = self.n_embd // self.n_head
        rope_dim = config.rope_dim
        if rope_dim is None:
            rope_dim = head_dim if head_dim % 2 == 0 else head_dim - 1

        self.g = AdaLNZeroStack(
            self.n_embd,
            int(config.n_layer),
            self.n_head,
            out_dim=x_dim,
            dropout=float(config.dropout),
            mlp_ratio=float(config.mlp_ratio),
            rope_dim=rope_dim,
            rope_theta=float(config.rope_theta),
            qk_norm=bool(config.qk_norm),
            attn_backend=self.attn_backend,
            num_time_tokens=0,
            use_scale=self.sc_cfg,
            t_freq_dim=int(config.t_freq_dim),
        )
        self.sc_proj = (
            nn.Linear(2 * self.n_embd, self.n_embd, bias=True) if self.sc_cfg else None
        )
        if self.exit_kind == "decoder":
            self.exit_head = None
        else:
            self.exit_head = ExitMap(
                kind="linear",
                n_embd=self.n_embd,
                out_dim=int(self.token_layout.vocab_size),
                bias=bool(config.lm_head_bias),
            )

        levels = ladder_levels(
            self.time_step, self.denoiser_p_mean, self.denoiser_p_std, self.t_clean_eps,
        )
        self.register_buffer("levels", levels, persistent=True)
        self._mask_cache: dict[tuple[int, int, int, torch.device, bool], torch.Tensor] = {}

        self.last_l2_loss = float("nan")
        self.last_ce_loss = float("nan")
        self.last_s1_loss = float("nan")

    def _whiten(self, z: torch.Tensor) -> torch.Tensor:
        """可选白化，仍在流空间 X。"""
        if not self.whiten:
            return z
        std = self.whiten_std.clamp(min=1e-8).to(dtype=z.dtype, device=z.device)
        mean = self.whiten_mean.to(dtype=z.dtype, device=z.device)
        return (z - mean) / std

    def stem(self, x: torch.Tensor) -> torch.Tensor:
        """G 茎：已白化的 X→D。"""
        return self.proj_norm(self.in_proj(x))

    def map_x(self, z: torch.Tensor) -> torch.Tensor:
        """白化后的流空间 X。"""
        return self._whiten(z)

    def exit_logits(
        self,
        x_hat: torch.Tensor,
        hidden: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.exit_kind == "decoder":
            kwargs: dict[str, Any] = {}
            if key_padding_mask is not None:
                kwargs["key_padding_mask"] = key_padding_mask
            return self.bundle.decode_logits(x_hat, tokens=tokens, **kwargs)
        if hidden is None:
            raise ValueError("exit=linear 须传入 G 隐状态")
        return self.exit_head(hidden)

    def on_tokens_seen(self, n: int, optimizer: Any = None) -> bool:
        return self.bundle.on_tokens_seen(n, optimizer)

    def _raw_2l_mask(
        self,
        left_len: int,
        right_len: int,
        device: torch.device,
        *,
        parallel: bool = False,
    ) -> torch.Tensor:
        key = (left_len, right_len, self.block_size, device, parallel)
        cached = self._mask_cache.get(key)
        if cached is None:
            if parallel:
                if left_len != right_len:
                    raise ValueError(
                        f"并行块 mask 要求左右等长，收到 left={left_len}, right={right_len}"
                    )
                raw = pack_2l_parallel_blocks_mask(
                    left_len, self.block_size, device,
                )
            else:
                raw = pack_2l_mask(
                    left_len, right_len, self.block_size, self.block_size, device,
                )
            cached = raw
            if len(self._mask_cache) > 16:
                self._mask_cache.pop(next(iter(self._mask_cache)))
            self._mask_cache[key] = cached
        return cached

    def _cached_2l_mask(
        self,
        left_len: int,
        right_len: int,
        device: torch.device,
        *,
        parallel: bool = False,
    ) -> torch.Tensor:
        raw = self._raw_2l_mask(
            left_len, right_len, device, parallel=parallel,
        )
        cached = as_sdpa_mask(raw)
        if cached is None:
            raise RuntimeError("2L mask 不能为空")
        return cached

    def _apply_sc(
        self,
        packed: torch.Tensor,
        sc_right: torch.Tensor | None,
        left_len: int,
        *,
        known_right: int = 0,
        known_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """把右段 sc 拼进 2L 后经 ``sc_proj``；左段 / 已知槽为 0。"""
        if self.sc_proj is None:
            return packed
        sc = torch.zeros_like(packed)
        if sc_right is not None:
            sc[:, left_len:] = self.stem(sc_right)
        if known_right > 0:
            sc[:, left_len : left_len + known_right] = 0
        if known_mask is not None:
            sc[:, left_len:] = torch.where(
                known_mask.unsqueeze(-1), torch.zeros_like(sc[:, left_len:]), sc[:, left_len:],
            )
        return self.sc_proj(torch.cat([packed, sc], dim=-1))

    def forward_g(
        self,
        h_left: torch.Tensor,
        h_right: torch.Tensor,
        t_right: torch.Tensor | float,
        m_right_mode: int,
        w_sc: torch.Tensor | None,
        *,
        known_right: int = 0,
        drop_left: bool = False,
        sc_right: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """2L G：左右均为流空间 X，茎升到 D；返回右段 x-pred（X）。"""
        h_right = self.stem(h_right)
        h_left = self.stem(h_left)
        bsz, right_len, _ = h_right.shape
        left_len = int(h_left.size(1))
        device = h_right.device
        dtype = h_right.dtype
        left = h_left
        if drop_left:
            left = maybe_drop_left(left, self.ctx_drop_prob)
        packed = pack_2l(left, h_right)
        if isinstance(t_right, torch.Tensor):
            t_r = t_right.to(device=device, dtype=dtype)
            if t_r.ndim == 1:
                t_r = t_r[:, None].expand(bsz, right_len)
        else:
            t_r = h_right.new_full((bsz, right_len), float(t_right))
        m_r = torch.full(
            (bsz, right_len), int(m_right_mode), device=device, dtype=torch.long,
        )
        if known_right > 0:
            t_r = t_r.clone()
            t_r[:, :known_right] = 1.0
            m_r[:, :known_right] = MODE_NONE
        t_l = packed.new_ones(bsz, left_len)
        m_l = torch.zeros(bsz, left_len, device=device, dtype=torch.long)
        t_all = torch.cat([t_l, t_r], dim=1) if left_len > 0 else t_r
        m_all = torch.cat([m_l, m_r], dim=1) if left_len > 0 else m_r
        w_all = w_sc
        mask = self._cached_2l_mask(left_len, right_len, device)
        pos_l = torch.arange(left_len, device=device, dtype=torch.long)
        pos_r = torch.arange(
            left_len, left_len + right_len, device=device, dtype=torch.long,
        )
        positions = torch.cat([pos_l, pos_r]) if left_len > 0 else pos_r
        packed = self._apply_sc(
            packed, sc_right, left_len, known_right=known_right,
        )
        want_h = self.exit_kind == "linear"
        out = self.g(
            packed, t_all, w_all, m_all, attn_mask=mask, positions=positions,
            return_hidden=want_h,
        )
        if want_h:
            x_hat, hidden = out
            x_r = x_hat[:, left_len:] if left_len > 0 else x_hat
            h_r = hidden[:, left_len:] if left_len > 0 else hidden
            return x_r, h_r
        x_hat = out
        return x_hat[:, left_len:] if left_len > 0 else x_hat

    def _masked_mean(
        self, per_token: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.to(dtype=per_token.dtype)
        denom = mask_f.sum().clamp(min=1.0)
        return (per_token * mask_f).sum() / denom

    def train_metrics(self) -> dict[str, float]:
        return {
            "mse": self.last_l2_loss,
            "ce": self.last_ce_loss,
            "denoise_mse": self.last_l2_loss,
            "decode_ce": self.last_ce_loss,
            "s1": self.last_s1_loss,
        }

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets, kwargs
        tokens = idx
        if tokens.ndim != 2:
            raise ValueError(f"idx 须为 (B, L)，收到 {tuple(tokens.shape)}")
        bsz, seq_len = tokens.shape
        w = self.block_size
        if seq_len % w != 0:
            raise ValueError(f"序列长度 {seq_len} 须能被 block_size {w} 整除")
        if seq_len > self.max_seq_len:
            raise ValueError(f"序列长度 {seq_len} 超过 max_seq_len={self.max_seq_len}")
        device = tokens.device
        levels = self.levels.to(device=device)

        z, mu, logvar = self.bundle.encode(tokens, sample=self.training)
        if z.shape[:2] != tokens.shape:
            raise ValueError(
                f"latent 形状 {tuple(z.shape[:2])} 须与 token {tuple(tokens.shape)} 一致"
            )
        src_x0 = mu if self.x0_source == "mu" else z
        src_ctx = mu if self.ctx_source == "mu" else z
        x0 = self._whiten(src_x0)
        x_ctx = self._whiten(src_ctx)
        dtype = x0.dtype
        pad_id = int(self.token_layout.pad_token_id)
        not_pad = tokens != pad_id

        hops = torch.randint(0, self.time_step, (bsz,), device=device)
        t_hop = levels[hops].to(dtype=dtype)
        is_decode = hops == (self.time_step - 1)

        rem = torch.zeros(bsz, device=device, dtype=torch.long)
        b_cur = torch.zeros(bsz, device=device, dtype=torch.long)
        n_blocks = seq_len // w
        if (
            self.cond_mode == "clean"
            and self.clean_block_prob > 0
            and n_blocks > 0
            and w > 1
        ):
            use_mix = torch.rand(bsz, device=device) < self.clean_block_prob
            rem = torch.where(
                use_mix,
                torch.randint(1, w, (bsz,), device=device),
                rem,
            )
            b_cur = torch.where(
                use_mix,
                torch.randint(0, n_blocks, (bsz,), device=device),
                b_cur,
            )
        pos = torch.arange(seq_len, device=device)
        known = (pos[None, :] // w == b_cur[:, None]) & (pos[None, :] % w < rem[:, None])
        unknown = (~known) & not_pad
        denoise_mask = unknown & (~is_decode[:, None])
        decode_mask = unknown & is_decode[:, None]

        noise = torch.randn_like(x0) * self.denoiser_noise_scale
        is_pad = ~not_pad
        t_right = torch.where(
            known | is_pad,
            x0.new_ones(bsz, seq_len),
            t_hop[:, None].expand(bsz, seq_len),
        )
        x_known = x0
        if bool((rem > 0).any().item()):
            cond_tokens = tokens.clone()
            leak = (
                (pos[None, :] // w == b_cur[:, None])
                & (pos[None, :] % w >= rem[:, None])
                & (rem[:, None] > 0)
            )
            cond_tokens = cond_tokens.masked_fill(leak, pad_id)
            z_c, mu_c, _ = self.bundle.encode(cond_tokens, sample=self.training)
            src_c = mu_c if self.x0_source == "mu" else z_c
            x_known = self._whiten(src_c)
        z_t = interpolate(x0, t_right, noise)
        z_t = torch.where((known | is_pad).unsqueeze(-1), x_known, z_t)

        m_den = torch.full(
            (bsz, seq_len), MODE_DENOISE, device=device, dtype=torch.long,
        )
        m_dec = torch.full(
            (bsz, seq_len), MODE_DECODE, device=device, dtype=torch.long,
        )
        m_right = torch.where(is_decode[:, None], m_dec, m_den)
        m_right = torch.where(
            known | is_pad, torch.zeros_like(m_right), m_right,
        )

        h_left, drop_left = maybe_drop_left(
            x_ctx.detach(),
            self.ctx_drop_prob if self.training else 0.0,
            return_drop=True,
        )
        w_sc = None
        guided: torch.Tensor | bool = False
        if self.sc_cfg:
            w_sc = sample_w_sc(
                bsz, self.sc_p_mean, self.sc_p_std,
                self.w_sc_min, self.w_sc_max, device,
            ).to(dtype=dtype)
            guided = torch.rand(bsz, device=device) < self.sc_guided_prob

        v_z = v_star(x0, z_t, t_right, self.vel_eps)
        sc_zero = torch.zeros_like(z_t)
        need_teacher = (
            self.sc_cfg and self.training and bool((~is_decode).any().item())
        )
        if need_teacher:
            with torch.no_grad():
                x_u, _ = self._forward_g_m(
                    h_left, z_t, t_right, m_right, w_sc, sc_right=sc_zero,
                    drop_left=drop_left, is_pad=is_pad, unknown=unknown,
                )
                v_u = x_to_v(x_u, z_t, t_right, self.vel_eps)
                sc_cond = x_u.detach()
                sc_cond = torch.where(known.unsqueeze(-1), torch.zeros_like(sc_cond), sc_cond)
                sc_cond = torch.where(
                    is_decode[:, None, None], torch.zeros_like(sc_cond), sc_cond,
                )
                x_c, _ = self._forward_g_m(
                    h_left, z_t, t_right, m_right, w_sc, sc_right=sc_cond,
                    drop_left=drop_left, is_pad=is_pad, unknown=unknown,
                )
                v_c = x_to_v(x_c, z_t, t_right, self.vel_eps)
            g = guided.to(dtype=dtype)[:, None, None]
            sc_stu = x_u.detach() * g
            sc_stu = torch.where(known.unsqueeze(-1), torch.zeros_like(sc_stu), sc_stu)
            sc_stu = torch.where(
                is_decode[:, None, None], torch.zeros_like(sc_stu), sc_stu,
            )
            x_hat, hidden = self._forward_g_m(
                h_left, z_t, t_right, m_right, w_sc, sc_right=sc_stu,
                drop_left=drop_left, is_pad=is_pad, unknown=unknown,
            )
            v_tgt = blend_v_tgt(v_z, v_u, v_c, w_sc, guided)
            v_tgt = torch.where(denoise_mask.unsqueeze(-1), v_tgt, v_z)
        else:
            x_hat, hidden = self._forward_g_m(
                h_left, z_t, t_right, m_right, w_sc, sc_right=sc_zero,
                drop_left=drop_left, is_pad=is_pad, unknown=unknown,
            )
            v_tgt = v_z

        v_hat = x_to_v(x_hat, z_t, t_right, self.vel_eps)
        l2_tok = (v_hat - v_tgt).pow(2).mean(dim=-1)
        zero = x0.new_zeros(())
        mse = self._masked_mean(l2_tok, denoise_mask)
        self.last_l2_loss = mse.detach()

        # 出口按样本因果，只跑 decode hop 行；与整批再 mask 的 CE 相同。
        dec_rows = is_decode.nonzero(as_tuple=True)[0]
        if dec_rows.numel() == 0:
            keep_mod = self.exit_head if self.exit_head is not None else self.g
            ce = zero + keep_params_in_graph(keep_mod, zero)
            self.last_ce_loss = zero.new_full((), float("nan"))
        else:
            x_ce = x_hat.index_select(0, dec_rows)
            h_ce = hidden.index_select(0, dec_rows) if hidden is not None else None
            if self.ce_detach_g:
                x_ce = x_ce.detach()
                if h_ce is not None:
                    h_ce = h_ce.detach()
            logits = self.exit_logits(
                x_ce, h_ce, tokens=tokens.index_select(0, dec_rows),
            )
            ce_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tokens.index_select(0, dec_rows).reshape(-1),
                ignore_index=self.token_layout.ignore_index,
                reduction="none",
            ).view(dec_rows.size(0), seq_len)
            ce = self._masked_mean(ce_tok, decode_mask.index_select(0, dec_rows))
            self.last_ce_loss = ce.detach()
            del logits

        s1 = self.bundle.s1_loss(tokens, z=z, mu=mu, logvar=logvar)
        if self.bundle.is_trainable:
            self.last_s1_loss = s1.detach()
        else:
            self.last_s1_loss = float("nan")
            s1 = zero.to(dtype=x0.dtype)

        # 规格总目标为 L+L_s1，无外层 λ_s1；lambda_s1 仅留配置以免换哈希。
        loss = self.lambda_mse * mse + self.lambda_ce * ce + s1
        empty = tokens.new_zeros(
            bsz, 0, int(self.token_layout.vocab_size),
        )
        return empty, loss

    def _forward_g_m(
        self,
        h_left: torch.Tensor,
        h_right: torch.Tensor,
        t_right: torch.Tensor,
        m_right: torch.Tensor,
        w_sc: torch.Tensor | None,
        *,
        sc_right: torch.Tensor | None = None,
        drop_left: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
        unknown: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """与 ``forward_g`` 相同，但右段 ``m`` 逐列给定；训练走并行块 mask。

        ``h_left`` / ``h_right`` 为流空间 X；返回右段 ``(x_hat, hidden)``。
        """
        h_left = self.stem(h_left)
        h_right = self.stem(h_right)
        bsz, right_len, _ = h_right.shape
        left_len = int(h_left.size(1))
        device = h_right.device
        packed = pack_2l(h_left, h_right)
        t_l = packed.new_ones(bsz, left_len)
        m_l = torch.zeros(bsz, left_len, device=device, dtype=torch.long)
        t_all = torch.cat([t_l, t_right], dim=1)
        m_all = torch.cat([m_l, m_right], dim=1)
        parallel = left_len == right_len
        raw = self._raw_2l_mask(left_len, right_len, device, parallel=parallel)
        if is_pad is not None and unknown is not None:
            raw = hide_right_pad_from_unknown(
                raw, is_pad, unknown, self.block_size,
            )
        mask = as_sdpa_mask(raw)
        if mask is None:
            raise RuntimeError("2L mask 不能为空")
        if drop_left is not None:
            mask = hide_left_keys(mask, drop_left, left_len)
        positions = pair_positions(left_len, device) if left_len == right_len else (
            torch.cat([
                torch.arange(left_len, device=device, dtype=torch.long),
                torch.arange(left_len, left_len + right_len, device=device, dtype=torch.long),
            ])
        )
        packed = self._apply_sc(
            packed,
            sc_right,
            left_len,
            known_mask=(m_right == MODE_NONE) | (m_right == MODE_DECODE),
        )
        want_h = self.exit_kind == "linear"
        out = self.g(
            packed, t_all, w_sc, m_all, attn_mask=mask, positions=positions,
            return_hidden=want_h,
        )
        if want_h:
            x_hat, hidden = out
            return x_hat[:, left_len:], hidden[:, left_len:]
        return out[:, left_len:], None

    @torch.compiler.disable
    @torch.no_grad()
    def block_generate(
        self,
        num_samples: int = 1,
        seqlen: int | None = None,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
        bos_token_id: int | None = None,
        prefix_tokens: torch.Tensor | None = None,
        sampling_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        return run_block_generate(
            self,
            num_samples,
            seqlen,
            temperature=temperature,
            top_k=top_k,
            bos_token_id=bos_token_id,
            prefix_tokens=prefix_tokens,
            sampling_cfg=sampling_cfg,
        )

    @torch.compiler.disable
    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        seqlen: int | None = None,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
        bos_token_id: int | None = None,
        prefix_tokens: torch.Tensor | None = None,
        sampling_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        return self.block_generate(
            num_samples,
            seqlen,
            temperature=temperature,
            top_k=top_k,
            bos_token_id=bos_token_id,
            prefix_tokens=prefix_tokens,
            sampling_cfg=sampling_cfg,
        )


class FL_BelfModel(FL_PreTrainedModel):
    config_class = FL_BelfConfig
    eval_ppl_from_ce = True

    def __init__(self, config: FL_BelfConfig, bundle: LatentBundle) -> None:
        super().__init__(config)
        self.backbone = _BelfBackbone(config, bundle)
        self.post_init()
        self._restore_adaln_zero_init()

    def _restore_adaln_zero_init(self) -> None:
        """post_init 会打乱 AdaLN-Zero / FinalLayer 零初始化；按 Cola 意图补回。"""
        stack = self.backbone.g
        for block in stack.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(stack.final.adaLN_modulation[-1].weight)
        nn.init.zeros_(stack.final.adaLN_modulation[-1].bias)
        nn.init.zeros_(stack.final.linear.weight)
        nn.init.zeros_(stack.final.linear.bias)
        nn.init.zeros_(stack.mode_embed.weight)

    def on_tokens_seen(self, n: int, optimizer: Any = None) -> bool:
        return self.backbone.on_tokens_seen(n, optimizer)


def build_model_from_config(
    config: FL_BelfConfig,
    *,
    variant: str | None = None,
    load_latent_weights: bool = True,
) -> FL_BelfModel:
    del variant
    ensure_token_layout(config)
    if not load_latent_weights:
        raise ValueError("BELF 入口须经 LatentBundle 加载 artifacts，禁止跳过权重")
    bundle = LatentBundle(
        latent_model=config.latent_model,
        tag=config.tag,
        tune=config.latent_tune,
        latent_thaw_tokens=config.latent_thaw_tokens,
        lambda_vae=config.lambda_vae,
        lambda_ref=config.lambda_ref,
    )
    print(f"[belf] loaded latent {bundle.latent_model}/{bundle.tag}")
    return FL_BelfModel(config, bundle)


def build_model(model_cfg: dict) -> FL_BelfModel:
    data, sampling = split_model_cfg(model_cfg)
    data.pop("train_variant", None)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_BelfConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
