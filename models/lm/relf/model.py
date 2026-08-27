"""RELF：局部时间场滚动窗 + AdaLN-Zero G。梯子 / 2L / CFG 走 ``belf_relf_core``。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lm.belf_relf_core import (
    AdaLNZeroStack,
    ExitMap,
    LatentBundle,
    as_sdpa_mask,
    blend_v_tgt,
    check_time_step,
    group_causal_mask,
    interpolate,
    ladder_levels,
    maybe_drop_left,
    pack_2l,
    pack_2l_mask,
    sample_w_sc,
    v_star,
    validate_loaded_block,
    x_to_v,
)

from models.lm.relf.config import FL_RelfConfig
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    sample_from_logits,
    split_model_cfg,
)
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg

_MODE_NONE = 0
_MODE_DENOISE = 1
_MODE_DECODE = 2


def _tile_window_starts(seq_len: int, offset: int, window_size: int) -> list[int]:
    """按 ``W`` 铺不重叠窗；``offset>0`` 时在 ``offset-W`` 补一扇余段窗。"""
    if window_size < 1:
        raise ValueError(f"window_size 须 >= 1，收到 {window_size}")
    if offset < 0 or offset >= window_size:
        raise ValueError(f"offset 须在 [0, W)，收到 {offset} (W={window_size})")
    starts: list[int] = []
    if offset > 0:
        starts.append(offset - window_size)
    starts.extend(range(offset, seq_len, window_size))
    return starts


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(dtype=x.dtype)
        return self.weight.to(dtype=x.dtype) * x


class _RelfBackbone(nn.Module):
    """RELF 骨干：``LatentBundle`` + 映射 + AdaLN-Zero G + 出口。"""

    full_sequence_training = True
    supports_prefix = True
    # 一次 forward 按槽拆 MSE/CE；不要 ELF decoder_prob 抽支。
    dual_branch_logging = False
    mixed_branch_training = False

    def __init__(self, config: FL_RelfConfig, bundle: LatentBundle) -> None:
        super().__init__()
        self.config = config
        self.token_layout = config.token_layout()
        self.max_seq_len = int(config.max_seq_len)
        self.n_embd = int(config.n_embd)
        self.n_head = int(config.n_head)
        self.window_size = int(config.window_size)
        self.step_size = int(config.step_size)
        self.time_step = int(config.time_step)
        check_time_step(self.time_step)
        if self.step_size * self.time_step != self.window_size:
            raise ValueError(
                f"relf 要求 step_size*time_step==window_size，"
                f"收到 {self.step_size}*{self.time_step}!={self.window_size}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} 须能被 n_head={self.n_head} 整除")
        if str(config.exit).lower() not in ("decoder", "linear"):
            raise ValueError(f"exit 须为 decoder|linear，收到 {config.exit!r}")
        if str(config.proj_type).lower() not in ("linear", "bottleneck"):
            raise ValueError(f"proj_type 须为 linear|bottleneck，收到 {config.proj_type!r}")
        if str(config.attn_backend).lower() != "sdpa":
            raise ValueError(f"relf attn_backend 仅支持 sdpa，收到 {config.attn_backend!r}")
        if str(config.train_t_schedule).strip().lower() != "mixed":
            raise ValueError(
                f"relf train_t_schedule 锁死 mixed，收到 {config.train_t_schedule!r}"
            )
        if str(config.window_t).strip().lower() != "ladder":
            raise ValueError(
                f"relf window_t 锁死 ladder，收到 {config.window_t!r}"
            )

        self.latent = bundle
        validate_loaded_block(family="relf", loaded_block=bundle.block_size)
        self.latent_dim = int(bundle.latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("LatentBundle.latent_dim 无效")
        if str(config.exit).strip().lower() == "decoder":
            bundle.require_vae_dec(reason="exit=decoder")

        self.whiten = bool(config.whiten)
        self.whiten_on = str(config.whiten_on).strip().lower()
        self.lambda_s1 = float(config.lambda_s1)
        mean = torch.zeros(self.latent_dim)
        std = torch.ones(self.latent_dim)
        src = getattr(bundle.latent, "backbone", bundle.latent)
        if self.whiten_on == "z":
            names_m = ("whitening_mean_z", "whitening_mean", "latent_mean")
            names_s = ("whitening_std_z", "whitening_std", "latent_std")
        else:
            names_m = ("whitening_mean", "latent_mean")
            names_s = ("whitening_std", "latent_std")
        for name in names_m:
            val = getattr(src, name, None)
            if isinstance(val, torch.Tensor) and val.numel() == self.latent_dim:
                mean = val.detach().float().reshape(-1)
                break
        for name in names_s:
            val = getattr(src, name, None)
            if isinstance(val, torch.Tensor) and val.numel() == self.latent_dim:
                std = val.detach().float().reshape(-1)
                break
        self.register_buffer("whiten_mean", mean, persistent=True)
        self.register_buffer("whiten_std", std, persistent=True)

        x_dim = self.latent_dim
        d_dim = self.n_embd
        bias = bool(config.proj_bias)
        if str(config.proj_type).lower() == "bottleneck":
            bdim = int(config.bottleneck_dim)
            self.in_proj = nn.Sequential(
                nn.Linear(x_dim, bdim, bias=bias),
                nn.GELU(),
                nn.Linear(bdim, d_dim, bias=bias),
            )
        else:
            self.in_proj = nn.Linear(x_dim, d_dim, bias=bias)
        self.proj_norm = (
            _RMSNorm(d_dim) if str(config.proj_norm).lower() == "rmsnorm" else nn.Identity()
        )

        self.sc_cfg = bool(config.sc_cfg)
        self.sc_proj = nn.Linear(2 * d_dim, d_dim, bias=True) if self.sc_cfg else None
        self.denoiser = AdaLNZeroStack(
            d_dim,
            int(config.n_layer),
            self.n_head,
            out_dim=d_dim,
            dropout=float(config.dropout),
            mlp_ratio=float(config.mlp_ratio),
            rope_theta=float(config.rope_theta),
            qk_norm=bool(config.qk_norm),
            attn_backend="sdpa",
            num_time_tokens=0,
            use_scale=self.sc_cfg,
            t_freq_dim=int(config.t_freq_dim),
        )
        vocab = int(self.token_layout.vocab_size)
        self.exit_kind = str(config.exit).strip().lower()
        self.use_decoder_exit = self.exit_kind == "decoder"
        if self.use_decoder_exit:
            self.exit_head = ExitMap(
                kind="decoder",
                n_embd=d_dim,
                out_dim=x_dim,
                bias=bias,
            )
        else:
            self.exit_head = ExitMap(
                kind="linear",
                n_embd=d_dim,
                out_dim=vocab,
                bias=bool(config.lm_head_bias),
            )

        levels = ladder_levels(
            self.time_step,
            float(config.denoiser_p_mean),
            float(config.denoiser_p_std),
            float(config.t_clean_eps),
        ).float()
        self.register_buffer("ladder", levels, persistent=False)
        f_k = torch.stack(
            [levels[self.time_step - 1 - (k // self.step_size)] for k in range(self.window_size)]
        )
        self.register_buffer("F", f_k, persistent=False)
        self._ladder_cap = float(levels[-1].item())
        self._mask_base_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}
        self._gen_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

        self.last_l2_loss: torch.Tensor | float = float("nan")
        self.last_ce_loss: torch.Tensor | float = float("nan")
        self.last_s1_loss: torch.Tensor | float = float("nan")

    def on_tokens_seen(self, n: int, optimizer: Any = None) -> bool:
        return bool(self.latent.on_tokens_seen(n, optimizer))

    def train_metrics(self) -> dict[str, Any]:
        return {
            "denoise_mse": self.last_l2_loss,
            "decode_ce": self.last_ce_loss,
            "mse": self.last_l2_loss,
            "ce": self.last_ce_loss,
            "s1": self.last_s1_loss,
        }

    def _whiten(self, z: torch.Tensor) -> torch.Tensor:
        if not self.whiten:
            return z
        std = self.whiten_std.clamp_min(1e-6).to(dtype=z.dtype, device=z.device)
        mean = self.whiten_mean.to(dtype=z.dtype, device=z.device)
        return (z - mean) / std

    def _map_h(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj_norm(self.in_proj(self._whiten(z)))

    def _bos_eos(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """每条样本的 BOS / EOS 下标（含该 token）；无 EOS 则落到最后一个非 PAD。"""
        bos_id = int(self.token_layout.bos_token_id)
        eos_id = int(self.token_layout.eos_token_id)
        pad_id = int(self.token_layout.pad_token_id)
        bsz, seq_len = tokens.shape
        pos = torch.arange(seq_len, device=tokens.device)
        bos_hit = tokens == bos_id
        eos_hit = tokens == eos_id
        nonpad = tokens != pad_id
        bos = torch.where(
            bos_hit.any(dim=1),
            bos_hit.float().argmax(dim=1),
            torch.zeros(bsz, device=tokens.device, dtype=torch.long),
        )
        last = nonpad.long().sum(dim=1).clamp(min=1) - 1
        eos = torch.where(eos_hit.any(dim=1), eos_hit.float().argmax(dim=1), last)
        del pos
        return bos, eos

    def _plan_windows(
        self,
        tokens: torch.Tensor,
        bos: torch.Tensor,
        eos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """按 ``W`` 铺窗；返回 ``u,k0,known_r,active`` 各 ``(B, n_win)``。

        训练抽一个 S 对齐偏移。未句首切时以 ``clean_block_prob`` 抽余数爬梯。
        评测偏移固定 0。
        """
        bsz, seq_len = tokens.shape
        device = tokens.device
        w_sz = self.window_size
        step = self.step_size
        t_steps = self.time_step
        n_phase = w_sz // step
        n_win = (seq_len + w_sz - 1) // w_sz + 1
        pad_id = int(self.token_layout.pad_token_id)
        p_clean = float(self.config.clean_block_prob)
        use_clean = (
            str(self.config.cond_mode).lower() == "clean"
            and step > 1
            and bool(self.training)
        )
        if self.training and n_phase > 1:
            off = torch.randint(0, n_phase, (bsz,), device=device) * step
        else:
            off = torch.zeros(bsz, device=device, dtype=torch.long)
        slot = torch.arange(n_win, device=device)
        u = off[:, None] + (slot - 1) * w_sz
        k = torch.arange(w_sz, device=device)
        j = u[:, :, None] + k
        valid = (
            (j >= bos[:, None, None])
            & (j <= eos[:, None, None])
            & (j >= 0)
            & (j < seq_len)
        )
        safe_j = j.clamp(0, max(seq_len - 1, 0))
        not_pad = tokens.gather(1, safe_j.reshape(bsz, -1)).reshape(
            bsz, n_win, w_sz,
        ) != pad_id
        active = (valid & not_pad).any(dim=-1)
        empty = ~active.any(dim=1)
        first = slot == 0
        active = active | (empty[:, None] & first)
        u = torch.where(empty[:, None] & first, torch.zeros_like(u), u)

        k0 = torch.zeros(bsz, n_win, device=device, dtype=torch.long)
        known_r = torch.zeros(bsz, n_win, device=device, dtype=torch.long)
        if use_clean and p_clean > 0:
            bos_cut = u < bos[:, None]
            draw = (
                active & (~bos_cut)
                & (torch.rand(bsz, n_win, device=device) < p_clean)
            )
            h = torch.randint(0, t_steps, (bsz, n_win), device=device)
            k0_h = step * (t_steps - 1 - h)
            n_prime = (
                (k >= k0_h[:, :, None]) & valid & not_pad
            ).sum(dim=-1)
            r_hi = torch.minimum(
                torch.full((), step - 1, device=device, dtype=torch.long),
                n_prime - 1,
            )
            can_r = draw & (r_hi >= 1)
            k0 = torch.where(can_r, k0_h, k0)
            known_r = torch.where(
                can_r,
                1 + (torch.rand(bsz, n_win, device=device) * r_hi.clamp(min=1).float()).to(
                    dtype=torch.long,
                ),
                known_r,
            )
        return u, k0, known_r, active

    def _windows_attn_mask(
        self,
        left_len: int,
        u: torch.Tensor,
        active: torch.Tensor,
        k0: torch.Tensor | None = None,
        in_win: torch.Tensor | None = None,
        drop_left: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, 1, two, two)`` SDPA 加性掩码：左因果，各窗看 ``left[:u+k0]`` + 窗内组因果。

        ``in_win`` 为 ``(B, n_win, W)``：丢掉格不当 key；dummy 只留自注意。
        ``drop_left`` 为 ``(B,)``：切断该样本左段对右段的可见性。
        """
        device = u.device
        bsz, n_win = u.shape
        w_sz = self.window_size
        right_len = n_win * w_sz
        two = left_len + right_len
        key = (left_len, n_win, device)
        base = self._mask_base_cache.get(key)
        if base is None:
            base = u.new_zeros(two, two, dtype=torch.bool)
            if left_len > 0:
                q = torch.arange(left_len, device=device)
                base[:left_len, :left_len] = q[None, :] <= q[:, None]
            ridx = torch.arange(right_len, device=device)
            loc = ridx % w_sz
            win_vis = group_causal_mask(w_sz, self.step_size, device=device)
            same = (ridx[:, None] // w_sz) == (ridx[None, :] // w_sz)
            base[left_len:, left_len:] = same & win_vis[loc[:, None], loc[None, :]]
            if len(self._mask_base_cache) > 16:
                self._mask_base_cache.pop(next(iter(self._mask_base_cache)))
            self._mask_base_cache[key] = base
        vis = base.expand(bsz, two, two).clone()
        ridx = torch.arange(right_len, device=device)
        u_clamped = u.clamp(min=0, max=left_len)
        u_of_r = u_clamped[:, ridx // w_sz]
        kv_end = u_of_r
        if k0 is not None:
            k0_of_r = k0[:, ridx // w_sz].clamp(min=0)
            kv_end = (u_of_r + k0_of_r).clamp(max=left_len)
        if left_len > 0:
            left_idx = torch.arange(left_len, device=device)
            vis[:, left_len:, :left_len] = left_idx[None, None, :] < kv_end[:, :, None]
        act_r = active[:, ridx // w_sz]
        vis[:, left_len:, :] = vis[:, left_len:, :] & act_r[:, :, None]
        eye_r = ridx[:, None] == ridx[None, :]
        vis[:, left_len:, left_len:] = vis[:, left_len:, left_len:] | (
            (~act_r[:, :, None]) & eye_r
        )
        if in_win is not None:
            in_flat = in_win.reshape(bsz, right_len)
            vis[:, left_len:, left_len:] = (
                vis[:, left_len:, left_len:] & in_flat[:, None, :]
            )
            vis[:, left_len:, left_len:] = vis[:, left_len:, left_len:] | (
                (~in_flat)[:, :, None] & eye_r
            )
        if drop_left is not None and left_len > 0:
            vis[drop_left, left_len:, :left_len] = False
        mask = as_sdpa_mask(vis)
        if mask is None:
            raise RuntimeError("2L mask 不能为空")
        return mask

    def _run_g(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
        w_sc: torch.Tensor | None,
        m: torch.Tensor,
        sc: torch.Tensor | None,
        *,
        attn_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.sc_proj is not None:
            if sc is None:
                sc = torch.zeros_like(h)
            x = self.sc_proj(torch.cat([h, sc], dim=-1))
        else:
            x = h
        return self.denoiser(
            x, t, w_sc, m, attn_mask=attn_mask, positions=positions,
        )

    def _exit_logits(self, x_hat: torch.Tensor) -> torch.Tensor:
        if self.config.ce_detach_g:
            x_hat = x_hat.detach()
        mapped = self.exit_head(x_hat)
        if self.use_decoder_exit:
            return self.latent.decode_logits(mapped)
        return mapped

    def _exit_right_windows(
        self,
        x_hat: torch.Tensor,
        left_len: int,
        u: torch.Tensor,
        k0: torch.Tensor,
    ) -> torch.Tensor:
        """每个窗 ``cat(x̂_left[<g], x̂_win)`` 再因果读出；``g=u+k0``。"""
        bsz, _, d_dim = x_hat.shape
        n_win = int(u.size(1))
        w_sz = self.window_size
        x_left = x_hat[:, :left_len] if left_len > 0 else x_hat[:, :0]
        x_right = x_hat[:, left_len:] if left_len > 0 else x_hat
        vocab = int(self.token_layout.vocab_size)
        if not self.use_decoder_exit:
            return self._exit_logits(x_right)
        g = (u + k0).clamp(min=0, max=left_len)
        out = x_hat.new_zeros(bsz, n_win * w_sz, vocab)
        for wi in range(n_win):
            g_w = g[:, wi]
            max_g = int(g_w.max().item()) if left_len > 0 else 0
            win = x_right[:, wi * w_sz : (wi + 1) * w_sz]
            dec_in = x_hat.new_zeros(bsz, max_g + w_sz, d_dim)
            if max_g > 0:
                for b in range(bsz):
                    gi = int(g_w[b].item())
                    if gi > 0:
                        dec_in[b, max_g - gi : max_g] = x_left[b, :gi]
            dec_in[:, max_g:] = win
            out[:, wi * w_sz : (wi + 1) * w_sz] = self._exit_logits(dec_in)[:, max_g:]
        return out

    def _pack_forward(
        self,
        tokens: torch.Tensor,
        h_ctx: torch.Tensor,
        h_x0: torch.Tensor,
        u: torch.Tensor,
        k0: torch.Tensor,
        known_r: torch.Tensor,
        active: torch.Tensor,
        bos: torch.Tensor,
        eos: torch.Tensor,
        *,
        compute_loss: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """一次 2L：``[n | n_win·W]``，与 BELF 同形的单次 G。"""
        device = tokens.device
        dtype = h_x0.dtype
        bsz, seq_len, d_dim = h_x0.shape
        n_win = int(u.size(1))
        w_sz = self.window_size
        step = self.step_size
        is_pad = tokens == int(self.token_layout.pad_token_id)
        ignore = int(self.token_layout.ignore_index)
        vel_eps = float(self.config.vel_eps)
        left_len = int(seq_len)
        right_len = n_win * w_sz

        h_left, drop_left = maybe_drop_left(
            h_ctx[:, :seq_len].detach(),
            float(self.config.ctx_p_drop) if (self.training and compute_loss) else 0.0,
            return_drop=True,
        )
        noise = torch.randn_like(h_x0) * float(self.config.noise_sigma)
        w_vec = h_x0.new_zeros(bsz)
        guided = torch.zeros(bsz, device=device, dtype=dtype)
        if self.sc_cfg and self.training and compute_loss:
            w_vec = sample_w_sc(
                bsz,
                float(self.config.self_cond_cfg_p_mean),
                float(self.config.self_cond_cfg_p_std),
                float(self.config.self_cond_cfg_min),
                float(self.config.self_cond_cfg_max),
                device,
            ).to(dtype=dtype)
            guided = (
                torch.rand(bsz, device=device) < float(self.config.sc_guided_prob)
            ).to(dtype=dtype)

        k = torch.arange(w_sz, device=device)
        j = u[:, :, None] + k
        i_bos = bos[:, None, None]
        i_eos = eos[:, None, None]
        in_win = (
            active[:, :, None]
            & (j >= i_bos) & (j <= i_eos)
            & (j < seq_len) & (j >= 0)
            & (k >= k0[:, :, None])
        )
        safe_j = j.clamp(0, max(seq_len - 1, 0))
        in_win = in_win & (
            ~is_pad.gather(1, safe_j.reshape(bsz, -1)).reshape(bsz, n_win, w_sz)
        )
        known = (
            (k >= k0[:, :, None])
            & (k < (k0[:, :, None] + known_r[:, :, None]))
            & in_win
            & (j != i_eos)
        )
        t_w = self.F.to(device=device).view(1, 1, w_sz)
        t_w = torch.where(
            in_win & (~known),
            t_w.expand(bsz, n_win, w_sz),
            torch.ones(bsz, n_win, w_sz, device=device, dtype=t_w.dtype),
        )
        cap = t_w.new_tensor(self._ladder_cap)
        unknown = in_win & (~known)
        is_dec = unknown & (t_w >= cap - 1e-8)
        is_den = unknown & (~is_dec)
        m_w = torch.zeros(bsz, n_win, w_sz, device=device, dtype=torch.long)
        m_w = torch.where(is_den, torch.full_like(m_w, _MODE_DENOISE), m_w)
        m_w = torch.where(is_dec, torch.full_like(m_w, _MODE_DECODE), m_w)

        valid_pos = (j >= 0) & (j < seq_len)
        idx = safe_j.reshape(bsz, right_len, 1).expand(bsz, right_len, d_dim)
        x0 = h_x0.gather(1, idx).reshape(bsz, n_win, w_sz, d_dim)
        nse = noise.gather(1, idx).reshape(bsz, n_win, w_sz, d_dim)
        x0 = torch.where(valid_pos.unsqueeze(-1), x0, torch.zeros_like(x0))
        nse = torch.where(valid_pos.unsqueeze(-1), nse, torch.zeros_like(nse))
        z = interpolate(x0, t_w, nse)
        z = torch.where(known.unsqueeze(-1), x0, z)
        z = torch.where(in_win.unsqueeze(-1), z, torch.zeros_like(z))

        h_right = z.reshape(bsz, right_len, d_dim)
        x0_right = x0.reshape(bsz, right_len, d_dim)
        t_right = t_w.reshape(bsz, right_len)
        m_right = m_w.reshape(bsz, right_len)
        md = is_den.to(dtype).reshape(bsz, right_len)
        mc = is_dec.to(dtype).reshape(bsz, right_len)
        tok_g = tokens.gather(1, safe_j.reshape(bsz, right_len)).reshape(
            bsz, n_win, w_sz,
        )
        tgt_right = torch.where(
            in_win & valid_pos, tok_g, torch.full_like(tok_g, ignore),
        ).reshape(bsz, right_len)

        attn = self._windows_attn_mask(
            left_len, u, active, k0, in_win=in_win, drop_left=drop_left,
        )
        t_left = h_right.new_ones(bsz, left_len)
        m_left = torch.zeros(bsz, left_len, device=device, dtype=torch.long)
        h = pack_2l(h_left, h_right)
        t_all = torch.cat([t_left, t_right], dim=1) if left_len > 0 else t_right
        m_all = torch.cat([m_left, m_right], dim=1) if left_len > 0 else m_right
        pos_left = torch.arange(left_len, device=device).expand(bsz, -1)
        pos_right = j.clamp(0, self.max_seq_len - 1).reshape(bsz, right_len)
        positions = (
            torch.cat([pos_left, pos_right], dim=1) if left_len > 0 else pos_right
        )
        w_pos = None
        if self.sc_cfg:
            den = (m_all == _MODE_DENOISE).to(dtype)
            w_pos = w_vec[:, None] * den

        v_z = v_star(x0_right, h_right, t_right, vel_eps)
        sc = torch.zeros_like(h) if self.sc_cfg else None
        v_u: torch.Tensor | None = None
        v_c: torch.Tensor | None = None
        need_teacher = (
            self.sc_cfg
            and self.training
            and compute_loss
            and bool((md > 0).any().item())
        )
        if need_teacher:
            with torch.no_grad():
                x_hat_u = self._run_g(
                    h, t_all, w_pos, m_all, torch.zeros_like(h),
                    attn_mask=attn, positions=positions,
                )
                x_u_r = x_hat_u[:, left_len:] if left_len > 0 else x_hat_u
                v_u = x_to_v(x_u_r, h_right, t_right, vel_eps)
                sc_cond = x_u_r.detach().clone().reshape(bsz, n_win, w_sz, d_dim)
                sc_cond[:, :, w_sz - step :, :] = 0
                sc_cond = sc_cond * is_den.unsqueeze(-1)
                sc_r_cond = sc_cond.reshape(bsz, right_len, d_dim)
                sc_teacher = torch.zeros_like(h)
                if left_len > 0:
                    sc_teacher[:, left_len:] = sc_r_cond
                else:
                    sc_teacher = sc_r_cond
                x_hat_c = self._run_g(
                    h, t_all, w_pos, m_all, sc_teacher,
                    attn_mask=attn, positions=positions,
                )
                x_c_r = x_hat_c[:, left_len:] if left_len > 0 else x_hat_c
                v_c = x_to_v(x_c_r, h_right, t_right, vel_eps)
            sc_stu = sc_r_cond * guided[:, None, None]
            sc = torch.zeros_like(h)
            if left_len > 0:
                sc[:, left_len:] = sc_stu
            else:
                sc = sc_stu

        x_hat = self._run_g(
            h, t_all, w_pos if self.sc_cfg else None, m_all,
            sc if self.sc_cfg else None,
            attn_mask=attn, positions=positions,
        )
        x_right = x_hat[:, left_len:] if left_len > 0 else x_hat
        v_hat = x_to_v(x_right, h_right, t_right, vel_eps)
        if v_u is not None and v_c is not None:
            v_tgt = blend_v_tgt(v_z, v_u, v_c, w_vec, guided)
            v_tgt = torch.where(md.unsqueeze(-1) > 0, v_tgt, v_z)
        else:
            v_tgt = v_z
        mse_tok = (v_hat - v_tgt).pow(2).mean(dim=-1)
        mse = (mse_tok * md).sum() / md.sum().clamp(min=1.0)

        log_r = self._exit_right_windows(x_hat, left_len, u, k0)
        ce_tok = F.cross_entropy(
            log_r.reshape(-1, log_r.size(-1)),
            tgt_right.reshape(-1),
            ignore_index=ignore,
            reduction="none",
        ).view(bsz, right_len)
        ce = (ce_tok * mc).sum() / mc.sum().clamp(min=1.0)
        loss = float(self.config.lambda_mse) * mse + float(self.config.lambda_ce) * ce
        return loss, mse, ce

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
        if tokens.size(1) > self.max_seq_len:
            raise ValueError(
                f"序列长度 {tokens.size(1)} 超过 max_seq_len={self.max_seq_len}"
            )
        z, mu, logvar = self.latent.encode(tokens, sample=self.training)
        if z.shape[:2] != tokens.shape:
            raise ValueError(
                f"latent 形状 {tuple(z.shape[:2])} 须与 token {tuple(tokens.shape)} 一致"
            )
        src_x0 = mu if str(self.config.x0_source).lower() == "mu" else z
        src_ctx = mu if str(self.config.ctx_source).lower() == "mu" else z
        h_x0 = self._map_h(src_x0)
        h_ctx = self._map_h(src_ctx)
        bos, eos = self._bos_eos(tokens)
        u, k0, known_r, active = self._plan_windows(tokens, bos, eos)
        loss, mse, ce = self._pack_forward(
            tokens, h_ctx, h_x0, u, k0, known_r, active, bos, eos, compute_loss=True,
        )
        s1 = self.latent.s1_loss(tokens, z=z, mu=mu, logvar=logvar)
        self.last_l2_loss = mse.detach()
        self.last_ce_loss = ce.detach()
        self.last_s1_loss = s1.detach()
        total = loss + self.lambda_s1 * s1
        empty = torch.zeros(
            tokens.size(0), 0, int(self.token_layout.vocab_size),
            device=tokens.device,
        )
        return empty, total

    def _slot_next_t(self, t: torch.Tensor) -> torch.Tensor:
        """升一档：接到梯子上更高的 ``L_i``（更干净）。"""
        levels = self.ladder.to(device=t.device, dtype=t.dtype)
        # 找当前档 i 使 L_i ≈ t，下一档 min(i+1, T-1)
        dist = (t.unsqueeze(-1) - levels.view(*([1] * t.ndim), -1)).abs()
        idx = dist.argmin(dim=-1)
        nxt = (idx + 1).clamp(max=int(self.time_step) - 1)
        return levels[nxt]

    def _sde_or_ode(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        v: torch.Tensor,
        *,
        method: str,
        gamma: float,
        last_flow: torch.Tensor,
    ) -> torch.Tensor:
        """``last_flow`` 槽关 churn（ODE）；其余 SDE。decode 槽调用方不更新。"""
        h = t_next - t
        if method != "sde":
            return z + h.unsqueeze(-1) * v
        alpha = (1.0 - float(gamma) * h).clamp(0.0, 1.0)
        t_back = alpha * t
        eps = torch.randn_like(z) * float(self.config.noise_sigma)
        z_back = alpha.unsqueeze(-1) * z + (1.0 - alpha).unsqueeze(-1) * eps
        z_sde = z_back + (t_next - t_back).unsqueeze(-1) * v
        z_ode = z + h.unsqueeze(-1) * v
        use_sde = (~last_flow).unsqueeze(-1).to(dtype=z.dtype)
        return use_sde * z_sde + (1.0 - use_sde) * z_ode

    def _sample_tokens(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        return sample_from_logits(logits, temperature=temperature, top_k=top_k)

    @torch.compiler.disable
    @torch.no_grad()
    def rolling_generate(
        self,
        num_samples: int = 1,
        seqlen: int | None = None,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        bos_token_id: int | None = None,
        prefix_tokens: torch.Tensor | None = None,
        sampling_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        cfg = dict(self.config.sampling or {})
        if sampling_cfg:
            cfg.update(sampling_cfg)
        seqlen = int(seqlen or self.max_seq_len)
        seqlen = min(seqlen, self.max_seq_len)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        method = str(cfg.get("sampling_method", "sde")).lower()
        gamma = float(cfg.get("sde_gamma", 1.5))
        temperature = float(cfg.get("temperature", temperature))
        top_k = cfg.get("top_k", top_k)
        if top_k is not None:
            top_k = int(top_k)
        commit_x0 = bool(cfg.get("commit_x0hat", True))
        w_sc_val = float(cfg.get("w_sc", cfg.get("self_cond_cfg_scale", 3.0)))
        w_ctx = float(cfg.get("w_ctx", cfg.get("ctx_cfg_scale", 1.0)))
        bos_id = int(
            self.token_layout.bos_token_id if bos_token_id is None else bos_token_id
        )
        eos_id = int(self.token_layout.eos_token_id)
        pad_id = int(self.token_layout.pad_token_id)
        w_sz = self.window_size
        step = self.step_size
        cap = float(self.ladder[-1].item())

        if prefix_tokens is not None:
            prefix = prefix_tokens.to(device=device, dtype=torch.long)
            if prefix.size(0) != num_samples:
                raise ValueError("prefix_tokens batch must match num_samples")
        else:
            prefix = torch.zeros((num_samples, 0), device=device, dtype=torch.long)

        tokens = prefix.clone()
        nfe = 0
        alive = torch.ones(num_samples, dtype=torch.bool, device=device)
        if tokens.size(1) > 0:
            alive = alive & ~(tokens == eos_id).any(dim=1)

        def _encode_h(tok: torch.Tensor) -> torch.Tensor:
            z, mu, _ = self.latent.encode(tok, sample=False)
            src = mu if str(self.config.x0_source).lower() == "mu" else z
            return self._map_h(src)

        def _one_g(
            h_left: torch.Tensor,
            z_win: torch.Tensor,
            t_win: torch.Tensor,
            m_win: torch.Tensor,
            sc_win: torch.Tensor | None,
            pos_left: torch.Tensor,
            pos_win: torch.Tensor,
            *,
            drop_left: bool,
            in_win: torch.Tensor | None = None,
        ) -> torch.Tensor:
            nonlocal nfe
            bsz = z_win.size(0)
            left_len = int(h_left.size(1))
            mk = (left_len, z_win.device)
            raw = self._gen_mask_cache.get(mk)
            if raw is None:
                raw = pack_2l_mask(left_len, w_sz, 1, step, device=z_win.device)
                if len(self._gen_mask_cache) > 32:
                    self._gen_mask_cache.pop(next(iter(self._gen_mask_cache)))
                self._gen_mask_cache[mk] = raw
            vis = raw
            if in_win is not None:
                two = left_len + w_sz
                vis = raw.expand(bsz, two, two).clone()
                vis[:, left_len:, left_len:] = (
                    vis[:, left_len:, left_len:] & in_win[:, None, :]
                )
                ridx = torch.arange(w_sz, device=device)
                eye = ridx[:, None] == ridx[None, :]
                vis[:, left_len:, left_len:] = vis[:, left_len:, left_len:] | (
                    (~in_win)[:, :, None] & eye
                )
            attn = as_sdpa_mask(vis)
            if attn is None:
                raise RuntimeError("2L mask 不能为空")
            if drop_left and left_len > 0:
                h_left = torch.zeros_like(h_left)
                two = left_len + w_sz
                if vis.ndim == 2:
                    vis = vis.expand(bsz, two, two).clone()
                else:
                    vis = vis.clone()
                vis[:, left_len:, :left_len] = False
                attn = as_sdpa_mask(vis)
                if attn is None:
                    raise RuntimeError("2L mask 不能为空")
            h = pack_2l(h_left, z_win)
            t = torch.cat(
                [
                    torch.ones(bsz, left_len, device=device, dtype=dtype),
                    t_win,
                ],
                dim=1,
            )
            m = torch.cat(
                [
                    torch.zeros(bsz, left_len, device=device, dtype=torch.long),
                    m_win,
                ],
                dim=1,
            )
            w_pos = torch.zeros(bsz, left_len + w_sz, device=device, dtype=dtype)
            if self.sc_cfg:
                den = (m == _MODE_DENOISE).to(dtype)
                w_pos = den * w_sc_val
                # decode 列 AdaLN 不吃 w；栈内已按 m 屏蔽。
            sc = torch.zeros_like(h)
            if self.sc_cfg and sc_win is not None:
                sc[:, left_len:] = sc_win
            positions = torch.cat([pos_left, pos_win], dim=0)
            x_hat = self._run_g(
                h, t, w_pos if self.sc_cfg else None, m, sc if self.sc_cfg else None,
                attn_mask=attn, positions=positions,
            )
            nfe += 1
            return x_hat

        def _fields_for_u(u: int, tok: torch.Tensor, *, target_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """推理窗：BOS 左切、EOS/目标长右切，其余铺 ``F``。"""
            bsz = tok.size(0)
            bos_hit = tok == bos_id
            bos_idx = torch.where(
                bos_hit.any(dim=1),
                bos_hit.float().argmax(dim=1),
                torch.zeros(bsz, device=device, dtype=torch.long),
            )
            has_eos = (tok == eos_id).any(dim=1)
            eos_idx = torch.where(
                has_eos,
                (tok == eos_id).float().argmax(dim=1),
                torch.full((bsz,), target_len - 1, device=device, dtype=torch.long),
            )
            k = torch.arange(w_sz, device=device)
            j = u + k
            in_win = (
                (j[None, :] >= bos_idx[:, None])
                & (j[None, :] <= eos_idx[:, None])
                & (j >= 0)
                & (j < target_len)
            )
            f = self.F.to(device=device, dtype=dtype)
            t_w = torch.where(
                in_win, f.expand(bsz, -1), torch.ones(bsz, w_sz, device=device, dtype=dtype),
            )
            dec = f >= (cap - 1e-8)
            m_w = torch.where(
                in_win,
                torch.where(
                    dec.expand(bsz, -1),
                    torch.full((bsz, w_sz), _MODE_DECODE, device=device, dtype=torch.long),
                    torch.full((bsz, w_sz), _MODE_DENOISE, device=device, dtype=torch.long),
                ),
                torch.zeros(bsz, w_sz, device=device, dtype=torch.long),
            )
            return t_w, m_w, in_win

        def _append_decoded(
            tok: torch.Tensor,
            sampled: torch.Tensor,
            dec: torch.Tensor,
            alive: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            is_eos = dec & (sampled == eos_id)
            prev_eos = is_eos.cumsum(dim=1) - is_eos.long()
            already = (
                (tok == eos_id).any(dim=1)
                if tok.size(1) > 0
                else torch.zeros(tok.size(0), device=device, dtype=torch.bool)
            )
            keep = dec & (prev_eos == 0) & alive[:, None] & (~already)[:, None]
            hit = already | is_eos.any(dim=1)
            counts = keep.sum(dim=1)
            max_n = int(counts.max().item())
            if max_n == 0:
                return tok, hit
            idx = torch.arange(w_sz, device=device).expand_as(keep)
            order = torch.where(keep, idx, torch.full_like(idx, w_sz)).sort(dim=1).values
            take = order[:, :max_n]
            valid = take < w_sz
            gathered = sampled.gather(1, take.clamp(max=w_sz - 1))
            add = torch.where(valid, gathered, torch.full_like(gathered, pad_id))
            return torch.cat([tok, add], dim=1), hit

        h_left_cache: torch.Tensor | None = _encode_h(tokens) if tokens.size(1) > 0 else None
        z_carry: torch.Tensor | None = None
        sc_carry: torch.Tensor | None = None

        while tokens.size(1) < seqlen:
            L = int(tokens.size(1))
            if not bool(alive.any().item()):
                break
            if bool((tokens == eos_id).any(dim=1).all()):
                break
            r = L % step
            g = (L // step) * step

            if L == 0:
                # preroll：BOS 自己爬梯，空前缀不注入已写 BOS。
                sc_win = torch.zeros(num_samples, w_sz, self.n_embd, device=device, dtype=dtype)
                z_win = torch.randn(
                    num_samples, w_sz, self.n_embd, device=device, dtype=dtype,
                ) * float(self.config.noise_sigma)
                h_left = z_win.new_zeros(num_samples, 0, self.n_embd)
                for hop in range(self.time_step):
                    k_bos = step * (self.time_step - hop - 1)
                    u = -k_bos
                    t_w, m_w, in_win = _fields_for_u(u, tokens, target_len=seqlen)
                    m_w = torch.where(in_win, m_w, torch.zeros_like(m_w))
                    pos_left = torch.arange(0, device=device)
                    pos_win = torch.arange(u, u + w_sz, device=device).clamp(
                        min=0, max=self.max_seq_len - 1,
                    )
                    sc_in = sc_win.clone()
                    sc_in[:, w_sz - step :] = 0
                    x_hat = _one_g(
                        h_left, z_win, t_w, m_w, sc_in, pos_left, pos_win,
                        drop_left=False, in_win=in_win,
                    )
                    x_right = x_hat[:, :w_sz]
                    v = x_to_v(x_right, z_win, t_w, float(self.config.vel_eps))
                    den = m_w == _MODE_DENOISE
                    t_next = self._slot_next_t(t_w)
                    last_flow = den & (t_next >= cap - 1e-8)
                    z_win = torch.where(
                        den.unsqueeze(-1),
                        self._sde_or_ode(
                            z_win, t_w, t_next, v,
                            method=method, gamma=gamma, last_flow=last_flow,
                        ),
                        z_win,
                    )
                    sc_win = x_right.detach()
                    sc_win[:, w_sz - step :] = 0
                    if hop < self.time_step - 1:
                        z_win = torch.cat(
                            [
                                z_win[:, step:],
                                torch.randn(
                                    num_samples, step, self.n_embd, device=device, dtype=dtype,
                                ) * float(self.config.noise_sigma),
                            ],
                            dim=1,
                        )
                        sc_win = torch.cat(
                            [sc_win[:, step:], torch.zeros_like(sc_win[:, :step])],
                            dim=1,
                        )
                    else:
                        logits = self._exit_logits(x_hat)[:, :w_sz]
                        sampled = self._sample_tokens(
                            logits, temperature=temperature, top_k=top_k,
                        )
                        tokens, hit = _append_decoded(
                            tokens, sampled, m_w == _MODE_DECODE, alive,
                        )
                        alive = alive & ~hit
                        if commit_x0:
                            h_left_cache = x_right[:, :step]
                        else:
                            h_left_cache = _encode_h(tokens)
                        z_carry = torch.cat(
                            [
                                z_win[:, step:],
                                torch.randn(
                                    num_samples, step, self.n_embd, device=device, dtype=dtype,
                                ) * float(self.config.noise_sigma),
                            ],
                            dim=1,
                        )
                        sc_carry = torch.cat(
                            [x_right[:, step:], torch.zeros_like(x_right[:, :step])],
                            dim=1,
                        )
                continue

            if L > 0 and r > 0:
                # 余数爬梯：与训练单帧 (h,r) 同构。完整 pop [0,g) 进 KV。
                # hop0 按 F 初始化；之后只钉已知、不重插值；中间 hop 不 pop。
                sc_win = torch.zeros(num_samples, w_sz, self.n_embd, device=device, dtype=dtype)
                z_win = torch.randn(
                    num_samples, w_sz, self.n_embd, device=device, dtype=dtype,
                ) * float(self.config.noise_sigma)
                h_all = h_left_cache if h_left_cache is not None else _encode_h(tokens)
                hop0 = True
                for hop in range(self.time_step):
                    k0 = step * (self.time_step - hop - 1)
                    u = g - k0
                    t_w, m_w, in_win = _fields_for_u(u, tokens, target_len=seqlen)
                    kk = torch.arange(w_sz, device=device)
                    in_win = in_win & (kk[None, :] >= k0)
                    f = self.F.to(device=device, dtype=dtype)
                    t_w = torch.where(
                        in_win,
                        f.expand(tokens.size(0), -1),
                        torch.ones(tokens.size(0), w_sz, device=device, dtype=dtype),
                    )
                    dec = f >= (cap - 1e-8)
                    m_w = torch.where(
                        in_win,
                        torch.where(
                            dec.expand(tokens.size(0), -1),
                            torch.full(
                                (tokens.size(0), w_sz),
                                _MODE_DECODE,
                                device=device,
                                dtype=torch.long,
                            ),
                            torch.full(
                                (tokens.size(0), w_sz),
                                _MODE_DENOISE,
                                device=device,
                                dtype=torch.long,
                            ),
                        ),
                        torch.zeros(tokens.size(0), w_sz, device=device, dtype=torch.long),
                    )
                    jj = u + kk
                    known = (
                        (kk >= k0) & (kk < k0 + r) & (jj >= 0) & (jj < L)
                    ).expand(tokens.size(0), -1)
                    m_w = torch.where(known, torch.zeros_like(m_w), m_w)
                    t_w = torch.where(known, torch.ones_like(t_w), t_w)
                    m_w = torch.where(in_win & (~known), m_w, torch.zeros_like(m_w))
                    h_left = h_all[:, :g] if g > 0 else h_all[:, :0]
                    gathered: torch.Tensor | None = None
                    if h_all.size(1) > 0:
                        safe = jj.clamp(0, h_all.size(1) - 1)
                        gathered = h_all[:, safe]
                        if hop0:
                            valid = ((jj >= 0) & (jj < L)).expand(tokens.size(0), -1)
                            unk = in_win & (~known) & valid
                            nse = torch.randn_like(gathered) * float(self.config.noise_sigma)
                            mixed = interpolate(gathered, t_w, nse)
                            z_win = torch.where(unk.unsqueeze(-1), mixed, z_win)
                            hop0 = False
                        z_win = torch.where(known.unsqueeze(-1), gathered, z_win)
                    pos_left = torch.arange(h_left.size(1), device=device)
                    pos_win = torch.arange(u, u + w_sz, device=device).clamp(
                        min=0, max=self.max_seq_len - 1,
                    )
                    sc_in = sc_win.clone()
                    sc_in[:, w_sz - step :] = 0
                    sc_in = torch.where(known.unsqueeze(-1), torch.zeros_like(sc_in), sc_in)
                    x_hat = _one_g(
                        h_left, z_win, t_w, m_w, sc_in, pos_left, pos_win,
                        drop_left=False, in_win=in_win,
                    )
                    x_right = x_hat[:, h_left.size(1) :]
                    if w_ctx != 1.0 and h_left.size(1) > 0:
                        x_u = _one_g(
                            h_left, z_win, t_w, m_w, sc_in, pos_left, pos_win,
                            drop_left=True, in_win=in_win,
                        )
                        v = x_to_v(x_u[:, h_left.size(1) :], z_win, t_w, float(self.config.vel_eps))
                        v = v + w_ctx * (
                            x_to_v(x_right, z_win, t_w, float(self.config.vel_eps)) - v
                        )
                    else:
                        v = x_to_v(x_right, z_win, t_w, float(self.config.vel_eps))
                    den = m_w == _MODE_DENOISE
                    t_next = self._slot_next_t(t_w)
                    last_flow = den & (t_next >= cap - 1e-8)
                    z_win = torch.where(
                        den.unsqueeze(-1),
                        self._sde_or_ode(
                            z_win, t_w, t_next, v,
                            method=method, gamma=gamma, last_flow=last_flow,
                        ),
                        z_win,
                    )
                    if gathered is not None:
                        z_win = torch.where(known.unsqueeze(-1), gathered, z_win)
                    sc_win = x_right.detach()
                    sc_win[:, w_sz - step :] = 0
                    sc_win = torch.where(known.unsqueeze(-1), torch.zeros_like(sc_win), sc_win)
                    past_end = bool((tokens == eos_id).any(dim=1).all()) or (g + step >= seqlen)
                    extra = (
                        torch.zeros(
                            num_samples, step, self.n_embd, device=device, dtype=dtype,
                        )
                        if past_end
                        else torch.randn(
                            num_samples, step, self.n_embd, device=device, dtype=dtype,
                        ) * float(self.config.noise_sigma)
                    )
                    if hop < self.time_step - 1:
                        z_win = torch.cat([z_win[:, step:], extra], dim=1)
                        sc_win = torch.cat(
                            [sc_win[:, step:], torch.zeros_like(sc_win[:, :step])],
                            dim=1,
                        )
                    else:
                        logits = self._exit_logits(x_hat)[:, h_left.size(1) :]
                        sampled = self._sample_tokens(
                            logits, temperature=temperature, top_k=top_k,
                        )
                        tokens, hit = _append_decoded(
                            tokens, sampled, m_w == _MODE_DECODE, alive,
                        )
                        alive = alive & ~hit
                        if commit_x0:
                            pop = x_right[:, :step]
                            if gathered is not None:
                                pop = torch.where(
                                    known[:, :step].unsqueeze(-1),
                                    gathered[:, :step],
                                    pop,
                                )
                            left_pref = h_all[:, :g] if g > 0 else h_all[:, :0]
                            h_left_cache = torch.cat([left_pref, pop], dim=1)
                        else:
                            h_left_cache = _encode_h(tokens)
                        z_carry = torch.cat([z_win[:, step:], extra], dim=1)
                        sc_carry = torch.cat(
                            [x_right[:, step:], torch.zeros_like(x_right[:, :step])], dim=1,
                        )
                continue

            # preroll / freeroll：r=0。窗从当前长度起，整窗未知铺 F。
            u = L
            t_w, m_w, in_win = _fields_for_u(u, tokens, target_len=seqlen)
            m_w = torch.where(in_win, m_w, torch.zeros_like(m_w))
            if z_carry is None:
                z_win = torch.randn(
                    num_samples, w_sz, self.n_embd, device=device, dtype=dtype,
                ) * float(self.config.noise_sigma)
            else:
                z_win = z_carry
            if h_left_cache is None:
                h_left = z_win.new_zeros(num_samples, 0, self.n_embd)
            else:
                h_left = h_left_cache
            sc_win = sc_carry if sc_carry is not None else torch.zeros_like(z_win)
            sc_win = sc_win.clone()
            sc_win[:, w_sz - step :] = 0
            pos_left = torch.arange(h_left.size(1), device=device)
            pos_win = torch.arange(u, u + w_sz, device=device).clamp(max=self.max_seq_len - 1)
            x_hat = _one_g(
                h_left, z_win, t_w, m_w, sc_win, pos_left, pos_win,
                drop_left=False, in_win=in_win,
            )
            x_right = x_hat[:, h_left.size(1) :]
            if w_ctx != 1.0 and h_left.size(1) > 0:
                x_u = _one_g(
                    h_left, z_win, t_w, m_w, sc_win, pos_left, pos_win,
                    drop_left=True, in_win=in_win,
                )
                v = x_to_v(x_u[:, h_left.size(1) :], z_win, t_w, float(self.config.vel_eps))
                v = v + w_ctx * (
                    x_to_v(x_right, z_win, t_w, float(self.config.vel_eps)) - v
                )
            else:
                v = x_to_v(x_right, z_win, t_w, float(self.config.vel_eps))
            den = m_w == _MODE_DENOISE
            t_next = self._slot_next_t(t_w)
            last_flow = den & (t_next >= cap - 1e-8)
            z_win = torch.where(
                den.unsqueeze(-1),
                self._sde_or_ode(
                    z_win, t_w, t_next, v, method=method, gamma=gamma, last_flow=last_flow,
                ),
                z_win,
            )
            logits = self._exit_logits(x_hat)[:, h_left.size(1) :]
            sampled = self._sample_tokens(logits, temperature=temperature, top_k=top_k)
            dec = m_w == _MODE_DECODE
            new_len = int(tokens.size(1))
            tokens, hit = _append_decoded(tokens, sampled, dec, alive)
            alive = alive & ~hit
            if int(tokens.size(1)) == new_len:
                if (not bool(in_win.any())) or u + step >= seqlen:
                    break
            if commit_x0:
                h_left_cache = torch.cat([h_left, x_right[:, :step]], dim=1)
            else:
                h_left_cache = _encode_h(tokens)
            past_end = (not bool(alive.any().item())) or (u + step >= seqlen)
            extra = (
                torch.zeros(
                    num_samples, step, self.n_embd, device=device, dtype=dtype,
                )
                if past_end
                else torch.randn(
                    num_samples, step, self.n_embd, device=device, dtype=dtype,
                ) * float(self.config.noise_sigma)
            )
            z_carry = torch.cat([z_win[:, step:], extra], dim=1)
            sc_carry = torch.cat(
                [x_right[:, step:], torch.zeros_like(x_right[:, :step])], dim=1,
            )

        if tokens.size(1) < seqlen:
            pad = torch.full(
                (num_samples, seqlen - tokens.size(1)), pad_id,
                device=device, dtype=torch.long,
            )
            tokens = torch.cat([tokens, pad], dim=1)
        else:
            tokens = tokens[:, :seqlen]
        return tokens, nfe

    def generate(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, int]:
        return self.rolling_generate(*args, **kwargs)


class FL_RelfModel(FL_PreTrainedModel):
    config_class = FL_RelfConfig

    def __init__(self, config: FL_RelfConfig, bundle: LatentBundle) -> None:
        super().__init__(config)
        self.backbone = _RelfBackbone(config, bundle)
        self.post_init()
        self._restore_adaln_zero_init()

    def _restore_adaln_zero_init(self) -> None:
        stack = self.backbone.denoiser
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
    config: FL_RelfConfig,
    *,
    variant: str | None = None,
    load_latent_weights: bool = True,
    latent: nn.Module | None = None,
) -> FL_RelfModel:
    del variant
    ensure_token_layout(config)
    if latent is not None:
        bundle = LatentBundle(
            latent_model=config.latent_model,
            tag=config.tag,
            latent=latent,
            tune=config.latent_tune,
            latent_thaw_tokens=config.latent_thaw_tokens,
            lambda_vae=config.lambda_vae,
            lambda_ref=config.lambda_ref,
        )
    elif load_latent_weights:
        bundle = LatentBundle(
            config.latent_model,
            config.tag,
            tune=config.latent_tune,
            latent_thaw_tokens=config.latent_thaw_tokens,
            lambda_vae=config.lambda_vae,
            lambda_ref=config.lambda_ref,
        )
    else:
        raise ValueError("relf 须 load_latent_weights=True 或注入 latent")
    return FL_RelfModel(config, bundle)


def build_model(model_cfg: dict) -> FL_RelfModel:
    data, sampling = split_model_cfg(model_cfg)
    variant = data.pop("train_variant", None)
    load_lat = bool(data.pop("load_latent_weights", True))
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_RelfConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(
        config, variant=variant, load_latent_weights=load_lat,
    )
