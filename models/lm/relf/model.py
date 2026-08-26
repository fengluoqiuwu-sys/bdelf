"""RELF：局部时间场滚动窗 + AdaLN-Zero G。梯子 / 2L / CFG 走 ``belf_relf_core``。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lm.belf_relf_core import (
    AdaLNZeroStack,
    LatentBundle,
    blend_v_tgt,
    check_time_step,
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
from torch.utils.checkpoint import checkpoint

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


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(dtype=x.dtype)
        return self.weight.to(dtype=x.dtype) * x


class _CausalExit(nn.Module):
    """等宽因果 decoder：G 的 ``x_hat`` → logits。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_layer: int,
        vocab_size: int,
        dropout: float,
        *,
        lm_head_bias: bool,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} 须能被 n_head={n_head} 整除")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.layers = nn.ModuleList()
        for _ in range(int(n_layer)):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(n_embd),
                        "attn": nn.Linear(n_embd, 3 * n_embd),
                        "proj": nn.Linear(n_embd, n_embd),
                        "ln2": nn.LayerNorm(n_embd),
                        "mlp": nn.Sequential(
                            nn.Linear(n_embd, 4 * n_embd),
                            nn.GELU(),
                            nn.Linear(4 * n_embd, n_embd),
                            nn.Dropout(dropout),
                        ),
                        "drop": nn.Dropout(dropout),
                    }
                )
            )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=lm_head_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        for layer in self.layers:
            h = layer["ln1"](x)
            qkv = layer["attn"](h)
            q, k, v = qkv.split(h.size(-1), dim=-1)

            def _shape(t: torch.Tensor) -> torch.Tensor:
                return t.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

            y = F.scaled_dot_product_attention(
                _shape(q), _shape(k), _shape(v), is_causal=True,
                dropout_p=layer["drop"].p if self.training else 0.0,
            )
            y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
            x = x + layer["drop"](layer["proj"](y))
            x = x + layer["mlp"](layer["ln2"](x))
        return self.lm_head(self.ln_f(x))


@dataclass
class _WindowPlan:
    u: int
    k0: int
    known_r: int
    bos_cut: bool
    eos_cut: bool


class _RelfBackbone(nn.Module):
    """RELF 骨干：``LatentBundle`` + 映射 + AdaLN-Zero G + 出口。"""

    full_sequence_training = True
    supports_prefix = True
    # 一次 forward 按槽拆 MSE/CE；不要 ELF decoder_prob 抽支。
    dual_branch_logging = False
    mixed_branch_training = False
    # 每块最多叠这么多窗，避免 2L 全长物化（左全序 + n_win×W）。
    _win_chunk = 8

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

        self.latent = bundle
        validate_loaded_block(family="relf", loaded_block=bundle.block_size)
        self.latent_dim = int(bundle.latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("LatentBundle.latent_dim 无效")

        self.whiten = bool(config.whiten)
        self.register_buffer("whiten_mean", torch.zeros(self.latent_dim), persistent=True)
        self.register_buffer("whiten_std", torch.ones(self.latent_dim), persistent=True)

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
        if str(config.exit).lower() == "linear":
            self.exit_head: nn.Module = nn.Linear(
                d_dim, vocab, bias=bool(config.lm_head_bias),
            )
            self.use_decoder_exit = False
        else:
            self.exit_head = _CausalExit(
                d_dim, self.n_head, int(config.n_exit_layer), vocab,
                float(config.dropout), lm_head_bias=bool(config.lm_head_bias),
            )
            self.use_decoder_exit = True

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

    def _plan_windows(self, tokens: torch.Tensor) -> list[list[_WindowPlan]]:
        """合法 S 对齐窗；未句首切时以 ``clean_block_prob`` 抽余数爬梯一帧。"""
        bsz, seq_len = tokens.shape
        bos, eos = self._bos_eos(tokens)
        w_sz = self.window_size
        step = self.step_size
        t_steps = self.time_step
        pad_id = int(self.token_layout.pad_token_id)
        p_clean = float(self.config.clean_block_prob)
        use_clean = str(self.config.cond_mode).lower() == "clean" and step > 1
        out: list[list[_WindowPlan]] = []
        for b in range(bsz):
            i_bos = int(bos[b].item())
            i_eos = int(eos[b].item())
            plans: list[_WindowPlan] = []
            for u in range(0, seq_len, step):
                if u >= seq_len:
                    break
                lo = max(u, i_bos)
                hi = min(u + w_sz - 1, i_eos, seq_len - 1)
                if lo > hi:
                    continue
                if not bool((tokens[b, lo : hi + 1] != pad_id).any()):
                    continue
                bos_cut = u < i_bos
                eos_cut = u + w_sz - 1 > i_eos
                k0 = 0
                known_r = 0
                if use_clean and (not bos_cut) and (torch.rand((), device=tokens.device) < p_clean):
                    h = int(torch.randint(0, t_steps, (1,), device=tokens.device).item())
                    k0 = step * (t_steps - h - 1)
                    # 真窗长：从 k0 起到 EOS / 右端。
                    n_prime = 0
                    for k in range(k0, w_sz):
                        j = u + k
                        if j > i_eos or j >= seq_len:
                            break
                        if int(tokens[b, j].item()) == pad_id:
                            continue
                        n_prime += 1
                    r_hi = min(step - 1, n_prime - 1)
                    if r_hi >= 1:
                        known_r = int(torch.randint(1, r_hi + 1, (1,), device=tokens.device).item())
                    else:
                        k0 = 0
                plans.append(
                    _WindowPlan(
                        u=u, k0=k0, known_r=known_r, bos_cut=bos_cut, eos_cut=eos_cut,
                    )
                )
            if not plans:
                # 至少放一个从 0 起的窗，避免空 pack。
                plans.append(_WindowPlan(u=0, k0=0, known_r=0, bos_cut=False, eos_cut=False))
            out.append(plans)
        return out

    def _window_fields(
        self,
        plan: _WindowPlan,
        *,
        i_bos: int,
        i_eos: int,
        seq_len: int,
        is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 ``(in_win, known, t, m, md)`` 各 ``(W,)``；``mc`` 由 t 推出。"""
        device = is_pad.device
        w_sz = self.window_size
        k = torch.arange(w_sz, device=device)
        j = plan.u + k
        in_win = (j >= i_bos) & (j <= i_eos) & (j < seq_len) & (j >= 0) & (k >= plan.k0)
        safe_j = j.clamp(0, seq_len - 1)
        in_win = in_win & (~is_pad[safe_j])
        known = torch.zeros(w_sz, device=device, dtype=torch.bool)
        if plan.known_r > 0:
            known = (k >= plan.k0) & (k < plan.k0 + plan.known_r) & in_win
            # EOS 不当已知。
            known = known & (j != i_eos)
        t = self.F.to(device=device)
        t = torch.where(in_win & (~known), t, torch.ones_like(t))
        cap = float(self.ladder[-1].item())
        unknown = in_win & (~known)
        is_dec = unknown & (t >= cap - 1e-8)
        is_den = unknown & (~is_dec)
        m = torch.full((w_sz,), _MODE_NONE, device=device, dtype=torch.long)
        m = torch.where(is_den, torch.full_like(m, _MODE_DENOISE), m)
        m = torch.where(is_dec, torch.full_like(m, _MODE_DECODE), m)
        return in_win, known, t, m, is_den

    def _build_pack_mask(
        self,
        left_len: int,
        plans: list[_WindowPlan],
        n_win: int,
        device: torch.device,
    ) -> torch.Tensor:
        """多窗右段拼接：各窗独立 ``pack_2l_mask``，禁止跨窗右段互看。"""
        w_sz = self.window_size
        right_len = n_win * w_sz
        two = left_len + right_len
        vis = torch.zeros(two, two, device=device, dtype=torch.bool)
        if left_len > 0:
            q = torch.arange(left_len, device=device)[:, None]
            kv = torch.arange(left_len, device=device)[None, :]
            vis[:left_len, :left_len] = kv <= q
        for i, plan in enumerate(plans):
            u = min(max(int(plan.u), 0), left_len)
            local = pack_2l_mask(u, w_sz, 1, self.step_size, device=device)
            r0 = left_len + i * w_sz
            if u > 0:
                vis[r0 : r0 + w_sz, :u] = local[u:, :u]
            vis[r0 : r0 + w_sz, r0 : r0 + w_sz] = local[u:, u:]
        eye = torch.eye(w_sz, device=device, dtype=torch.bool)
        for i in range(len(plans), n_win):
            r0 = left_len + i * w_sz
            vis[r0 : r0 + w_sz, r0 : r0 + w_sz] = eye
        return vis

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
        return self.exit_head(x_hat)

    def _forward_win_chunk(
        self,
        tokens: torch.Tensor,
        h_left_full: torch.Tensor,
        h_x0: torch.Tensor,
        plans: list[list[_WindowPlan]],
        *,
        noise: torch.Tensor,
        i0: int,
        n_chunk: int,
        orig_left: int,
        w_vec: torch.Tensor,
        guided: torch.Tensor,
        compute_loss: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """一块最多 ``n_chunk`` 窗的 2L。返回 MSE/CE 的 (分子, 分母)。"""
        device = tokens.device
        dtype = h_x0.dtype
        bsz, seq_len, d_dim = h_x0.shape
        w_sz = self.window_size
        step = self.step_size
        bos, eos = self._bos_eos(tokens)
        is_pad = tokens == int(self.token_layout.pad_token_id)
        ignore = int(self.token_layout.ignore_index)
        vel_eps = float(self.config.vel_eps)

        max_u = 0
        for plist in plans:
            for plan in plist:
                max_u = max(max_u, int(plan.u))
        left_len = int(max_u)
        right_len = n_chunk * w_sz
        h_left = h_left_full[:, :left_len]
        h_right = h_x0.new_zeros(bsz, right_len, d_dim)
        t_right = torch.ones(bsz, right_len, device=device, dtype=dtype)
        m_right = torch.zeros(bsz, right_len, device=device, dtype=torch.long)
        md = torch.zeros(bsz, right_len, device=device, dtype=dtype)
        mc = torch.zeros(bsz, right_len, device=device, dtype=dtype)
        tgt_right = torch.full(
            (bsz, right_len), ignore, device=device, dtype=torch.long,
        )
        x0_right = h_x0.new_zeros(bsz, right_len, d_dim)
        vis_b: list[torch.Tensor] = []

        for b in range(bsz):
            plist = plans[b]
            i_bos = int(bos[b].item())
            i_eos = int(eos[b].item())
            vis_b.append(self._build_pack_mask(left_len, plist, n_chunk, device))
            for i, plan in enumerate(plist):
                r0 = i * w_sz
                in_win, known, t_w, m_w, is_den = self._window_fields(
                    plan, i_bos=i_bos, i_eos=i_eos, seq_len=seq_len, is_pad=is_pad[b],
                )
                sl = slice(plan.u, min(plan.u + w_sz, seq_len))
                local_n = sl.stop - sl.start
                x0 = h_x0.new_zeros(w_sz, d_dim)
                nse = noise.new_zeros(w_sz, d_dim)
                if local_n > 0:
                    x0[:local_n] = h_x0[b, sl]
                    nse[:local_n] = noise[b, sl]
                z = interpolate(x0, t_w, nse)
                z = torch.where(known.unsqueeze(-1), x0, z)
                z = torch.where(in_win.unsqueeze(-1), z, torch.zeros_like(z))
                h_right[b, r0 : r0 + w_sz] = z
                x0_right[b, r0 : r0 + w_sz] = x0
                t_right[b, r0 : r0 + w_sz] = t_w
                m_right[b, r0 : r0 + w_sz] = m_w
                md[b, r0 : r0 + w_sz] = is_den.to(dtype)
                mc[b, r0 : r0 + w_sz] = (m_w == _MODE_DECODE).to(dtype)
                if local_n > 0:
                    tgt_right[b, r0 : r0 + local_n] = tokens[b, sl]
                    tgt_right[b, r0 : r0 + w_sz].masked_fill_(
                        ~in_win, ignore,
                    )
        attn = torch.stack(vis_b, dim=0)
        t_left = h_right.new_ones(bsz, left_len)
        m_left = torch.zeros(bsz, left_len, device=device, dtype=torch.long)
        h = pack_2l(h_left, h_right)
        t_all = torch.cat([t_left, t_right], dim=1) if left_len > 0 else t_right
        m_all = torch.cat([m_left, m_right], dim=1) if left_len > 0 else m_right
        pos_left = torch.arange(left_len, device=device)
        pos_right = torch.arange(
            orig_left + i0 * w_sz,
            orig_left + i0 * w_sz + right_len,
            device=device,
        )
        positions = torch.cat([pos_left, pos_right]) if left_len > 0 else pos_right
        w_pos = None
        if self.sc_cfg:
            den = (m_all == _MODE_DENOISE).to(dtype)
            w_pos = w_vec[:, None] * den

        v_z = v_star(x0_right, h_right, t_right, vel_eps)
        has_den = bool((md.sum() > 0).item())
        guided_any = bool((guided > 0).any().item())
        sc = torch.zeros_like(h) if self.sc_cfg else None
        v_u: torch.Tensor | None = None
        if (
            self.sc_cfg
            and self.training
            and compute_loss
            and has_den
            and guided_any
        ):
            with torch.no_grad():
                x_hat_u = self._run_g(
                    h, t_all, w_pos, m_all, torch.zeros_like(h),
                    attn_mask=attn, positions=positions,
                )
                x_u_r = x_hat_u[:, left_len:] if left_len > 0 else x_hat_u
                v_u = x_to_v(x_u_r, h_right, t_right, vel_eps)
                sc_stu = x_u_r.detach().clone()
                sc_stu = sc_stu.reshape(bsz, n_chunk, w_sz, d_dim)
                sc_stu[:, :, w_sz - step :, :] = 0
                sc_stu = sc_stu.reshape(bsz, right_len, d_dim)
                sc = torch.zeros_like(h)
                sc_r = sc_stu * guided[:, None, None]
                if left_len > 0:
                    sc[:, left_len:] = sc_r
                else:
                    sc = sc_r

        x_hat = self._run_g(
            h, t_all, w_pos if self.sc_cfg else None, m_all,
            sc if self.sc_cfg else None,
            attn_mask=attn, positions=positions,
        )
        x_right = x_hat[:, left_len:] if left_len > 0 else x_hat
        v_hat = x_to_v(x_right, h_right, t_right, vel_eps)
        if v_u is not None:
            v_c = x_to_v(x_right.detach(), h_right, t_right, vel_eps)
            v_tgt = blend_v_tgt(v_z, v_u, v_c, w_vec, guided)
            v_tgt = torch.where(md.unsqueeze(-1) > 0, v_tgt, v_z)
        else:
            v_tgt = v_z
        mse_tok = (v_hat - v_tgt).pow(2).mean(dim=-1)
        mse_num = (mse_tok * md).sum()
        mse_den = md.sum()

        logits = self._exit_logits(x_hat)
        log_r = logits[:, left_len:] if left_len > 0 else logits
        ce_tok = F.cross_entropy(
            log_r.reshape(-1, log_r.size(-1)),
            tgt_right.reshape(-1),
            ignore_index=ignore,
            reduction="none",
        ).view(bsz, right_len)
        ce_num = (ce_tok * mc).sum()
        ce_den = mc.sum()
        return mse_num, mse_den, ce_num, ce_den

    def _pack_forward(
        self,
        tokens: torch.Tensor,
        h_ctx: torch.Tensor,
        h_x0: torch.Tensor,
        plans: list[list[_WindowPlan]],
        *,
        compute_loss: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """分块 2L，避免一次物化 ``L + n_win·W``。返回 ``(loss, mse, ce)``。"""
        device = tokens.device
        dtype = h_x0.dtype
        bsz, seq_len, _ = h_x0.shape
        n_win = max(len(p) for p in plans)
        h_left = maybe_drop_left(
            h_ctx[:, :seq_len].detach(),
            float(self.config.ctx_p_drop) if (self.training and compute_loss) else 0.0,
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

        mse_num = h_x0.new_zeros(())
        mse_den = h_x0.new_zeros(())
        ce_num = h_x0.new_zeros(())
        ce_den = h_x0.new_zeros(())
        chunk = max(1, int(self._win_chunk))
        for i0 in range(0, n_win, chunk):
            n_chunk = min(chunk, n_win - i0)
            plans_c = [p[i0 : i0 + n_chunk] for p in plans]

            def _ck(
                h_left_t: torch.Tensor,
                h_x0_t: torch.Tensor,
                noise_t: torch.Tensor,
                w_vec_t: torch.Tensor,
                guided_t: torch.Tensor,
                _i0: int = i0,
                _n: int = n_chunk,
                _plans: list[list[_WindowPlan]] = plans_c,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                return self._forward_win_chunk(
                    tokens, h_left_t, h_x0_t, _plans,
                    noise=noise_t, i0=_i0, n_chunk=_n, orig_left=seq_len,
                    w_vec=w_vec_t, guided=guided_t, compute_loss=compute_loss,
                )

            if self.training and compute_loss:
                a, b, c, d = checkpoint(
                    _ck, h_left, h_x0, noise, w_vec, guided,
                    use_reentrant=False,
                )
            else:
                a, b, c, d = _ck(h_left, h_x0, noise, w_vec, guided)
            mse_num = mse_num + a
            mse_den = mse_den + b
            ce_num = ce_num + c
            ce_den = ce_den + d

        mse = mse_num / mse_den.clamp(min=1.0)
        ce = ce_num / ce_den.clamp(min=1.0)
        if not bool((ce_den > 0).item()):
            ce = ce + 0.0 * sum(p.sum() for p in self.exit_head.parameters())
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
        plans = self._plan_windows(tokens)
        loss, mse, ce = self._pack_forward(
            tokens, h_ctx, h_x0, plans, compute_loss=True,
        )
        s1 = self.latent.s1_loss(tokens, z=z, mu=mu, logvar=logvar)
        self.last_l2_loss = mse.detach() if bool(torch.isfinite(mse).item()) else float("nan")
        self.last_ce_loss = ce.detach() if bool(torch.isfinite(ce).item()) else float("nan")
        self.last_s1_loss = s1.detach() if bool(torch.isfinite(s1).item()) else float("nan")
        total = loss + s1
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
            prefix = torch.full((num_samples, 1), bos_id, device=device, dtype=torch.long)

        tokens = prefix.clone()
        nfe = 0

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
        ) -> torch.Tensor:
            nonlocal nfe
            bsz = z_win.size(0)
            left_len = h_left.size(1)
            vis = pack_2l_mask(left_len, w_sz, 1, step, device=z_win.device)
            attn = vis
            if drop_left and left_len > 0:
                h_left = torch.zeros_like(h_left)
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
            t_w = self.F.to(device=device, dtype=dtype).expand(bsz, -1).clone()
            m_w = torch.full((bsz, w_sz), _MODE_NONE, device=device, dtype=torch.long)
            in_win = torch.zeros(bsz, w_sz, device=device, dtype=torch.bool)
            has_eos = (tok == eos_id).any(dim=1)
            bos_idx = torch.zeros(bsz, device=device, dtype=torch.long)
            bos_hit = tok == bos_id
            bos_idx = torch.where(bos_hit.any(dim=1), bos_hit.float().argmax(dim=1), bos_idx)
            eos_idx = torch.where(
                has_eos,
                (tok == eos_id).float().argmax(dim=1),
                torch.full((bsz,), target_len - 1, device=device, dtype=torch.long),
            )
            for b in range(bsz):
                i_bos = int(bos_idx[b].item())
                i_eos = int(eos_idx[b].item())
                for k in range(w_sz):
                    j = u + k
                    if j < i_bos or j > i_eos or j >= target_len:
                        continue
                    in_win[b, k] = True
                    t_w[b, k] = self.F[k]
                    m_w[b, k] = (
                        _MODE_DECODE if float(self.F[k].item()) >= cap - 1e-8 else _MODE_DENOISE
                    )
            return t_w, m_w, in_win

        def _append_decoded(
            tok: torch.Tensor,
            sampled: torch.Tensor,
            dec: torch.Tensor,
        ) -> tuple[torch.Tensor, bool]:
            rows: list[list[int]] = []
            stop = False
            for b in range(tok.size(0)):
                row: list[int] = []
                for k in range(w_sz):
                    if not bool(dec[b, k]):
                        continue
                    tid = int(sampled[b, k].item())
                    row.append(tid)
                    if tid == eos_id:
                        stop = True
                        break
                rows.append(row)
            max_n = max((len(r) for r in rows), default=0)
            if max_n == 0:
                return tok, stop
            padded = [r + [pad_id] * (max_n - len(r)) for r in rows]
            add = torch.tensor(padded, device=device, dtype=torch.long)
            return torch.cat([tok, add], dim=1), stop

        h_left_cache: torch.Tensor | None = _encode_h(tokens) if tokens.size(1) > 0 else None
        z_carry: torch.Tensor | None = None
        sc_carry: torch.Tensor | None = None

        while tokens.size(1) < seqlen:
            L = int(tokens.size(1))
            if bool((tokens == eos_id).any(dim=1).all()):
                break
            r = L % step
            g = (L // step) * step

            if L > 0 and r > 0:
                sc_win = torch.zeros(num_samples, w_sz, self.n_embd, device=device, dtype=dtype)
                z_win = torch.randn(
                    num_samples, w_sz, self.n_embd, device=device, dtype=dtype,
                ) * float(self.config.noise_sigma)
                for hop in range(self.time_step):
                    k0 = step * (self.time_step - hop - 1)
                    u = g - k0
                    if u < 0:
                        u = 0
                    t_w, m_w, in_win = _fields_for_u(u, tokens, target_len=seqlen)
                    known = torch.zeros_like(in_win)
                    for k in range(w_sz):
                        if k0 <= k < k0 + r:
                            j = u + k
                            if 0 <= j < L:
                                known[:, k] = True
                                m_w[:, k] = _MODE_NONE
                                t_w[:, k] = 1.0
                    m_w = torch.where(in_win & (~known), m_w, torch.zeros_like(m_w))
                    h_all = h_left_cache if h_left_cache is not None else _encode_h(tokens)
                    h_left = h_all[:, :u] if u > 0 else h_all[:, :0]
                    if L > u:
                        sl_n = min(w_sz, L - u)
                        clean = h_all[:, u : u + sl_n]
                        nse = torch.randn_like(clean) * float(self.config.noise_sigma)
                        mixed = interpolate(clean, t_w[:, :sl_n], nse)
                        z_win[:, :sl_n] = torch.where(known[:, :sl_n].unsqueeze(-1), clean, mixed)
                    pos_left = torch.arange(h_left.size(1), device=device)
                    pos_win = torch.arange(u, u + w_sz, device=device).clamp(max=self.max_seq_len - 1)
                    sc_in = sc_win.clone()
                    sc_in[:, w_sz - step :] = 0
                    x_hat = _one_g(
                        h_left, z_win, t_w, m_w, sc_in, pos_left, pos_win, drop_left=False,
                    )
                    x_right = x_hat[:, h_left.size(1) :]
                    if w_ctx != 1.0 and h_left.size(1) > 0:
                        x_u = _one_g(
                            h_left, z_win, t_w, m_w, sc_in, pos_left, pos_win, drop_left=True,
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
                    if L > u:
                        sl_n = min(w_sz, L - u)
                        z_win[:, :sl_n] = torch.where(
                            known[:, :sl_n].unsqueeze(-1),
                            h_all[:, u : u + sl_n],
                            z_win[:, :sl_n],
                        )
                    sc_win = x_right.detach()
                    sc_win[:, w_sz - step :] = 0
                    if hop == self.time_step - 1:
                        logits = self._exit_logits(x_hat)[:, h_left.size(1) :]
                        sampled = self._sample_tokens(
                            logits, temperature=temperature, top_k=top_k,
                        )
                        tokens, _ = _append_decoded(tokens, sampled, m_w == _MODE_DECODE)
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
                h_left, z_win, t_w, m_w, sc_win, pos_left, pos_win, drop_left=False,
            )
            x_right = x_hat[:, h_left.size(1) :]
            if w_ctx != 1.0 and h_left.size(1) > 0:
                x_u = _one_g(
                    h_left, z_win, t_w, m_w, sc_win, pos_left, pos_win, drop_left=True,
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
            tokens, stop = _append_decoded(tokens, sampled, dec)
            if int(tokens.size(1)) == new_len:
                if (not bool(in_win.any())) or u + step >= seqlen:
                    break
            if commit_x0:
                h_left_cache = torch.cat([h_left, x_right[:, :step]], dim=1)
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
                [x_right[:, step:], torch.zeros_like(x_right[:, :step])], dim=1,
            )
            if stop:
                break

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
