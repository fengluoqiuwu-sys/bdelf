"""AR3: AR2.5 + long-range attention alignment. Spec: temp/ar3.md."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.ar2_5.model import _AR25Backbone
from models.ar3.config import FL_AR3Config
from models.model import FL_PreTrainedModel, ensure_token_layout, split_model_cfg
from models.tokens import apply_token_layout_to_config, token_layout_from_cfg


class _AR3Backbone(_AR25Backbone):
    def __init__(
        self,
        *,
        align_topk: int = 8,
        align_power: float = 2.0,
        align_mass_power: float = 1.0,
        align_logit_coef: float = 1.0,
        align_loss_coef: float = 0.1,
        align_warmup_ratio: float = 0.1,
        align_query_prob: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.collect_qkv = True
        self.align_topk = align_topk
        self.align_power = align_power
        self.align_mass_power = align_mass_power
        self.align_logit_coef = align_logit_coef
        self.align_loss_coef = align_loss_coef
        self.align_warmup_ratio = align_warmup_ratio
        self.align_query_prob = align_query_prob
        self._align_step = 0
        self._align_total = 0
        self.last_align_metrics: dict[str, float] = {}

    def set_align_progress(self, step: int, total: int) -> None:
        self._align_step = int(step)
        self._align_total = int(total)

    def _align_lambda(self) -> float:
        warm = self.align_warmup_ratio
        total = self._align_total
        step = self._align_step
        if total <= 0:
            return float(self.align_loss_coef)
        if step < warm * total:
            return 0.0
        t0 = warm * total
        return float(self.align_loss_coef) * (step - t0) / max(total - t0, 1)

    def _ext_meta(self, n: int, device: torch.device):
        """Per extended index: block id, is_s, is_t."""
        bs = self.block_size
        n_blocks = n // bs
        total = self._ext_len(n)
        period = bs + 1
        last_start = (n_blocks - 1) * period
        idx = torch.arange(total, device=device)
        in_last = idx >= last_start
        blk = torch.where(in_last, n_blocks - 1, idx // period)
        off = torch.where(in_last, idx - last_start, idx % period)
        is_s = (~in_last) & (off == bs)
        is_t = ~is_s
        return {
            "blk": blk,
            "is_s": is_s,
            "is_t": is_t,
            "n_blocks": n_blocks,
            "total": total,
        }

    def _compute_align(
        self,
        layer_qkv: list,
        meta: dict,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-layer batched oracle align (one GEMM for all queries)."""
        device = layer_qkv[0][0].device
        bs = self.block_size
        w = self.t_window
        n_blocks = meta["n_blocks"]
        blk = meta["blk"]
        is_t = meta["is_t"]
        head_dim = self.h[0].attn.head_dim
        k_align = self.align_topk
        alpha = self.align_power
        gamma = self.align_mass_power
        eps = 1e-8
        scale = head_dim ** -0.5

        t_idx = torch.nonzero(is_t & (blk >= w + 1), as_tuple=False).squeeze(-1)
        if t_idx.numel() == 0:
            z = torch.zeros((), device=device)
            return z, z, z

        keep = torch.rand(t_idx.numel(), device=device) < self.align_query_prob
        q_idx = t_idx[keep]
        if q_idx.numel() == 0:
            z = torch.zeros((), device=device)
            return z, z, z

        q_blk = blk[q_idx]  # (Nq,)
        r_max_allowed = q_blk - (w + 1)  # inclusive max remote block per query
        global_r_max = int(r_max_allowed.max().item())
        if global_r_max < 0:
            z = torch.zeros((), device=device)
            return z, z, z

        r_idx = torch.nonzero(is_t & (blk <= global_r_max), as_tuple=False).squeeze(-1)
        if r_idx.numel() == 0:
            z = torch.zeros((), device=device)
            return z, z, z

        r_blk = blk[r_idx]  # (Nr,)
        # s_j at j*(B+1)+B for non-final blocks
        n_s = n_blocks - 1
        if n_s <= 0:
            z = torch.zeros((), device=device)
            return z, z, z
        s_pos = torch.arange(n_s, device=device) * (bs + 1) + bs  # (n_s,)

        nq = q_idx.numel()
        nr = r_idx.numel()
        kk = min(k_align, nr)

        # Invalid remote for a query: r_blk > q_blk - W - 1
        # (Nq, Nr)
        invalid = r_blk.unsqueeze(0) > r_max_allowed.unsqueeze(1)

        out_acc = torch.zeros((), device=device, dtype=torch.float32)
        logit_acc = torch.zeros((), device=device, dtype=torch.float32)
        mass_acc = torch.zeros((), device=device, dtype=torch.float32)

        for q, k, v in layer_qkv:
            # (B, H, T, D)
            bt, n_head, _, d_h = q.shape
            qv = q[:, :, q_idx].detach()  # (B, H, Nq, D)
            kr = k[:, :, r_idx]
            vr = v[:, :, r_idx]
            kr_det = kr.detach()
            vr_det = vr.detach()

            scores = torch.matmul(qv, kr_det.transpose(-1, -2)) * scale
            scores = scores.masked_fill(invalid.view(1, 1, nq, nr), float("-inf"))

            topv, topi = torch.topk(scores, kk, dim=-1)  # (B, H, Nq, K)
            # Queries with fewer than kk valid remotes get -inf tops — drop them.
            valid_topk = torch.isfinite(topv)
            topv = topv.masked_fill(~valid_topk, 0.0)

            a_hat = F.softplus(topv) * valid_topk
            rho = a_hat.pow(alpha)
            rho_sum = rho.sum(dim=-1, keepdim=True).clamp_min(eps)
            rho = rho / rho_sum

            b_ix = torch.arange(bt, device=device)[:, None, None, None]
            h_ix = torch.arange(n_head, device=device)[None, :, None, None]
            k_sel = kr_det[b_ix, h_ix, topi]  # (B, H, Nq, K, D)
            v_sel = vr_det[b_ix, h_ix, topi]
            sel_blk = r_blk[topi]  # (B, H, Nq, K)
            # Invalidate block id for padded topk slots
            sel_blk = torch.where(valid_topk, sel_blk, torch.full_like(sel_blk, -1))

            # Anchor K/V for all non-final blocks: (B, H, n_s, D) — grads flow here
            k_s_all = k[:, :, s_pos, :]
            v_s_all = v[:, :, s_pos, :]

            # Aggregate over remote blocks j that appear in top-k (≤K distinct / query).
            # Scatter mass and weighted KV into block bins via one-hot over n_s.
            # one_hot: (B, H, Nq, K, n_s)
            j_ok = (sel_blk >= 0) & (sel_blk < n_s)
            sel_safe = sel_blk.clamp(min=0, max=max(n_s - 1, 0))
            oh = F.one_hot(sel_safe, n_s).to(dtype=a_hat.dtype)  # (..., n_s)
            oh = oh * j_ok.unsqueeze(-1) * valid_topk.unsqueeze(-1)

            # M[j] = sum_{i in T ∩ X_j} a_hat_i  → (B, H, Nq, n_s)
            m_j = (a_hat.unsqueeze(-1) * oh).sum(dim=-2)
            # Block-local rho renormalization: ρ_i / sum_{u in T∩X_j} ρ_u
            rho_j = (rho.unsqueeze(-1) * oh).sum(dim=-2).clamp_min(eps)  # (B,H,Nq,n_s)
            # Per-slot weight into its block: (B,H,Nq,K,n_s)
            w_slot = rho.unsqueeze(-1) * oh / rho_j.unsqueeze(-2)

            # v*/k* via einsum (avoid materializing K×n_s×D broadcast): (B,H,Nq,n_s,D)
            v_star = torch.einsum("bhqks,bhqkd->bhqsd", w_slot, v_sel)
            k_star = torch.einsum("bhqks,bhqkd->bhqsd", w_slot, k_sel)

            valid_j = m_j > 0  # (B, H, Nq, n_s)
            m_g = m_j.detach().pow(gamma) * valid_j

            # L_out: ||v_s - v*||^2
            l_out = (v_s_all.unsqueeze(2) - v_star).pow(2).sum(dim=-1)  # (B,H,Nq,n_s)

            # L_logit: spec uses unscaled (q·k_s - q·k*)^2 / d_h.
            # With already-scaled scores this equals (score_s - score_star)^2.
            score_s = (qv.unsqueeze(3) * k_s_all.unsqueeze(2)).sum(dim=-1) * scale
            score_star = (qv.unsqueeze(3) * k_star).sum(dim=-1) * scale
            l_logit = (score_s - score_star).pow(2)

            out_acc = out_acc + (m_g * l_out).sum()
            logit_acc = logit_acc + (m_g * l_logit).sum()
            mass_acc = mass_acc + m_g.sum()

        if mass_acc <= 0:
            z = torch.zeros((), device=device)
            return z, z, z
        return out_acc / (mass_acc + eps), logit_acc / (mass_acc + eps), mass_acc

    def forward(self, idx, targets=None):
        _, loss_lm = super().forward(idx, targets)
        lam = self._align_lambda()
        self.last_align_metrics = {
            "lambda": lam,
            "loss_out": 0.0,
            "loss_logit": 0.0,
        }
        # Align only in training with fresh QKV from this forward (eval does not
        # collect_qkv; reusing a stale cache would corrupt loss or crash).
        if (
            not self.training
            or lam <= 0.0
            or not getattr(self, "_last_layer_qkv", None)
        ):
            return torch.empty(0), loss_lm

        n = int(self._last_n)
        meta = self._ext_meta(n, idx.device)
        loss_out, loss_logit, _ = self._compute_align(self._last_layer_qkv, meta, n)
        loss_align = loss_out + self.align_logit_coef * loss_logit
        self.last_align_metrics = {
            "lambda": lam,
            "loss_out": float(loss_out.detach()) if loss_out.ndim == 0 else 0.0,
            "loss_logit": float(loss_logit.detach()) if loss_logit.ndim == 0 else 0.0,
        }
        return torch.empty(0), loss_lm + lam * loss_align


class FL_AR3Model(FL_PreTrainedModel):
    config_class = FL_AR3Config

    def __init__(self, config: FL_AR3Config) -> None:
        super().__init__(config)
        self.backbone = _AR3Backbone(**config.backbone_kwargs())


def build_model_from_config(config: FL_AR3Config) -> FL_AR3Model:
    ensure_token_layout(config)
    return FL_AR3Model(config)


def build_model(cfg: dict) -> FL_AR3Model:
    data, sampling = split_model_cfg(cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_AR3Config(**data)
    config.tokenizer = cfg.get("tokenizer", "gpt2")
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
