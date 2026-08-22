"""TrACE ACE facade: shared core + TrACE-only SC/rep collection helpers.

Inference ACE uses the shared ``estimate_ace_direction`` from ``elf_core``.
Training-time slow tracking (``models.lm.trace.track``) uses the local
``collect_ace_sc_rep_pairs`` / ``direction_from_sc_rep_pairs`` /
``estimate_ace_direction_with_stats`` below.
"""

from __future__ import annotations

from typing import Any

import torch

from models.lm.elf_core.ace import *  # noqa: F403
from models.lm.elf_core.ace import (
    DEFAULT_ACE_ESTIMATE_BS,
    DEFAULT_ACE_ESTIMATE_N,
    DEFAULT_ACE_ESTIMATE_SEED,
    _log,
    _run_trajectory_collect_sc,
)


@torch.no_grad()
def collect_ace_sc_rep_pairs(
    backbone: Any,
    *,
    sampling_cfg: dict[str, Any],
    tokenizer_name: str,
    n: int,
    batch_size: int = DEFAULT_ACE_ESTIMATE_BS,
    seed: int = DEFAULT_ACE_ESTIMATE_SEED,
    seqlen: int | None = None,
    log_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """采样 ``n`` 条轨迹，返回 ``(S, r)``：``S`` 为 ``(n, e)`` 池化 SC，``r`` 为 seq-rep-4。"""
    from eval.repetition import seq_rep_4
    from tokenizer import get_tokenizer

    if n < 1:
        raise ValueError(f"ace collect n must be >= 1, got {n}")
    tok = get_tokenizer(tokenizer_name)
    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype
    if seqlen is None:
        seqlen = int(backbone.max_seq_len)
    bs = max(1, int(batch_size))

    feats: list[torch.Tensor] = []
    reps: list[float] = []
    done = 0
    if log_progress:
        _log(
            f"estimating direction n={n} bs={bs} seed={seed} "
            f"steps={sampling_cfg.get('num_sampling_steps')} "
            f"sc_cfg={sampling_cfg.get('self_cond_cfg_scale')}"
        )
    while done < n:
        cur = min(bs, n - done)
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) + done * 100003)
        z = (
            torch.randn(
                cur,
                seqlen,
                backbone.text_encoder_dim,
                device=device,
                dtype=dtype,
                generator=g,
            )
            * backbone.denoiser_noise_scale
        )
        z_final, u = _run_trajectory_collect_sc(
            backbone, z=z, sampling_cfg=sampling_cfg,
        )
        tokens = backbone._decode_tokens(
            z_final,
            temperature=0.0,
            top_k=None,
            self_cond_cfg_scale=(
                torch.full(
                    (cur,),
                    float(sampling_cfg.get("self_cond_cfg_scale", 1.0)),
                    dtype=dtype,
                    device=device,
                )
                if backbone.num_self_cond_cfg_tokens > 0
                or float(sampling_cfg.get("self_cond_cfg_scale", 1.0)) != 1.0
                else None
            ),
        )
        tokens = backbone._mask_after_eos(
            tokens,
            eos_token_id=backbone.token_layout.eos_token_id,
            pad_token_id=backbone.token_layout.pad_token_id,
        )
        for i in range(cur):
            text = tok.decode(
                tokens[i].detach().cpu().tolist(), skip_special_tokens=True,
            )
            feats.append(u[i].detach().float().cpu())
            reps.append(seq_rep_4(text))
        done += cur
        if log_progress:
            _log(f"estimate progress {done}/{n}")
    return torch.stack(feats, dim=0), torch.tensor(reps, dtype=torch.float32)


def direction_from_sc_rep_pairs(
    feats: torch.Tensor,
    reps: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """对池化 SC + seq-rep-4 做上/下三分位差分均值，单位化。"""
    if feats.size(0) < 6:
        raise ValueError(
            f"ace estimate n must be >= 6 (need tertiles), got {feats.size(0)}"
        )
    order = torch.argsort(reps)
    t = max(1, len(order) // 3)
    d = feats[order[-t:]].mean(0) - feats[order[:t]].mean(0)
    d = d / (d.norm() + 1e-8)
    rep_lo = float(reps[order[:t]].mean())
    rep_hi = float(reps[order[-t:]].mean())
    meta = {
        "rep_lo": rep_lo,
        "rep_hi": rep_hi,
        "rep_gap": rep_hi - rep_lo,
        "n": float(feats.size(0)),
        "tertile": float(t),
    }
    return d, meta


@torch.no_grad()
def estimate_ace_direction_with_stats(
    backbone: Any,
    *,
    sampling_cfg: dict[str, Any],
    tokenizer_name: str,
    n: int = DEFAULT_ACE_ESTIMATE_N,
    batch_size: int = DEFAULT_ACE_ESTIMATE_BS,
    seed: int = DEFAULT_ACE_ESTIMATE_SEED,
    seqlen: int | None = None,
    log_progress: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """差分均值方向 + tertile 统计（供 TrACE 慢跟踪判断可否更新 ``d``）。"""
    if n < 6:
        raise ValueError(f"ace estimate n must be >= 6 (need tertiles), got {n}")
    S, r = collect_ace_sc_rep_pairs(
        backbone,
        sampling_cfg=sampling_cfg,
        tokenizer_name=tokenizer_name,
        n=n,
        batch_size=batch_size,
        seed=seed,
        seqlen=seqlen,
        log_progress=log_progress,
    )
    d, meta = direction_from_sc_rep_pairs(S, r)
    if log_progress:
        _log(
            f"estimate done tertile={int(meta['tertile'])} "
            f"rep_lo={meta['rep_lo']:.4f} "
            f"rep_hi={meta['rep_hi']:.4f} "
            f"|d|={float(d.norm()):.4f}"
        )
    return d, meta
