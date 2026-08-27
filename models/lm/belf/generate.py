"""BELF 块采样：流 Euler / SDE churn 与 ``block_generate``。"""

from __future__ import annotations

from typing import Any

import torch

from models.lm.belf_relf_core import x_to_v
from models.model import sample_from_logits

# 与 AdaLNZeroStack 逐列模式一致。
MODE_NONE = 0
MODE_DENOISE = 1
MODE_DECODE = 2


def ode_update(z: torch.Tensor, v: torch.Tensor, t: float, t_next: float) -> torch.Tensor:
    """流 Euler：``z ← z + (t_next - t) v``（t 增大则更干净）。"""
    return z + (float(t_next) - float(t)) * v


def sde_churn(
    z: torch.Tensor,
    t: float,
    t_next: float,
    gamma: float,
    noise_scale: float,
) -> tuple[torch.Tensor, float]:
    """ELF 风格回噪声：返回 ``(z_back, t_back)``，调用方再前向并 Euler 到 ``t_next``。"""
    h = float(t_next) - float(t)
    alpha = max(0.0, min(1.0, 1.0 - float(gamma) * h))
    t_back = alpha * float(t)
    eps = torch.randn_like(z) * float(noise_scale)
    z_back = alpha * z + (1.0 - alpha) * eps
    return z_back, t_back


@torch.no_grad()
def block_generate(
    backbone: Any,
    num_samples: int = 1,
    seqlen: int | None = None,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    bos_token_id: int | None = None,
    prefix_tokens: torch.Tensor | None = None,
    sampling_cfg: dict | None = None,
) -> tuple[torch.Tensor, int]:
    """逐块：``T-1`` 次流 Euler + 1 次 decode（decode 不做 Euler）。

    末流关闭 SDE churn。默认 ``commit_x0hat=true``。末块可短；EOS 后该样本停。
    """
    _ = bos_token_id
    cfg = sampling_cfg or {}
    seqlen = int(seqlen or backbone.max_seq_len)
    w = int(backbone.block_size)
    if seqlen < 1:
        raise ValueError(f"seqlen 须为正，收到 {seqlen}")
    if seqlen > int(backbone.max_seq_len):
        raise ValueError(
            f"seqlen {seqlen} 超过 max_seq_len={backbone.max_seq_len}"
        )

    method = str(cfg.get("sampling_method", "sde")).lower()
    sde_gamma = float(cfg.get("sde_gamma", 1.5))
    temperature = float(cfg.get("temperature", temperature))
    top_k = cfg.get("top_k", top_k)
    if top_k is not None:
        top_k = int(top_k)
    commit_x0hat = bool(cfg.get("commit_x0hat", True))
    w_sc_val = float(cfg.get("w_sc", cfg.get("self_cond_cfg_scale", 3.0)))
    w_ctx = float(cfg.get("w_ctx", cfg.get("ctx_cfg_scale", 1.0)))
    x0_src = str(getattr(backbone, "x0_source", "z")).strip().lower()

    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype
    levels = backbone.levels.to(device=device, dtype=dtype)
    t_steps = int(backbone.time_step)
    n_blocks = (seqlen + w - 1) // w
    pad_id = int(backbone.token_layout.pad_token_id)
    eos_id = int(backbone.token_layout.eos_token_id)

    prefix_len = 0
    prefix: torch.Tensor | None = None
    if prefix_tokens is not None:
        prefix = prefix_tokens.to(device=device, dtype=torch.long)
        if prefix.size(0) != num_samples:
            raise ValueError("prefix_tokens batch 须与 num_samples 一致")
        prefix_len = int(prefix.size(1))
        if prefix_len >= seqlen:
            raise ValueError("prefix 长度须 < seqlen")

    n_full = prefix_len // w
    rem = prefix_len % w
    tokens = torch.full(
        (num_samples, seqlen), pad_id, device=device, dtype=torch.long,
    )
    if prefix is not None and prefix_len > 0:
        tokens[:, :prefix_len] = prefix

    if n_full > 0:
        z_pref, mu_pref, _ = backbone.bundle.encode(
            prefix[:, : n_full * w], sample=False,
        )
        src_pref = mu_pref if x0_src == "mu" else z_pref
        clean = backbone.to_d(src_pref)
    else:
        clean = torch.zeros(
            num_samples, 0, backbone.n_embd, device=device, dtype=dtype,
        )

    w_sc = None
    if backbone.sc_cfg:
        w_sc = torch.full((num_samples,), w_sc_val, device=device, dtype=dtype)

    nfe = 0
    start_block = n_full
    known_rem = rem if str(backbone.cond_mode) == "clean" else 0
    alive = torch.ones(num_samples, dtype=torch.bool, device=device)
    if prefix_len > 0:
        alive = alive & ~(tokens[:, :prefix_len] == eos_id).any(dim=1)

    for b_idx in range(start_block, n_blocks):
        if not bool(alive.any().item()):
            break
        w_cur = min(w, seqlen - b_idx * w)
        z_block = torch.randn(
            num_samples, w_cur, backbone.n_embd, device=device, dtype=dtype,
        ) * float(backbone.denoiser_noise_scale)

        known_n = known_rem if b_idx == start_block else 0
        if known_n > w_cur:
            known_n = w_cur
        x_known: torch.Tensor | None = None
        if known_n > 0 and prefix is not None:
            z_k, mu_k, _ = backbone.bundle.encode(
                prefix[:, :prefix_len], sample=False,
            )
            src_k = mu_k if x0_src == "mu" else z_k
            x_known = backbone.to_d(src_k)[:, n_full * w : prefix_len]
            z_block = z_block.clone()
            z_block[:, :known_n] = x_known

        left = clean
        drop_ctx = abs(float(w_ctx) - 1.0) > 1e-8 and left.size(1) > 0

        def _predict_v(
            z_cur: torch.Tensor,
            t_scalar: float,
            sc_right: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal nfe
            nfe += 1
            x_hat = backbone.forward_g(
                left, z_cur, t_scalar, MODE_DENOISE, w_sc,
                known_right=known_n,
                sc_right=sc_right,
            )
            t_tok = z_cur.new_full((z_cur.size(0), z_cur.size(1)), float(t_scalar))
            if known_n > 0:
                t_tok[:, :known_n] = 1.0
            v_c = x_to_v(x_hat, z_cur, t_tok, backbone.vel_eps)
            if drop_ctx:
                nfe += 1
                empty = left[:, :0]
                x_u = backbone.forward_g(
                    empty, z_cur, t_scalar, MODE_DENOISE, w_sc,
                    known_right=known_n,
                    sc_right=sc_right,
                )
                v_u = x_to_v(x_u, z_cur, t_tok, backbone.vel_eps)
                v = v_u + float(w_ctx) * (v_c - v_u)
            else:
                v = v_c
            return v, x_hat

        x_hat = z_block
        sc_right = torch.zeros_like(z_block)
        for hop in range(t_steps - 1):
            t_curr = float(levels[hop].item())
            t_next = float(levels[hop + 1].item())
            last_flow = hop == t_steps - 2
            if method == "sde" and not last_flow:
                z_back, t_back = sde_churn(
                    z_block, t_curr, t_next, sde_gamma,
                    backbone.denoiser_noise_scale,
                )
                if x_known is not None:
                    z_back[:, :known_n] = x_known
                v, x_hat = _predict_v(z_back, t_back, sc_right)
                z_block = ode_update(z_back, v, t_back, t_next)
            elif method in ("sde", "ode"):
                v, x_hat = _predict_v(z_block, t_curr, sc_right)
                z_block = ode_update(z_block, v, t_curr, t_next)
            else:
                raise ValueError(f"未知 sampling_method={method!r}")
            if x_known is not None:
                z_block[:, :known_n] = x_known
            sc_right = x_hat.detach()
            if known_n > 0:
                sc_right = sc_right.clone()
                sc_right[:, :known_n] = 0

        nfe += 1
        t_dec = float(levels[-1].item())
        w_sc_dec = torch.zeros_like(w_sc) if w_sc is not None else None
        x_hat = backbone.forward_g(
            left, z_block, t_dec, MODE_DECODE, w_sc_dec,
            known_right=known_n,
            sc_right=sc_right,
        )
        dec_in = x_hat
        if clean.size(1) > 0:
            dec_in = torch.cat([clean, x_hat], dim=1)
        logits = backbone.exit_logits(dec_in)
        block_logits = logits[:, -w_cur:]
        if temperature <= 0:
            blk_tok = block_logits.argmax(dim=-1)
        else:
            blk_tok = sample_from_logits(
                block_logits, temperature=temperature, top_k=top_k,
            )
        if known_n > 0 and prefix is not None:
            blk_tok = blk_tok.clone()
            blk_tok[:, :known_n] = prefix[:, n_full * w : prefix_len][:, :known_n]
        start = b_idx * w
        write = blk_tok.clone()
        write[~alive] = pad_id
        tokens[:, start : start + w_cur] = write

        if commit_x0hat:
            committed = x_hat
        else:
            z_c, mu_c, _ = backbone.bundle.encode(
                tokens[:, : start + w_cur], sample=x0_src != "mu",
            )
            src_c = mu_c if x0_src == "mu" else z_c
            committed = backbone.to_d(src_c)[:, start : start + w_cur]
        if x_known is not None:
            committed = committed.clone()
            committed[:, :known_n] = x_known
        clean = torch.cat([clean, committed], dim=1)
        known_rem = 0
        alive = alive & ~(tokens[:, : start + w_cur] == eos_id).any(dim=1)

    if prefix_len > 0 and prefix is not None:
        tokens = tokens.clone()
        tokens[:, :prefix_len] = prefix
    return tokens, nfe
