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

    末流关闭 SDE churn。默认 ``commit_x0hat=true``。
    """
    del bos_token_id
    cfg = sampling_cfg or {}
    seqlen = int(seqlen or backbone.max_seq_len)
    w = int(backbone.block_size)
    if seqlen % w != 0:
        raise ValueError(f"seqlen {seqlen} 须能被 block_size {w} 整除")
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
    w_sc_val = float(cfg.get("w_sc", 1.0))
    w_ctx = float(cfg.get("w_ctx", 1.0))

    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype
    levels = backbone.levels.to(device=device, dtype=dtype)
    t_steps = int(backbone.time_step)
    n_blocks = seqlen // w
    pad_id = int(backbone.token_layout.pad_token_id)

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
        z_pref, _, _ = backbone.bundle.encode(prefix[:, : n_full * w], sample=False)
        clean = backbone.to_d(z_pref)
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

    for b_idx in range(start_block, n_blocks):
        z_block = torch.randn(
            num_samples, w, backbone.n_embd, device=device, dtype=dtype,
        ) * float(backbone.denoiser_noise_scale)

        known_n = known_rem if b_idx == start_block else 0
        x_known: torch.Tensor | None = None
        if known_n > 0 and prefix is not None:
            # 当前块余数：编码已给出的 token，钉为已知（t=1）。
            raw = prefix[:, n_full * w : prefix_len]
            pad_blk = raw.new_full((num_samples, w), pad_id)
            pad_blk[:, :known_n] = raw
            z_k, _, _ = backbone.bundle.encode(pad_blk, sample=False)
            x_known = backbone.to_d(z_k)[:, :known_n]
            z_block = z_block.clone()
            z_block[:, :known_n] = x_known

        left = clean
        if w_ctx == 0.0:
            left = left[:, :0]

        def _predict_v(z_cur: torch.Tensor, t_scalar: float) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal nfe
            nfe += 1
            x_hat = backbone.forward_g(
                left, z_cur, t_scalar, MODE_DENOISE, w_sc,
                known_right=known_n,
            )
            t_tok = z_cur.new_full((z_cur.size(0), z_cur.size(1)), float(t_scalar))
            if known_n > 0:
                t_tok[:, :known_n] = 1.0
            v = x_to_v(x_hat, z_cur, t_tok, backbone.vel_eps)
            return v, x_hat

        x_hat = z_block
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
                v, x_hat = _predict_v(z_back, t_back)
                z_block = ode_update(z_back, v, t_back, t_next)
            elif method in ("sde", "ode"):
                v, x_hat = _predict_v(z_block, t_curr)
                z_block = ode_update(z_block, v, t_curr, t_next)
            else:
                raise ValueError(f"未知 sampling_method={method!r}")
            if x_known is not None:
                z_block[:, :known_n] = x_known

        nfe += 1
        t_dec = float(levels[-1].item())
        x_hat = backbone.forward_g(
            left, z_block, t_dec, MODE_DECODE, w_sc,
            known_right=known_n,
        )
        dec_in = x_hat
        if clean.size(1) > 0 and w_ctx != 0.0:
            dec_in = torch.cat([clean, x_hat], dim=1)
        logits = backbone.exit_logits(dec_in)
        block_logits = logits[:, -w:]
        if temperature <= 0:
            blk_tok = block_logits.argmax(dim=-1)
        else:
            blk_tok = sample_from_logits(
                block_logits, temperature=temperature, top_k=top_k,
            )
        if known_n > 0 and prefix is not None:
            blk_tok = blk_tok.clone()
            blk_tok[:, :known_n] = prefix[:, n_full * w : prefix_len]
        start = b_idx * w
        tokens[:, start : start + w] = blk_tok

        if commit_x0hat:
            committed = x_hat
        else:
            z_c, _, _ = backbone.bundle.encode(blk_tok, sample=False)
            committed = backbone.to_d(z_c)
        if x_known is not None:
            committed = committed.clone()
            committed[:, :known_n] = x_known
        clean = torch.cat([clean, committed], dim=1)
        known_rem = 0

    if prefix_len > 0 and prefix is not None:
        tokens = tokens.clone()
        tokens[:, :prefix_len] = prefix
    return tokens, nfe
