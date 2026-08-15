"""I-1 探针用：从 ELF ``generate`` / decode 头复制出来的只读包装，不改 ``models/elf``。

采样循环与官方 ELF 一致，额外返回终态 ``z`` 与 unembed 前 512-d hidden。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from models.elf.ace import (
    ace_step_active,
    apply_ace_steer,
    parse_ace_step_range,
    resolve_ace_steering,
)


@torch.no_grad()
def elf_generate_latent(
    backbone: torch.nn.Module,
    *,
    num_samples: int,
    seqlen: int,
    sampling_cfg: dict[str, Any],
    skip_decode: bool = False,
    bgee: bool = False,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """复制 ``_ELFBackbone.generate``，始终多返回终态 ``z``。

    skip_decode：只跑去噪（拆墙钟）。bgee：每步额外跑一次 full native g。
    """
    cfg = sampling_cfg or {}
    if seqlen > backbone.max_seq_len:
        raise ValueError(
            f"seqlen {seqlen} exceeds max_seq_len {backbone.max_seq_len}"
        )
    method = str(cfg.get("sampling_method", "sde")).lower()
    num_sampling_steps = int(cfg.get("num_sampling_steps", 32))
    sde_gamma = float(cfg.get("sde_gamma", 1.5))
    temperature = float(cfg.get("temperature", 0.0))
    top_k = cfg.get("top_k")
    if top_k is not None:
        top_k = int(top_k)
    infer_schedule = cfg.get("time_schedule")
    if infer_schedule is not None:
        backbone._infer_time_schedule = infer_schedule
    sc_cfg_w = float(cfg.get("self_cond_cfg_scale", 1.0))

    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype
    ace_lam, ace_d = resolve_ace_steering(
        cfg,
        device=device,
        dtype=dtype,
        expected_dim=backbone.text_encoder_dim,
        backbone=backbone,
    )
    ace_step_lo, ace_step_hi = parse_ace_step_range(cfg)
    t_steps = backbone._get_sampling_steps(num_sampling_steps, device, dtype)
    z = (
        torch.randn(
            num_samples, seqlen, backbone.text_encoder_dim,
            device=device, dtype=dtype,
        )
        * backbone.denoiser_noise_scale
    )
    if backbone.num_self_cond_cfg_tokens > 0 or sc_cfg_w != 1.0:
        self_cond_cfg_scale = torch.full(
            (num_samples,), sc_cfg_w, dtype=dtype, device=device,
        )
    else:
        self_cond_cfg_scale = None
    x_pred: torch.Tensor | None = None
    nfe = 0
    for i in range(t_steps.numel() - 2):
        t = float(t_steps[i].item())
        t_next = float(t_steps[i + 1].item())
        if method == "sde":
            z, x_pred = backbone._sde_step(
                z, t, t_next, x_pred, sde_gamma,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )
        elif method == "ode":
            z, x_pred = backbone._ode_step(
                z, t, t_next, x_pred,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )
        else:
            raise ValueError(f"unknown sampling_method: {method}")
        if (
            ace_d is not None
            and x_pred is not None
            and ace_step_active(i, step_lo=ace_step_lo, step_hi=ace_step_hi)
        ):
            x_pred = apply_ace_steer(x_pred, lam=ace_lam, direction=ace_d)
        if bgee:
            elf_decode_probe(backbone, z, self_cond_cfg_scale)
        nfe += 1

    t = float(t_steps[-2].item())
    t_next = float(t_steps[-1].item())
    z, x_pred = backbone._ode_step(
        z, t, t_next, x_pred,
        self_cond_cfg_scale=self_cond_cfg_scale,
    )
    nfe += 1
    if not torch.isfinite(z).all():
        raise RuntimeError("ELF sampling produced non-finite latents")
    if bgee:
        elf_decode_probe(backbone, z, self_cond_cfg_scale)
    if skip_decode:
        dummy = torch.zeros(
            num_samples, seqlen, dtype=torch.long, device=z.device,
        )
        return dummy, nfe, z

    tokens = backbone._decode_tokens(
        z,
        temperature=temperature,
        top_k=top_k,
        self_cond_cfg_scale=self_cond_cfg_scale,
    )
    nfe += 1
    tokens = backbone._mask_after_eos(
        tokens,
        eos_token_id=backbone.token_layout.eos_token_id,
        pad_token_id=backbone.token_layout.pad_token_id,
    )
    return tokens, nfe, z


@torch.no_grad()
def elf_decode_probe(
    backbone: torch.nn.Module,
    z: torch.Tensor,
    self_cond_cfg_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native logits 与 factored-unembed 前 hidden；hook 捕获 ``x_h``，不改 ELF。"""
    bsz = z.shape[0]
    t = torch.ones(bsz, device=z.device, dtype=z.dtype)
    if backbone.self_cond_prob > 0:
        model_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
    else:
        model_in = z
    captured: dict[str, torch.Tensor] = {}

    def _hook(_mod, inp, _out):
        captured["x_h"] = inp[0]

    handle = backbone.final_layer.register_forward_hook(_hook)
    try:
        _x_pred, logits = backbone.net_forward(
            model_in, t, decoder_step_active=True, deterministic=True,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
    finally:
        handle.remove()
    assert logits is not None
    xf = captured["x_h"].to(dtype=backbone.proj_kernel.dtype)
    hidden = F.gelu(
        xf @ backbone.proj_kernel + backbone.proj_bias,
        approximate="tanh",
    )
    return logits, hidden
