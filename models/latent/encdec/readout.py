"""瓶颈读出与 KL / 采样工具。"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

SIGMA_TAG_SUFFIX = "-sigma"


def parse_kl_entropy(value: Any = None) -> bool:
    """缺键 / ``None`` / ``false`` → ``False``（与省略同指纹）。"""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and value != 0
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"kl_entropy 须为布尔，收到 {value!r}")


def drop_off_kl_entropy(mapping: dict[str, Any]) -> dict[str, Any]:
    """``kl_entropy`` 为关时从映射去掉，使 ``false`` 与缺键同哈希。"""
    out = dict(mapping)
    if "kl_entropy" in out and not parse_kl_entropy(out.get("kl_entropy")):
        del out["kl_entropy"]
    return out


def ensure_sigma_tag(tag: str, kl_entropy: bool) -> str:
    """``kl_entropy`` 开时 tag 以 ``-sigma`` 结尾。"""
    name = str(tag).strip()
    if not kl_entropy:
        return name
    if name.endswith(SIGMA_TAG_SUFFIX):
        return name
    return name + SIGMA_TAG_SUFFIX


def _masked_mean(per_tok: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return per_tok.mean()
    weight = mask.to(dtype=per_tok.dtype)
    denom = weight.sum().clamp_min(1.0)
    return (per_tok * weight).sum() / denom


def kl_gaussian(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """逐位置 KL；``mask`` 为 True 的位置才计入（忽略 pad）。"""
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return _masked_mean(kl.mean(dim=-1), mask)


def gaussian_log_q(
    logvar: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """对角高斯 E[log q] = -0.5 (1 + log(2π) + logvar)；平均方式同 ``kl_gaussian``。"""
    log_q = -0.5 * (1.0 + math.log(2.0 * math.pi) + logvar)
    return _masked_mean(log_q.mean(dim=-1), mask)


def posterior_regularizer(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    kl_entropy: bool = False,
) -> torch.Tensor:
    """关：``KL(q‖N(0,I))``；开：``KL + E[log q]``（先验仍约束 μ，另抬 σ）。"""
    kl = kl_gaussian(mu, logvar, mask=mask)
    if not kl_entropy:
        return kl
    return kl + gaussian_log_q(logvar, mask=mask)


def sample_posterior(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    sample: bool,
) -> torch.Tensor:
    """``logvar`` 须已 clamp；此处再截一次以防调用方漏夹。"""
    logvar = logvar.clamp(-30.0, 20.0)
    if sample:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    return mu


class PosteriorBReadout(nn.Module):
    """VAE / T5 readout=b：μ 与 logvar 均为 E→B。"""

    def __init__(self, n_embd: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.to_mu = nn.Linear(n_embd, latent_dim, bias=True)
        self.to_logvar = nn.Linear(n_embd, latent_dim, bias=True)
        self.enc_latent_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)

    def forward(
        self,
        h: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.enc_latent_norm(self.to_mu(h))
        # 与 cola_vae / 规格一致：采样与 KL 共用 clamp 后的 logσ²。
        logvar = self.to_logvar(h).clamp(-30.0, 20.0)
        z = sample_posterior(mu, logvar, sample=sample)
        return z, mu, logvar


class PosteriorEReadout(nn.Module):
    """T5 readout=e：瓶颈 E→B，再 B→E 得 μ/logvar。"""

    def __init__(self, n_embd: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_embd = n_embd
        self.to_bottleneck = nn.Linear(n_embd, latent_dim, bias=True)
        self.to_mu = nn.Linear(latent_dim, n_embd, bias=True)
        self.to_logvar = nn.Linear(latent_dim, n_embd, bias=True)
        self.enc_latent_norm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(
        self,
        h: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = self.to_bottleneck(h)
        mu = self.enc_latent_norm(self.to_mu(b))
        logvar = self.to_logvar(b).clamp(-30.0, 20.0)
        z = sample_posterior(mu, logvar, sample=sample)
        return z, mu, logvar
