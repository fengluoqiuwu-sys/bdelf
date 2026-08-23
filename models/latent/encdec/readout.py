"""瓶颈读出与 KL / 采样工具。"""

from __future__ import annotations

import torch
import torch.nn as nn


def kl_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()


def sample_posterior(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    sample: bool,
) -> torch.Tensor:
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
        logvar = self.to_logvar(h)
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
        logvar = self.to_logvar(b)
        z = sample_posterior(mu, logvar, sample=sample)
        return z, mu, logvar
