"""Muon optimizer and hybrid Muon + AdamW builder for transformer training."""

from __future__ import annotations

import math
import re
from typing import Any

import torch
from torch import nn

from train.train import FL_TrainConfig

_HIDDEN_LINEAR_WEIGHT_RE = re.compile(
    r"\.(attn|mlp)\.(c_attn|c_proj|c_fc)\.weight$"
)


def _is_muon_weight(name: str, param: nn.Parameter) -> bool:
    if param.dim() != 2:
        return False
    if "wte" in name or "lm_head" in name:
        return False
    return _HIDDEN_LINEAR_WEIGHT_RE.search(name) is not None


def split_muon_adamw_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Return (muon_weights, adamw_decay, adamw_nodecay) for a model."""
    muon_params: list[nn.Parameter] = []
    decay_params: list[nn.Parameter] = []
    nodecay_params: list[nn.Parameter] = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if _is_muon_weight(name, param):
            muon_params.append(param)
        elif param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    return muon_params, decay_params, nodecay_params


def zeropower_via_newtonschulz5(
    grad: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Orthogonalize a 2D update via Newton-Schulz iteration (bf16-safe)."""
    if grad.ndim != 2:
        raise ValueError(f"Muon expects 2D gradients, got shape {tuple(grad.shape)}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    work = grad.bfloat16()
    work = work / (work.norm() + eps)
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        work = work.T
    for _ in range(steps):
        gram = work @ work.T
        gram_poly = b * gram + c * (gram @ gram)
        work = a * work + gram_poly @ work
    if transposed:
        work = work.T
    return work.to(dtype=grad.dtype)


def _muon_lr_scale(param: nn.Parameter) -> float:
    rows, cols = param.shape
    return max(1.0, rows / cols) ** 0.5


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-Schulz (2D weight matrices only)."""

    def __init__(
        self,
        params,
        *,
        lr: float = 0.003,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad)
                    state["momentum_buffer"] = buf
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                orth = zeropower_via_newtonschulz5(update, steps=ns_steps, eps=eps)
                param.add_(orth, alpha=-lr * _muon_lr_scale(param))

        return loss


class FL_CombinedOptimizer:
    """Muon for hidden Linear weights; AdamW for everything else."""

    def __init__(
        self,
        muon: Muon,
        adamw: torch.optim.AdamW,
    ) -> None:
        self.muon = muon
        self.adamw = adamw
        for group in self.muon.param_groups:
            group["optim_kind"] = "muon"
        for group in self.adamw.param_groups:
            group["optim_kind"] = "adamw"

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "muon_hybrid",
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("kind") != "muon_hybrid":
            raise ValueError(
                "Expected hybrid Muon checkpoint optimizer state (kind='muon_hybrid')"
            )
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def build_optimizer(
    model: nn.Module,
    cfg: FL_TrainConfig,
) -> torch.optim.AdamW | FL_CombinedOptimizer:
    if not cfg.use_muon:
        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": cfg.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
        )

    muon_params, decay_params, nodecay_params = split_muon_adamw_params(model)
    if not muon_params:
        raise ValueError(f"{cfg.name}: use_muon enabled but no Muon-eligible weights found")

    muon = Muon(
        muon_params,
        lr=cfg.muon_learning_rate,
        momentum=cfg.muon_momentum,
        ns_steps=cfg.muon_ns_steps,
    )
    adamw = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )
    return FL_CombinedOptimizer(muon, adamw)


def schedule_optimizer_lrs(
    optimizer: torch.optim.AdamW | FL_CombinedOptimizer,
    *,
    adam_lr: float,
    muon_lr: float,
) -> None:
    for group in optimizer.param_groups:
        if group.get("optim_kind") == "muon":
            group["lr"] = muon_lr
        else:
            group["lr"] = adam_lr


def scaled_lr(step: int, cfg: FL_TrainConfig, base_lr: float) -> float:
    if step < cfg.warmup_steps:
        return base_lr * step / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return base_lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)
