"""入口 latent：加载 artifacts、编码、s1 损失与 mid 解冻。"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.latent.artifact_loader import load_latent_artifact
from models.latent.encdec.readout import posterior_regularizer

_TUNE_FROZEN = "frozen"
_TUNE_FULL = "full"
_TUNE_MID = "mid"
_VALID_TUNE = (_TUNE_FROZEN, _TUNE_FULL, _TUNE_MID)


def validate_loaded_block(
    *,
    family: str,
    loaded_block: int,
    W: int | None = None,
) -> None:
    """校验加载入口块长。

    BELF：``loaded ∈ {1, W}``；RELF：``loaded == 1``。
    """
    name = str(family).strip().lower()
    loaded = int(loaded_block)
    if name == "belf":
        if W is None:
            raise ValueError("belf 须提供 W 才能校验 loaded_block")
        w_int = int(W)
        if loaded not in (1, w_int):
            raise ValueError(
                f"belf 要求 loaded_block ∈ {{1, W={w_int}}}，收到 {loaded}"
            )
        return
    if name == "relf":
        if loaded != 1:
            raise ValueError(f"relf 要求 loaded_block==1，收到 {loaded}")
        return
    raise ValueError(f"未知 family={family!r}，须为 'belf' 或 'relf'")


def validate_joint_tune(*, loaded_block: int, tune: str) -> None:
    """可训档（``full`` / ``mid``）要求入口块长为 1。

    块因果 encoder（``block_size != 1``）只允许 ``frozen``：
    否则已知余数会看到同块未来，训推条件不一致。
    """
    mode = str(tune).strip().lower()
    if mode not in (_TUNE_FULL, _TUNE_MID):
        return
    loaded = int(loaded_block)
    if loaded != 1:
        raise ValueError(
            f"latent_tune={mode} 联合训练要求入口 block_size==1，收到 {loaded}"
        )


def _try_add_to_optimizer(optimizer: Any, params: list[nn.Parameter]) -> None:
    """把尚未在 param_groups 里的新参加进去；optimizer 可为 None。"""
    if optimizer is None or not params:
        return
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return
    seen = {id(p) for g in groups for p in g["params"]}
    fresh = [p for p in params if id(p) not in seen]
    if not fresh:
        return
    extra = {k: v for k, v in groups[0].items() if k != "params"}
    optimizer.add_param_group({**extra, "params": fresh})


class LatentBundle(nn.Module):
    """加载入口 VAE，属性名为 ``latent``（不要 ``.vae.``）。

    无论是否训练都保留完整参数。``frozen`` 不算 ``L_s1``。
    """

    def __init__(
        self,
        latent_model: str | None = None,
        tag: str | None = None,
        *,
        latent: nn.Module | None = None,
        tune: str = _TUNE_FROZEN,
        latent_thaw_tokens: int | float = 5_000_000_000,
        lambda_vae: float = 1.0,
        lambda_ref: float = 1.0,
        kl_entropy: bool = False,
        device: torch.device | str | None = None,
        apply_ema: bool = True,
        checkpoint_root: str | None = None,
    ) -> None:
        super().__init__()
        mode = str(tune).strip().lower()
        if mode not in _VALID_TUNE:
            raise ValueError(f"latent_tune 须为 frozen|full|mid，收到 {tune!r}")
        if latent is None:
            if not latent_model or not tag:
                raise ValueError("须提供 latent_model+tag，或注入已构建的 latent")
            loaded = load_latent_artifact(
                latent_model,
                tag,
                device=device,
                apply_ema=apply_ema,
                checkpoint_root=checkpoint_root,
            )
            self.latent = loaded.model
            self.latent_model = loaded.latent_model
            self.tag = loaded.tag
        else:
            self.latent = latent
            self.latent_model = latent_model or ""
            self.tag = tag or ""
        self.tune = mode
        self.latent_thaw_tokens = int(latent_thaw_tokens)
        self.lambda_vae = float(lambda_vae)
        self.lambda_ref = float(lambda_ref)
        self.kl_entropy = bool(kl_entropy)
        self._thawed = mode == _TUNE_FULL
        self._require_vae_dec_if_needed()
        # 冻结副本供 ref-KL。不注册进 module 树，避免 .cuda() 时 GPU 上双份 VAE。
        # 新开进程先从刚拷入的 live 冻一份；Stage2 在 Stage1 EMA 写入 live
        # 之后会再冻（或从 Stage1 checkpoint 的 EMA 重冻），与 artifacts 目录无关。
        self._ref_cpu: list[nn.Module] = [self._freeze_copy(self.latent)]
        self.set_tune(mode)

    def _has_vae_dec(self) -> bool:
        return callable(getattr(self._module(), "decode_logits", None))

    def require_vae_dec(self, *, reason: str) -> None:
        """``exit=decoder`` 或可训档必须能 ``decode_logits``。"""
        if not self._has_vae_dec():
            raise ValueError(
                f"{reason} 须具备 VAE-dec（decode_logits），"
                f"加载 {self.latent_model}/{self.tag} 没有该接口"
            )

    def decode_logits(
        self,
        z: torch.Tensor,
        tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """走加载入口的 VAE-dec。参数是否可训由 ``latent_tune`` 决定。

        ``tokens`` 与 ``z`` 对齐时补 ``key_padding_mask``（与 ``s1_loss`` 同一条路）。
        """
        if tokens is not None and "key_padding_mask" not in kwargs:
            attn_pad = getattr(self._module(), "_attn_pad_mask", None)
            if callable(attn_pad):
                kpm = attn_pad(tokens)
                if kpm is not None:
                    kwargs["key_padding_mask"] = kpm
        return self._module().decode_logits(z, **kwargs)

    def _require_vae_dec_if_needed(self) -> None:
        """full，或 mid（含尚未解冻）须具备 VAE-dec，否则启动即拒绝。"""
        if self.tune == _TUNE_FROZEN:
            return
        if self.tune not in (_TUNE_FULL, _TUNE_MID):
            return
        self.require_vae_dec(reason=f"latent_tune={self.tune!r}")

    def _module(self) -> nn.Module:
        """真正的 VAE backbone（加载器给的是 FL_PreTrainedModel 包装）。"""
        return getattr(self.latent, "backbone", self.latent)

    def _ref_module(self) -> nn.Module:
        ref = self._ref_cpu[0]
        return getattr(ref, "backbone", ref)

    @property
    def is_trainable(self) -> bool:
        if self.tune == _TUNE_FROZEN:
            return False
        if self.tune == _TUNE_FULL:
            return True
        return self._thawed

    def set_tune(self, tune: str) -> None:
        """``frozen`` / ``full`` / ``mid``。mid 未解冻时与 frozen 相同。"""
        mode = str(tune).strip().lower()
        if mode not in _VALID_TUNE:
            raise ValueError(f"latent_tune 须为 frozen|full|mid，收到 {tune!r}")
        validate_joint_tune(loaded_block=self.block_size, tune=mode)
        self.tune = mode
        if mode == _TUNE_FULL:
            self._thawed = True
            self._set_requires_grad(True)
        elif mode == _TUNE_FROZEN:
            self._thawed = False
            self._set_requires_grad(False)
        else:
            if self._thawed:
                self._set_requires_grad(True)
            else:
                self._set_requires_grad(False)

    def _set_requires_grad(self, enabled: bool) -> None:
        for p in self.latent.parameters():
            p.requires_grad_(enabled)

    @staticmethod
    def _freeze_copy(src: nn.Module) -> nn.Module:
        ref = copy.deepcopy(src)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        ref.to("cpu")
        return ref

    def snapshot_ref_from_live(self) -> None:
        """把当前 live encoder 冻成 ``q_ref``（Stage2 首启：EMA 覆盖之后）。"""
        self._ref_cpu[0] = self._freeze_copy(self.latent)

    def snapshot_ref_from_named_tensors(
        self,
        named: dict[str, torch.Tensor],
        owner: nn.Module,
    ) -> int:
        """从整模 ``named``（通常是 Stage1 EMA）抽出本 ``latent`` 子树写入 ref。

        续训时 live 已是本 run 中段权重，不能从 live 重冻，否则 ref-KL 被抹掉。
        """
        prefix: str | None = None
        for name, mod in owner.named_modules():
            if mod is self.latent:
                prefix = f"{name}." if name else ""
                break
        if prefix is None:
            raise ValueError("找不到 LatentBundle.latent 在模型中的模块前缀")
        ref = self._freeze_copy(self.latent)
        n = 0
        with torch.no_grad():
            for pname, param in ref.named_parameters():
                src = named.get(prefix + pname)
                if src is None or not isinstance(src, torch.Tensor):
                    continue
                param.copy_(src.detach().cpu().to(dtype=param.dtype))
                n += 1
        if n < 1:
            raise ValueError("Stage1 EMA 与当前 latent 没有同名参数，无法重冻 q_ref")
        self._ref_cpu[0] = ref
        return n

    def on_tokens_seen(self, n: int, optimizer: Any = None) -> bool:
        """``mid`` 到达解冻点时解冻一次，并尝试把新参加进 optimizer。"""
        if self.tune != _TUNE_MID or self._thawed:
            return False
        if int(n) < self.latent_thaw_tokens:
            return False
        self._thawed = True
        self._set_requires_grad(True)
        self._require_vae_dec_if_needed()
        _try_add_to_optimizer(optimizer, list(self.latent.parameters()))
        return True

    def encode(
        self,
        tokens: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_trainable:
            with torch.no_grad():
                return self._module().encode(tokens, sample=sample)
        return self._module().encode(tokens, sample=sample)

    def _pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        layout = getattr(self._module(), "token_layout", None)
        pad_id = getattr(layout, "pad_token_id", None)
        if pad_id is None:
            return torch.zeros(tokens.shape, device=tokens.device, dtype=torch.bool)
        return tokens == pad_id

    @property
    def block_size(self) -> int:
        mod = self._module()
        if not hasattr(mod, "block_size"):
            raise ValueError("入口须暴露 block_size，禁止缺省为 1")
        return int(mod.block_size)

    @property
    def latent_dim(self) -> int:
        if not hasattr(self._module(), "latent_dim"):
            raise ValueError("入口须暴露 latent_dim，禁止缺省为 0")
        dim = int(self._module().latent_dim)
        if dim <= 0:
            raise ValueError(f"入口 latent_dim 须为正，收到 {dim}")
        return dim

    def _ignore_index(self) -> int:
        layout = getattr(self._module(), "token_layout", None)
        return int(getattr(layout, "ignore_index", -100))

    def _vocab_size(self, logits: torch.Tensor) -> int:
        return int(getattr(self._module(), "vocab_size", logits.size(-1)))

    def _beta_kl(self) -> float:
        return float(getattr(self._module(), "beta_kl", 0.1))

    def _lambda_mask(self) -> float:
        return float(getattr(self._module(), "lambda_mask", 0.0))

    def _lambda_vae(self) -> float:
        """重建项：优先加载入口，缺省回退 YAML（默认 1）。"""
        return float(getattr(self._module(), "lambda_vae", self.lambda_vae))

    def _lambda_ref(self) -> float:
        """ref-KL：优先加载入口，缺省回退 YAML（默认 1）。"""
        return float(getattr(self._module(), "lambda_ref", self.lambda_ref))

    def _ref_kl(
        self,
        tokens: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``KL(q||q_ref)`` 对有效位平均；PAD 与 prior-KL / CE 一样排除。"""
        with torch.no_grad():
            ref = self._ref_module()
            # 冻副本不进 module 树；首次 s1 时挪到与 q 同设备，避免 CPU encode。
            ref_dev = next(ref.parameters()).device
            if ref_dev != mu.device:
                ref.to(device=mu.device)
            _, mu_ref, logvar_ref = ref.encode(tokens, sample=False)
            mu_ref = mu_ref.to(dtype=mu.dtype)
            logvar_ref = logvar_ref.to(dtype=logvar.dtype)
        var = logvar.exp()
        var_ref = logvar_ref.exp()
        kl = 0.5 * (
            (logvar_ref - logvar)
            + (var + (mu - mu_ref).pow(2)) / var_ref.clamp(min=1e-6)
            - 1.0
        )
        per_tok = kl.mean(dim=-1)
        if pad is None:
            pad = self._pad_mask(tokens)
        weight = (~pad).to(dtype=per_tok.dtype)
        denom = weight.sum().clamp_min(1.0)
        return (per_tok * weight).sum() / denom

    def s1_loss(
        self,
        tokens: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        logvar: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """重建 CE + 后验项（prior-KL，开 kl_entropy 时再加 E[log q]）+ BERT-mask + ref-KL。frozen 为 0。"""
        zero = torch.zeros((), device=tokens.device, dtype=torch.float32)
        if not self.is_trainable:
            return zero

        if mu is None or logvar is None or z is None:
            z, mu, logvar = self.encode(tokens, sample=self.training)
        if logits is None:
            kpm = None
            attn_pad = getattr(self._module(), "_attn_pad_mask", None)
            if callable(attn_pad):
                kpm = attn_pad(tokens)
            logits = self._module().decode_logits(z, key_padding_mask=kpm)

        ignore = self._ignore_index()
        targets = tokens.clone()
        pad = self._pad_mask(tokens)
        targets = targets.masked_fill(pad, ignore)
        ce = F.cross_entropy(
            logits.reshape(-1, self._vocab_size(logits)),
            targets.reshape(-1),
            ignore_index=ignore,
        )
        kl = posterior_regularizer(
            mu, logvar, mask=~pad, kl_entropy=self.kl_entropy,
        )

        mask_loss = zero.to(dtype=ce.dtype)
        lambda_mask = self._lambda_mask()
        mask_ratio = float(getattr(self._module(), "mask_ratio", 0.0))
        if (
            self.training
            and lambda_mask > 0
            and mask_ratio > 0
            and hasattr(self._module(), "bert_mask_loss")
        ):
            mask_loss = self._module().bert_mask_loss(tokens)

        ref_loss = zero.to(dtype=ce.dtype)
        lambda_ref = self._lambda_ref()
        if lambda_ref > 0:
            ref_loss = self._ref_kl(tokens, mu, logvar, pad=pad)

        return (
            self._lambda_vae() * ce
            + self._beta_kl() * kl
            + lambda_mask * mask_loss
            + lambda_ref * ref_loss
        )
