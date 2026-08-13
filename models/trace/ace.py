"""ACE（Attractor-Contrast-Escape）：从 SC 反馈中减去重复吸引子方向。

推理期干预（Zhang et al., arXiv:2607.00588）：每步
``x_pred ← x_pred - λ · d``（``d`` 广播到长度维）。``ace`` 未写 / false / 0 时关闭。

方向 ``d`` 默认缓存在
``cache/checkpoints/full/ace/{model-hash}/{step}/direction.pt``：
有则加载，换 hash/step（缓存未命中）时用当前权重现算并写入。

估计轨迹兼容 ELF（``_sde/_ode_step`` → ``(z, x_pred)``）与 ODAR
（``(z, x_pred, logits)``；有 ``_apply_dma`` 时按采样路径做 DMA-H 再收集 SC）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

# 与 train.run_path.CHECKPOINT_ROOT 对齐（避免 models→train 循环导入）
CHECKPOINT_ROOT = "cache/checkpoints"

# 论文默认剂量；可用窗约 [1.5, 5]
DEFAULT_ACE_LAMBDA = 2.0
# 估计轨迹数（论文常用 400；本地默认 512）
DEFAULT_ACE_ESTIMATE_N = 512
DEFAULT_ACE_ESTIMATE_BS = 8
DEFAULT_ACE_ESTIMATE_SEED = 0

DIRECTION_FILENAME = "direction.pt"
META_FILENAME = "meta.json"


def ace_cache_root() -> Path:
    """``cache/checkpoints/full/ace``。"""
    return Path(CHECKPOINT_ROOT) / "full" / "ace"


def ace_cache_dir(model_hash: str, step: int) -> Path:
    if not model_hash or any(c in model_hash for c in "/\\"):
        raise ValueError(f"invalid ace model_hash: {model_hash!r}")
    return ace_cache_root() / str(model_hash) / str(int(step))


def ace_direction_path(model_hash: str, step: int) -> Path:
    return ace_cache_dir(model_hash, step) / DIRECTION_FILENAME


def model_hash_from_checkpoint(ckpt_path: str | Path) -> str | None:
    """从 ``cache/checkpoints/{variant}/{model}/{hash}/…`` 解析训练 hash。"""
    path = Path(ckpt_path).resolve()
    root = Path(CHECKPOINT_ROOT).resolve()
    try:
        rel = path.relative_to(root)
    except ValueError:
        # 允许 ckpt 已是 run 目录下的文件但 CHECKPOINT_ROOT 相对 cwd 不一致
        parts = path.parts
        if "checkpoints" not in parts:
            return None
        i = parts.index("checkpoints")
        rel_parts = parts[i + 1 :]
    else:
        rel_parts = rel.parts
    # {variant}/{model}/{hash}/checkpoint_*.pt
    if len(rel_parts) < 3:
        return None
    return str(rel_parts[2])


def attach_ace_identity(
    model: Any,
    *,
    model_hash: str,
    step: int,
    tokenizer: str | None = None,
) -> None:
    """在 model / backbone 上挂 ACE 缓存身份（hash + step）。"""
    step_i = int(step)
    for obj in (model, getattr(model, "backbone", None)):
        if obj is None:
            continue
        obj._ace_model_hash = str(model_hash)
        obj._ace_step = step_i
        if tokenizer:
            obj._ace_tokenizer = str(tokenizer)


def ace_is_enabled(ace: Any) -> bool:
    """``ace`` 是否表示开启（缺省 / false / 0 / 空串 → 关）。"""
    if ace is None or ace is False:
        return False
    if isinstance(ace, (int, float)) and float(ace) == 0.0:
        return False
    if isinstance(ace, str) and not ace.strip():
        return False
    if isinstance(ace, dict) and not ace:
        return False
    return True


def parse_ace_lambda(ace: Any) -> float:
    """解析 ACE 剂量 λ；关闭时返回 0。"""
    if not ace_is_enabled(ace):
        return 0.0
    if ace is True:
        return DEFAULT_ACE_LAMBDA
    if isinstance(ace, (int, float)):
        lam = float(ace)
        if lam < 0:
            raise ValueError(f"ace lambda must be >= 0, got {lam}")
        return lam
    if isinstance(ace, str):
        s = ace.strip()
        try:
            lam = float(s)
        except ValueError:
            return DEFAULT_ACE_LAMBDA  # 当作方向文件路径
        if lam < 0:
            raise ValueError(f"ace lambda must be >= 0, got {lam}")
        return lam
    if isinstance(ace, dict):
        raw = ace.get("lambda", ace.get("lam", DEFAULT_ACE_LAMBDA))
        if raw is True or raw is None:
            return DEFAULT_ACE_LAMBDA
        lam = float(raw)
        if lam < 0:
            raise ValueError(f"ace.lambda must be >= 0, got {lam}")
        return lam
    raise ValueError(
        f"ace must be false/0/true/number/path/dict, got {type(ace).__name__}"
    )


def resolve_ace_direction_path(ace: Any, ace_direction: Any) -> str | None:
    """显式方向路径；无显式指定时返回 None（走自动缓存）。"""
    if not ace_is_enabled(ace):
        return None
    if isinstance(ace, str) and ace.strip():
        # 纯数字字符串当作 λ，不算路径
        try:
            float(ace.strip())
        except ValueError:
            return ace.strip()
    if isinstance(ace, dict):
        for key in ("direction", "path", "d"):
            val = ace.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    if ace_direction is None or ace_direction is False:
        return None
    if isinstance(ace_direction, str) and ace_direction.strip():
        return ace_direction.strip()
    raise ValueError(
        f"ace_direction must be a path string, got {type(ace_direction).__name__}"
    )


def _ace_identity(
    cfg: dict[str, Any],
    backbone: Any,
) -> tuple[str | None, int | None]:
    h = cfg.get("ace_model_hash")
    if h is None:
        h = getattr(backbone, "_ace_model_hash", None)
    s = cfg.get("ace_step")
    if s is None:
        s = getattr(backbone, "_ace_step", None)
    if h is not None:
        h = str(h)
    if s is not None:
        s = int(s)
    return h, s


def load_ace_direction(
    path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """加载 ``d``（dict 含 ``d`` 或裸 tensor）；返回 shape ``(C,)``。"""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        if "d" not in blob:
            raise ValueError(f"ACE direction file {path} dict missing key 'd'")
        d = blob["d"]
    else:
        d = blob
    if not isinstance(d, torch.Tensor):
        raise TypeError(f"ACE direction must be a Tensor, got {type(d).__name__}")
    d = d.detach().float().reshape(-1)
    if expected_dim is not None and d.numel() != expected_dim:
        raise ValueError(
            f"ACE direction dim {d.numel()} != text_encoder_dim {expected_dim}"
        )
    return d.to(device=device, dtype=dtype)


def save_ace_direction(
    path: str | Path,
    d: torch.Tensor,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """原子写入 ``direction.pt``（及可选 ``meta.json``）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"d": d.detach().float().cpu().reshape(-1)}
    if meta:
        payload["meta"] = dict(meta)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    if meta is not None:
        meta_path = path.parent / META_FILENAME
        meta_tmp = meta_path.with_suffix(".tmp")
        meta_tmp.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(meta_tmp, meta_path)
    return path


def _log(msg: str) -> None:
    print(f"[ace] {msg}", file=sys.stderr, flush=True)


def _unpack_sampler_step(out: Any) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
    """兼容 ELF ``(z, x_pred)`` 与 ODAR ``(z, x_pred, logits)`` 步进返回值。"""
    if not isinstance(out, tuple) or len(out) < 2:
        raise TypeError(
            f"sampler step must return (z, x_pred[, ...]), got {type(out).__name__}"
        )
    z, x_pred = out[0], out[1]
    logits = out[2] if len(out) >= 3 else None
    return z, x_pred, logits


@torch.no_grad()
def _run_trajectory_collect_sc(
    backbone: Any,
    *,
    z: torch.Tensor,
    sampling_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """无 ACE 采样一条轨迹，返回 ``(z_final, 轨迹平均 SC 反馈 u)``，``u`` 为 ``(B, C)``。

    步进返回值兼容 ELF 二元组与 ODAR 三元组（多出的 logits 仅用于可选 DMA）。
    若 backbone 有 ``_apply_dma``（ODAR），在收集 SC 前按采样路径做 DMA-H，
    使估计的吸引子方向与真实 SC 反馈一致。
    """
    cfg = dict(sampling_cfg)
    cfg["ace"] = False
    cfg.pop("ace_direction", None)

    method = str(cfg.get("sampling_method", "sde")).lower()
    num_sampling_steps = int(cfg.get("num_sampling_steps", 32))
    sde_gamma = float(cfg.get("sde_gamma", 1.5))
    sc_cfg_w = float(cfg.get("self_cond_cfg_scale", 1.0))
    infer_schedule = cfg.get("time_schedule")
    if infer_schedule is not None:
        backbone._infer_time_schedule = infer_schedule

    device = z.device
    dtype = z.dtype
    bsz = z.shape[0]
    t_steps = backbone._get_sampling_steps(num_sampling_steps, device, dtype)

    if backbone.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = torch.full(
            (bsz,), sc_cfg_w, dtype=dtype, device=device,
        )
    elif sc_cfg_w != 1.0:
        self_cond_cfg_scale = torch.full(
            (bsz,), sc_cfg_w, dtype=dtype, device=device,
        )
    else:
        self_cond_cfg_scale = None

    apply_dma = getattr(backbone, "_apply_dma", None)
    x_pred: torch.Tensor | None = None
    sc_sum: torch.Tensor | None = None
    sc_count = 0

    for i in range(t_steps.numel() - 2):
        t = float(t_steps[i].item())
        t_next = float(t_steps[i + 1].item())
        if method == "sde":
            out = backbone._sde_step(
                z, t, t_next, x_pred, sde_gamma,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )
        elif method == "ode":
            out = backbone._ode_step(
                z, t, t_next, x_pred,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )
        else:
            raise ValueError(f"unknown sampling_method: {method}")
        z, x_pred, step_logits = _unpack_sampler_step(out)
        # ODAR：与 generate 一致，SC 反馈在送入下一步前经 DMA-H
        if (
            apply_dma is not None
            and x_pred is not None
            and step_logits is not None
        ):
            x_pred = apply_dma(x_pred, t, logits=step_logits)
        if x_pred is not None:
            # 池化长度维，与 ACE-DLM collect 一致
            u = x_pred.mean(dim=1)
            sc_sum = u if sc_sum is None else sc_sum + u
            sc_count += 1

    t = float(t_steps[-2].item())
    t_next = float(t_steps[-1].item())
    z, _, _ = _unpack_sampler_step(
        backbone._ode_step(
            z, t, t_next, x_pred,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
    )
    if sc_sum is None or sc_count == 0:
        raise RuntimeError("ACE estimate collected no self-conditioning feedback")
    return z, sc_sum / float(sc_count)


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


@torch.no_grad()
def estimate_ace_direction(
    backbone: Any,
    *,
    sampling_cfg: dict[str, Any],
    tokenizer_name: str,
    n: int = DEFAULT_ACE_ESTIMATE_N,
    batch_size: int = DEFAULT_ACE_ESTIMATE_BS,
    seed: int = DEFAULT_ACE_ESTIMATE_SEED,
    seqlen: int | None = None,
) -> torch.Tensor:
    """差分均值方向：``mean(u|top-rep) - mean(u|bottom-rep)``，再单位化。"""
    d, _meta = estimate_ace_direction_with_stats(
        backbone,
        sampling_cfg=sampling_cfg,
        tokenizer_name=tokenizer_name,
        n=n,
        batch_size=batch_size,
        seed=seed,
        seqlen=seqlen,
    )
    return d


def _tokenizer_name(backbone: Any, cfg: dict[str, Any]) -> str:
    name = cfg.get("ace_tokenizer")
    if isinstance(name, str) and name.strip():
        return name.strip()
    attached = getattr(backbone, "_ace_tokenizer", None)
    if isinstance(attached, str) and attached.strip():
        return attached.strip()
    parent_cfg = getattr(backbone, "config", None)
    if parent_cfg is not None and getattr(parent_cfg, "tokenizer", None):
        return str(parent_cfg.tokenizer)
    return "t5-small"


def _mem_cache_get(
    backbone: Any | None,
    *,
    key: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """进程内已加载的 ``d``；换路径 / device / dtype 则失效。"""
    if backbone is None:
        return None
    cached = getattr(backbone, "_ace_d_cache", None)
    if not isinstance(cached, dict):
        return None
    if cached.get("key") != key:
        return None
    d = cached.get("d")
    if not isinstance(d, torch.Tensor):
        return None
    if d.device != device or d.dtype != dtype:
        d = d.to(device=device, dtype=dtype)
        cached["d"] = d
    return d


def _mem_cache_set(backbone: Any | None, *, key: str, d: torch.Tensor) -> None:
    if backbone is None:
        return
    backbone._ace_d_cache = {"key": key, "d": d}


def get_or_estimate_ace_direction(
    backbone: Any,
    *,
    sampling_cfg: dict[str, Any],
    model_hash: str,
    step: int,
    device: torch.device,
    dtype: torch.dtype,
    expected_dim: int,
) -> tuple[torch.Tensor, Path]:
    """磁盘缓存命中则加载，否则现算并写入 ``full/ace/{hash}/{step}/``。

    同一 backbone 上对同一路径只从磁盘读一次（micro-batch 复用内存中的 ``d``）。
    """
    path = ace_direction_path(model_hash, step)
    key = str(path.resolve()) if path.exists() else str(path)
    cached = _mem_cache_get(backbone, key=key, device=device, dtype=dtype)
    if cached is not None:
        return cached, path

    if path.is_file():
        _log(f"load direction {path}")
        d = load_ace_direction(
            path, device=device, dtype=dtype, expected_dim=expected_dim,
        )
        _mem_cache_set(backbone, key=key, d=d)
        return d, path

    n = int(sampling_cfg.get("ace_estimate_n", DEFAULT_ACE_ESTIMATE_N))
    bs = int(sampling_cfg.get("ace_estimate_bs", DEFAULT_ACE_ESTIMATE_BS))
    seed = int(sampling_cfg.get("ace_estimate_seed", DEFAULT_ACE_ESTIMATE_SEED))
    tok_name = _tokenizer_name(backbone, sampling_cfg)
    _log(f"cache miss → estimate → {path}")
    d_cpu = estimate_ace_direction(
        backbone,
        sampling_cfg=sampling_cfg,
        tokenizer_name=tok_name,
        n=n,
        batch_size=bs,
        seed=seed,
    )
    meta = {
        "model_hash": model_hash,
        "step": int(step),
        "n": n,
        "batch_size": bs,
        "seed": seed,
        "tokenizer": tok_name,
        "text_encoder_dim": int(expected_dim),
        "num_sampling_steps": sampling_cfg.get("num_sampling_steps"),
        "self_cond_cfg_scale": sampling_cfg.get("self_cond_cfg_scale"),
        "sampling_method": sampling_cfg.get("sampling_method"),
        "sde_gamma": sampling_cfg.get("sde_gamma"),
    }
    save_ace_direction(path, d_cpu, meta=meta)
    _log(f"saved direction {path}")
    d = d_cpu.to(device=device, dtype=dtype)
    _mem_cache_set(backbone, key=str(path.resolve()), d=d)
    return d, path


def resolve_ace_steering(
    cfg: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    expected_dim: int,
    backbone: Any | None = None,
) -> tuple[float, torch.Tensor | None]:
    """从 sampling_cfg 解析 ``(λ, d)``；关闭时 ``(0.0, None)``。

    未给显式 ``ace_direction`` / 路径时：按 ``full/ace/{hash}/{step}/`` 加载或现算。
    """
    ace = cfg.get("ace", False)
    lam = parse_ace_lambda(ace)
    if lam == 0.0:
        return 0.0, None

    explicit = resolve_ace_direction_path(ace, cfg.get("ace_direction"))
    if explicit:
        key = str(Path(explicit).resolve())
        cached = _mem_cache_get(backbone, key=key, device=device, dtype=dtype)
        if cached is not None:
            return lam, cached
        _log(f"load direction {explicit}")
        d = load_ace_direction(
            explicit, device=device, dtype=dtype, expected_dim=expected_dim,
        )
        _mem_cache_set(backbone, key=key, d=d)
        return lam, d

    if backbone is None:
        raise ValueError(
            "ace is enabled without ace_direction; need backbone to auto "
            "load/estimate under cache/checkpoints/full/ace/{hash}/{step}/"
        )
    model_hash, step = _ace_identity(cfg, backbone)
    if not model_hash or step is None:
        raise ValueError(
            "ace is enabled but cache identity unknown: attach via "
            "attach_ace_identity(model, model_hash=..., step=...) "
            "(from checkpoint path), or set ace_direction=/path/to/direction.pt"
        )
    d, _path = get_or_estimate_ace_direction(
        backbone,
        sampling_cfg=cfg,
        model_hash=model_hash,
        step=step,
        device=device,
        dtype=dtype,
        expected_dim=expected_dim,
    )
    return lam, d


def apply_ace_steer(
    x_pred: torch.Tensor,
    *,
    lam: float,
    direction: torch.Tensor,
) -> torch.Tensor:
    """``x_pred - λ · d``，``d`` 广播到 ``(1, 1, C)``。"""
    return x_pred - lam * direction.view(1, 1, -1)
