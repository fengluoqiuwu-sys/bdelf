"""JacEllipsoid：冻 T5 表上的对称化偶二次精度与 QDA logit。

主模型把 token k 写成 p(z|x=k)=N(mu_k, G_k^{-1})，
G_k = F^T F + lambda I（F 为 r x d），再 Bayes 得读出 logit。
F 由 unembed 行差 / 近面 m 闭式给出（jacobian），或同秩可学（learned）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

QDA_MODES = ("jacobian", "isotropic", "softmax", "learned")

# claims.md：M、秩、λ 比例
DEFAULT_TOP_M = 16
DEFAULT_RANK = 16
DEFAULT_LAMBDA_SCALE = 1.0e-2
COVERAGE_FLOOR = 0.50


@dataclass
class QDATables:
    """冻表 QDA 参数（与 ELF 的 512 维 encoder 空间对齐：embedding / latent_std）。"""

    mu: torch.Tensor  # [K, d]
    factor: torch.Tensor  # [K, r, d]；G^{(0)}=F^T F
    logdet: torch.Tensor  # [K]
    log_pi: torch.Tensor  # [K]
    lam: float
    constructable: torch.Tensor  # [K] bool；top-M 对手全部 m>0
    principal: torch.Tensor  # [K, d] 单位主轴（G^{(0)} 最大特征）
    radius_unit: float  # median m/||v||，探针半径网格用
    coverage: float
    meta: dict[str, Any]


def _as_lam(lam: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(lam):
        return lam.to(device=ref.device, dtype=ref.dtype)
    return ref.new_tensor(float(lam))


def logdet_from_factor(factor: torch.Tensor, lam: float | torch.Tensor, dim: int) -> torch.Tensor:
    """log det(F^T F + λ I_d) = (d-r) log λ + log det(λ I_r + F F^T)。"""
    k, rank, _ = factor.shape
    lam_t = _as_lam(lam, factor)
    if rank == 0:
        return factor.new_full((k,), dim * torch.log(lam_t))
    gram = torch.matmul(factor, factor.transpose(-1, -2))
    eye = torch.eye(rank, device=factor.device, dtype=factor.dtype)
    sign, logabs = torch.linalg.slogdet(gram + lam_t * eye)
    extra = (dim - rank) * torch.log(lam_t)
    return logabs + extra


def qda_logits(
    z: torch.Tensor,
    *,
    mu: torch.Tensor,
    factor: torch.Tensor,
    lam: float | torch.Tensor,
    logdet: torch.Tensor,
    log_pi: torch.Tensor,
    vocab_chunk: int | None = None,
) -> torch.Tensor:
    """QDA logit：-½(z-μ)^T (F^T F+λI)(z-μ) + ½ logdet + log π。

    ``z``: ``[B, S, d]``；返回 ``[B, S, K]``。按词表分块；秩方向一次 GEMM，
    避免 Python 循环 ``r`` 次，也避免物化 ``[B,S,K,d]``。
    """
    bsz, seq, dim = z.shape
    vocab = mu.shape[0]
    n_tok = bsz * seq
    z_sq = (z * z).sum(dim=-1, keepdim=True)
    mu_sq = (mu * mu).sum(dim=-1)
    out = z.new_empty(bsz, seq, vocab)
    rank = int(factor.shape[1])
    lam_t = _as_lam(lam, z)
    z_flat = z.reshape(n_tok, dim)
    if vocab_chunk is None:
        vocab_chunk = _adaptive_vocab_chunk(n_tok, rank)
    for start in range(0, vocab, vocab_chunk):
        sl = slice(start, min(start + vocab_chunk, vocab))
        mu_c = mu[sl]
        cross = torch.matmul(z, mu_c.transpose(0, 1))
        mahal = z_sq + mu_sq[sl] - 2.0 * cross
        ell = -0.5 * lam_t * mahal
        if rank > 0:
            fac = factor[sl]
            chunk = int(fac.shape[0])
            a_dot_mu = torch.einsum("crd,cd->cr", fac, mu_c)
            dots = z_flat.matmul(fac.reshape(chunk * rank, dim).transpose(0, 1))
            dots = dots.view(bsz, seq, chunk, rank)
            ell = ell - 0.5 * ((dots - a_dot_mu) ** 2).sum(dim=-1)
        ell = ell + 0.5 * logdet[sl] + log_pi[sl]
        out[:, :, sl] = ell
    return out


def _adaptive_vocab_chunk(n_tok: int, rank: int, *, default: int = 2048) -> int:
    """限制 ``[N, C, r]`` 峰值：约 256MiB float。"""
    if rank <= 0:
        return default
    cap = 64_000_000
    chunk = cap // max(int(n_tok) * int(rank), 1)
    return max(64, min(default, chunk))


def align_codebook_vocab(
    mu: torch.Tensor,
    unembed: torch.Tensor,
    vocab: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """T5 权重词表（常 32128）截到 tokenizer 词表（常 32100）。"""
    n = int(mu.shape[0])
    vocab = int(vocab)
    if n < vocab:
        raise ValueError(f"T5 codebook rows {n} < tokenizer vocab {vocab}")
    if n > vocab:
        mu = mu[:vocab].contiguous()
        unembed = unembed[:vocab].contiguous()
    return mu, unembed


def load_t5_codebook(
    encoder_model_name: str = "t5-small",
    *,
    latent_mean: float = 0.0,
    latent_std: float = 0.2,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    vocab: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """冻 T5 shared embedding：μ 与 unembed 行（tied 时同一张表）缩放到 ELF latent。"""
    import hf_config  # noqa: F401
    from transformers import T5ForConditionalGeneration

    from models.jac_ellipsoid.t5_encoder import ensure_t5_encoder_cached

    local_dir = ensure_t5_encoder_cached(encoder_model_name)
    hf = T5ForConditionalGeneration.from_pretrained(
        local_dir, local_files_only=True,
    )
    hf.eval()
    shared = hf.shared.weight.detach().to(dtype=dtype)
    lm = hf.lm_head.weight.detach().to(dtype=dtype)
    scale = max(float(latent_std), 1e-8)
    mu = (shared - float(latent_mean)) / scale
    unembed = (lm - float(latent_mean)) / scale
    special = {
        "pad_token_id": int(getattr(hf.config, "pad_token_id", 0) or 0),
        "eos_token_id": int(getattr(hf.config, "eos_token_id", 1) or 1),
        "unk_token_id": int(getattr(hf.config, "unk_token_id", 2) or 2),
        "vocab_size": int(shared.shape[0]),
        "dim": int(shared.shape[1]),
    }
    del hf
    if vocab is not None:
        mu, unembed = align_codebook_vocab(mu, unembed, vocab)
        special["vocab_size"] = int(vocab)
    if device is not None:
        mu = mu.to(device)
        unembed = unembed.to(device)
    return mu, unembed, special


def svd_lowrank_slabs(
    a: torch.Tensor,
    *,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对 ``A_k ∈ R^{M×d}`` 做薄 SVD，返回 ``(factor[k,r,d], principal[k,d], eigmax[k])``。"""
    _u, s_svd, vh = torch.linalg.svd(a, full_matrices=False)
    del _u
    r_use = min(int(rank), int(s_svd.shape[-1]))
    factor = a.new_zeros(a.shape[0], rank, a.shape[-1])
    factor[:, :r_use] = s_svd[:, :r_use].unsqueeze(-1) * vh[:, :r_use]
    principal = F.normalize(vh[:, 0], dim=-1)
    eigmax = (s_svd[:, 0] ** 2).clamp(min=0.0)
    return factor, principal, eigmax


def factorize_slabs(
    a: torch.Tensor,
    *,
    rank: int,
    batch: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """分块 ``svd_lowrank_slabs``，避免一次 SVD 全词表。"""
    n = int(a.shape[0])
    if n <= int(batch):
        return svd_lowrank_slabs(a, rank=rank)
    factor = a.new_zeros(n, rank, a.shape[-1])
    principal = a.new_zeros(n, a.shape[-1])
    eigmax = a.new_zeros(n)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        f, p, e = svd_lowrank_slabs(a[start:end], rank=rank)
        factor[start:end] = f
        principal[start:end] = p
        eigmax[start:end] = e
    return factor, principal, eigmax


@torch.no_grad()
def build_jacobian_tables(
    mu: torch.Tensor,
    unembed: torch.Tensor,
    *,
    top_m: int = DEFAULT_TOP_M,
    rank: int = DEFAULT_RANK,
    lambda_scale: float = DEFAULT_LAMBDA_SCALE,
    log_pi: torch.Tensor | None = None,
    batch_k: int = 1024,
) -> QDATables:
    """M0：v=u_k-u_j，G_k=sum vv^T/m^2+λI（SVD 截断到 r）。"""
    vocab, dim = mu.shape
    top_m = int(top_m)
    rank = min(int(rank), top_m, dim)
    device = mu.device
    dtype = mu.dtype

    factor = mu.new_zeros(vocab, rank, dim)
    constructable = torch.zeros(vocab, dtype=torch.bool, device=device)
    principal = mu.new_zeros(vocab, dim)
    radius_vals: list[torch.Tensor] = []
    eigmax = mu.new_zeros(vocab)
    pos_counts = mu.new_zeros(vocab, dtype=torch.long)

    for start in range(0, vocab, batch_k):
        end = min(start + batch_k, vocab)
        idx = torch.arange(start, end, device=device)
        mu_b = mu[idx]
        logits = F.linear(mu_b, unembed)
        logits = logits.clone()
        logits[torch.arange(end - start, device=device), idx] = torch.finfo(dtype).min
        _topv, comp = logits.topk(top_m, dim=-1)
        u_k = unembed[idx].unsqueeze(1)
        u_j = unembed[comp]
        v = u_k - u_j
        m = (v * mu_b.unsqueeze(1)).sum(dim=-1)
        pos = m > 0
        pos_counts[idx] = pos.sum(dim=-1)
        constructable[idx] = pos.all(dim=-1)
        inv_m = torch.where(pos, 1.0 / m.clamp(min=1e-8), torch.zeros_like(m))
        a = v * inv_m.unsqueeze(-1)
        fac_b, prin_b, eig_b = svd_lowrank_slabs(a, rank=rank)
        factor[idx, : fac_b.shape[1]] = fac_b
        principal[idx] = prin_b
        eigmax[idx] = eig_b
        nrm = v.norm(dim=-1).clamp(min=1e-8)
        radius_vals.append((m / nrm)[pos])

    if radius_vals:
        radius_cat = torch.cat([x.reshape(-1) for x in radius_vals if x.numel() > 0])
        radius_unit = float(radius_cat.median().item()) if radius_cat.numel() else 1.0
    else:
        radius_unit = 1.0

    pos_eig = eigmax[eigmax > 0]
    med_lmax = float(pos_eig.median().item()) if pos_eig.numel() else 1.0
    lam = float(lambda_scale) * max(med_lmax, 1e-8)
    logdet = logdet_from_factor(factor, lam, dim)
    if log_pi is None:
        log_pi = mu.new_full((vocab,), -torch.log(torch.tensor(float(vocab), dtype=dtype)))
    coverage = float(constructable.float().mean().item())
    meta = {
        "top_m": top_m,
        "rank": rank,
        "lambda_scale": float(lambda_scale),
        "lam": lam,
        "coverage": coverage,
        "coverage_floor": COVERAGE_FLOOR,
        "median_lambda_max": med_lmax,
        "radius_unit": radius_unit,
        "n_constructable": int(constructable.sum().item()),
        "vocab_size": vocab,
        "dim": dim,
    }
    return QDATables(
        mu=mu,
        factor=factor,
        logdet=logdet,
        log_pi=log_pi,
        lam=lam,
        constructable=constructable,
        principal=principal,
        radius_unit=radius_unit,
        coverage=coverage,
        meta=meta,
    )


def build_isotropic_tables(
    mu: torch.Tensor,
    *,
    log_pi: torch.Tensor | None = None,
) -> QDATables:
    """Plaid 式对照：全体 G∝I。λ 取 1 / median(||μ||²/d)。"""
    vocab, dim = mu.shape
    sq = (mu * mu).sum(dim=-1) / float(dim)
    med = float(sq.median().clamp(min=1e-8).item())
    lam = 1.0 / med
    factor = mu.new_zeros(vocab, 0, dim)
    logdet = mu.new_full((vocab,), dim * torch.log(torch.tensor(lam, device=mu.device, dtype=mu.dtype)))
    if log_pi is None:
        log_pi = mu.new_full((vocab,), -torch.log(torch.tensor(float(vocab), dtype=mu.dtype)))
    principal = F.normalize(mu, dim=-1)
    return QDATables(
        mu=mu,
        factor=factor,
        logdet=logdet,
        log_pi=log_pi,
        lam=lam,
        constructable=torch.ones(vocab, dtype=torch.bool, device=mu.device),
        principal=principal,
        radius_unit=1.0,
        coverage=1.0,
        meta={"mode": "isotropic", "lam": lam, "coverage": 1.0},
    )


def qda_logits_diag(
    z: torch.Tensor,
    *,
    mu: torch.Tensor,
    diag_w: torch.Tensor,
    log_pi: torch.Tensor,
    vocab_chunk: int = 2048,
) -> torch.Tensor:
    """对角 G：-½ Σ_i w_{k,i}(z_i-μ_{k,i})² + ½ Σ log w + log π。"""
    bsz, seq, dim = z.shape
    vocab = mu.shape[0]
    z2 = z * z
    out = z.new_empty(bsz, seq, vocab)
    logdet = torch.log(diag_w.clamp(min=1e-12)).sum(dim=-1)
    for start in range(0, vocab, vocab_chunk):
        sl = slice(start, min(start + vocab_chunk, vocab))
        w = diag_w[sl]
        mu_c = mu[sl]
        term_z2 = torch.matmul(z2, w.transpose(0, 1))
        term_cross = torch.matmul(z, (w * mu_c).transpose(0, 1))
        const = (w * mu_c * mu_c).sum(dim=-1)
        ell = -0.5 * (term_z2 - 2.0 * term_cross + const)
        ell = ell + 0.5 * logdet[sl] + log_pi[sl]
        out[:, :, sl] = ell
    return out


def diag_from_factor(factor: torch.Tensor, lam: float | torch.Tensor) -> torch.Tensor:
    """diag(F^T F) + λ，形状 [K, d]。"""
    return (factor * factor).sum(dim=1) + _as_lam(lam, factor)


def pad_tables_to_vocab(
    blob: dict,
    *,
    vocab: int,
    dim: int,
    fallback_mu: torch.Tensor | None = None,
) -> dict[str, Any]:
    """把探针 tables.pt（可能只覆盖前 n 个 k）补到全词表，缺的 token 用 λI。"""
    mu = blob["mu"]
    factor = blob["factor"]
    n, d = int(mu.shape[0]), int(mu.shape[1])
    if d != dim:
        raise ValueError(f"tables dim {d} != {dim}")
    if n > vocab:
        mu = mu[:vocab]
        factor = factor[:vocab]
        n = vocab
    rank = int(factor.shape[1]) if factor.ndim == 3 else 0
    lam = float(blob.get("lam", 1.0))
    device, dtype = mu.device, mu.dtype
    mu_full = torch.zeros(vocab, dim, device=device, dtype=dtype)
    if fallback_mu is not None:
        mu_full.copy_(fallback_mu.to(device=device, dtype=dtype)[:vocab])
    mu_full[:n] = mu
    fac_full = torch.zeros(vocab, rank, dim, device=device, dtype=dtype)
    if rank > 0:
        fac_full[:n] = factor
    log_pi = blob.get("log_pi")
    if log_pi is None or int(log_pi.shape[0]) != vocab:
        log_pi = mu_full.new_full((vocab,), -torch.log(torch.tensor(float(vocab), dtype=dtype)))
    else:
        log_pi = log_pi.to(device=device, dtype=dtype)
    constructable = torch.zeros(vocab, dtype=torch.bool, device=device)
    if "constructable" in blob:
        c = blob["constructable"].to(device=device)
        constructable[: min(n, int(c.shape[0]))] = c[: min(n, int(c.shape[0]))]
    return {
        "mu": mu_full,
        "factor": fac_full,
        "log_pi": log_pi,
        "lam": lam,
        "constructable": constructable,
        "n_loaded": n,
        "meta": blob.get("meta") or {},
        "_path": blob.get("_path"),
    }


def load_qda_tables_file(path: str | Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """读 M0/M1 探针写出的 tables.pt。"""
    p = Path(path)
    if not p.is_absolute():
        root = repo_root or Path(__file__).resolve().parents[2]
        p = root / p
    if not p.is_file():
        raise FileNotFoundError(f"QDA tables 不存在: {p}")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "mu" not in blob or "factor" not in blob:
        raise ValueError(f"{p} 不是 JacEllipsoid tables.pt（需 mu/factor）")
    blob["_path"] = str(p)
    return blob


def unembed_argmax(z: torch.Tensor, unembed: torch.Tensor, *, chunk: int = 4096) -> torch.Tensor:
    """线性 unembed：argmax_k u_k^T z。``z`` ``[..., d]`` → 同前缀 long。"""
    flat = z.reshape(-1, z.shape[-1])
    n = flat.shape[0]
    vocab = unembed.shape[0]
    best = flat.new_full((n,), -1.0e9)
    arg = torch.zeros(n, dtype=torch.long, device=z.device)
    for start in range(0, vocab, chunk):
        sl = unembed[start : start + chunk]
        scores = torch.matmul(flat, sl.transpose(0, 1))
        val, idx = scores.max(dim=-1)
        better = val > best
        best = torch.where(better, val, best)
        arg = torch.where(better, idx + start, arg)
    return arg.view(*z.shape[:-1])
