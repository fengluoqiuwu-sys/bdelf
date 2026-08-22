#!/usr/bin/env python3
"""I-1 Hybrid bulk readout：M0 终态探针 + M1 结构/随机/类型门。

M0：现成 ELF EMA 采样，比 ZSBD vs native agreement、尾占比、类型直方图。
M1：同终态、同 |T|/L，结构门 vs 均匀随机门 vs 类型门；agreement + Gen.PPL。
读出不写回 SC。产物默认 ``temp/auto-research/hybrid-i1/``。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import repo_env

ROOT = repo_env.ensure_repo_root()
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import hf_config  # noqa: F401
from eval.gen_ppl import score_texts
from generate import (
    load_model_from_checkpoint,
    resolve_checkpoint,
    resolve_device,
    resolve_dtype,
    set_seed,
)
from hybrid_readout_elf import elf_decode_probe, elf_generate_latent
from models.lm.elf.t5_encoder import ensure_t5_encoder_cached
from tokenizer import get_tokenizer
from train.ema import swap_ema_weights
from train.generate_config import get_generate

RHO_GRID = (0.05, 0.07, 0.10)
ZSBD_VARIANTS = ("z", "z_unnorm", "h")


def _log(msg: str) -> None:
    print(f"[hybrid-probe] {msg}", flush=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def visible_gpu_ids() -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return list(range(n))
    return [int(x) for x in raw.split(",") if x.strip() != ""]


def shard_counts(n: int, shards: int) -> list[int]:
    base, rem = divmod(n, shards)
    return [base + (1 if i < rem else 0) for i in range(shards)]


def load_t5_emb(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    from transformers import T5EncoderModel

    local = ensure_t5_encoder_cached("t5-small")
    enc = T5EncoderModel.from_pretrained(local, local_files_only=True)
    emb = enc.get_input_embeddings().weight.detach().to(device=device, dtype=torch.float32)
    del enc
    return F.normalize(emb, dim=-1)


def zsbd_topk(
    x: torch.Tensor,
    emb_norm: torch.Tensor,
    *,
    chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """余弦近邻 argmax 与 top1-top2 gap。``x`` 未归一化亦可。"""
    x_n = F.normalize(x.float(), dim=-1)
    bsz, seq, dim = x_n.shape
    flat = x_n.reshape(-1, dim)
    n = flat.shape[0]
    vocab = emb_norm.shape[0]
    best = torch.full((n,), -1.0e9, device=x.device, dtype=torch.float32)
    best2 = torch.full((n,), -1.0e9, device=x.device, dtype=torch.float32)
    arg = torch.zeros(n, dtype=torch.long, device=x.device)
    for start in range(0, vocab, chunk):
        sl = emb_norm[start : start + chunk]
        scores = flat @ sl.T
        v1, i1 = scores.max(dim=-1)
        better = v1 > best
        best2 = torch.where(better, torch.maximum(best, best2), torch.maximum(best2, v1))
        best = torch.where(better, v1, best)
        arg = torch.where(better, i1 + start, arg)
    gap = (best - best2).view(bsz, seq)
    return arg.view(bsz, seq), gap


def classify_types(
    tokens: torch.Tensor,
    tokenizer,
    pad_id: int,
    eos_id: int,
    bos_id: int,
) -> dict[str, torch.Tensor]:
    pieces = tokenizer.convert_ids_to_tokens(tokens.reshape(-1).tolist())
    n = tokens.numel()
    digit = torch.zeros(n, dtype=torch.bool)
    subword = torch.zeros(n, dtype=torch.bool)
    special = torch.zeros(n, dtype=torch.bool)
    ids = tokens.reshape(-1)
    for i, piece in enumerate(pieces):
        tid = int(ids[i].item())
        if tid in (pad_id, eos_id, bos_id) or piece in ("<pad>", "</s>", "<unk>"):
            special[i] = True
            continue
        if any(ch.isdigit() for ch in piece):
            digit[i] = True
        if not piece.startswith("▁"):
            subword[i] = True
    shape = tokens.shape
    valid = ~special.view(shape)
    # 稀有：本批出现次数 = 1 的非 special token
    counts: dict[int, int] = {}
    for tid, is_sp in zip(ids.tolist(), special.tolist()):
        if is_sp:
            continue
        counts[tid] = counts.get(tid, 0) + 1
    rare = torch.zeros(n, dtype=torch.bool)
    for i, tid in enumerate(ids.tolist()):
        if not special[i] and counts.get(tid, 0) == 1:
            rare[i] = True
    return {
        "valid": valid,
        "digit": digit.view(shape),
        "subword": subword.view(shape),
        "rare": rare.view(shape),
        "special": special.view(shape),
    }


def agreement(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> float:
    m = valid.float()
    hit = ((a == b) & valid).float()
    den = float(m.sum().item())
    if den <= 0:
        return float("nan")
    return float((hit.sum() / den).item())


def frac_wrong(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> float:
    return 1.0 - agreement(a, b, valid)


def pick_k_indices(score: torch.Tensor, k: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """每行取 top-k（k 可变）；无效位置分数打到 -inf。返回 bool mask。"""
    filled = score.masked_fill(~valid, -1.0e9)
    bsz, seq = filled.shape
    # k 上限为有效位数
    k = torch.minimum(k, valid.sum(dim=-1).to(k.dtype))
    k = k.clamp(min=0)
    k_max = int(k.max().item()) if bsz else 0
    mask = torch.zeros(bsz, seq, dtype=torch.bool, device=score.device)
    if k_max <= 0:
        return mask
    topv, topi = filled.topk(k_max, dim=-1)
    for i in range(bsz):
        ki = int(k[i].item())
        if ki > 0:
            mask[i, topi[i, :ki]] = True
    return mask


def gate_masks(
    *,
    native: torch.Tensor,
    zsbd: torch.Tensor,
    margin: torch.Tensor,
    types: dict[str, torch.Tensor],
    rho: float,
    rng: torch.Generator,
) -> dict[str, torch.Tensor]:
    valid = types["valid"]
    bsz, seq = native.shape
    n_valid = valid.sum(dim=-1)
    k = (n_valid.float() * rho).round().to(torch.long)
    wrong = (native != zsbd) & valid
    type_pos = (types["digit"] | types["subword"] | types["rare"]) & valid
    # 结构：错位优先，其次类型，再次低 margin
    struct_score = (
        wrong.float() * 1.0e6
        + type_pos.float() * 1.0e3
        - margin.float()
    )
    type_score = type_pos.float() * 1.0e3 - margin.float()
    rand_score = torch.rand(
        bsz, seq, generator=rng, device=native.device, dtype=torch.float32,
    )
    rand_score = rand_score.masked_fill(~valid, -1.0e9)
    return {
        "struct": pick_k_indices(struct_score, k, valid),
        "rand": pick_k_indices(rand_score, k, valid),
        "type": pick_k_indices(type_score, k, valid),
    }


def hybrid_tokens(native: torch.Tensor, zsbd: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return torch.where(gate, native, zsbd)


def decode_texts(tokens: torch.Tensor, tokenizer) -> list[str]:
    out: list[str] = []
    for row in tokens:
        out.append(tokenizer.decode(row.tolist(), skip_special_tokens=True))
    return out


def sc_cfg_tensor(bb, sampling_cfg: dict, n: int, device, dtype) -> torch.Tensor | None:
    w = float(sampling_cfg.get("self_cond_cfg_scale", 1.0))
    if getattr(bb, "num_self_cond_cfg_tokens", 0) > 0 or w != 1.0:
        return torch.full((n,), w, device=device, dtype=dtype)
    return None


def run_m0_shard(
    *,
    args: argparse.Namespace,
    n_local: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_checkpoint(checkpoint=None, run=args.run)
    _log(f"load {ckpt} n={n_local} seed={seed} device={device}")
    model, model_meta, step, train_cfg = load_model_from_checkpoint(ckpt, device)
    dtype = resolve_dtype(device, train_cfg)
    tokenizer_name = model_meta["config"].get("tokenizer") or "t5-small"
    tokenizer = get_tokenizer(tokenizer_name)
    layout = tokenizer.get_token_layout()
    if model_meta["name"] == "elf":
        from models.lm.elf.ace import attach_ace_identity

        attach_ace_identity(
            model,
            model_hash=Path(args.run).name,
            step=step,
            tokenizer=tokenizer_name,
        )

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ema_raw = ck.get("ema")
    ema_state = None
    if isinstance(ema_raw, dict) and ema_raw:
        ema_state = {k: v.to(device=device) for k, v in ema_raw.items()}
    del ck

    sampling_cfg = get_generate(
        model_meta["name"], args.generate, overrides=None,
    ).to_sampling_cfg()
    bb = getattr(model, "backbone", model)
    emb_norm = load_t5_emb(device, dtype)
    mean = float(getattr(bb, "latent_mean", 0.0))
    std = float(getattr(bb, "latent_std", 0.2))

    zs: list[torch.Tensor] = []
    natives: list[torch.Tensor] = []
    zsbd: dict[str, list[torch.Tensor]] = {k: [] for k in ZSBD_VARIANTS}
    gaps: dict[str, list[torch.Tensor]] = {k: [] for k in ZSBD_VARIANTS}
    margins: list[torch.Tensor] = []

    remaining = n_local
    chunk_i = 0
    t0 = time.time()
    with swap_ema_weights(model, ema_state):
        model.eval()
        while remaining > 0:
            bs = min(args.micro_bs, remaining)
            set_seed(seed + chunk_i * 100003)
            sc = sc_cfg_tensor(bb, sampling_cfg, bs, device, dtype)
            with torch.no_grad(), torch.amp.autocast(
                device.type, dtype=dtype, enabled=device.type == "cuda",
            ):
                _tokens, _nfe, z = elf_generate_latent(
                    bb,
                    num_samples=bs,
                    seqlen=args.num_tokens,
                    sampling_cfg=sampling_cfg,
                )
                logits, hidden = elf_decode_probe(bb, z, sc)
            top2 = logits.float().topk(2, dim=-1)
            native = top2.indices[..., 0]
            margin = top2.values[..., 0] - top2.values[..., 1]
            native = bb._mask_after_eos(
                native,
                eos_token_id=layout.eos_token_id,
                pad_token_id=layout.pad_token_id,
            )
            pred_z, gap_z = zsbd_topk(z, emb_norm)
            pred_zu, gap_zu = zsbd_topk(z.float() * std + mean, emb_norm)
            pred_h, gap_h = zsbd_topk(hidden, emb_norm)
            for pred in (pred_z, pred_zu, pred_h):
                pred.copy_(
                    bb._mask_after_eos(
                        pred,
                        eos_token_id=layout.eos_token_id,
                        pad_token_id=layout.pad_token_id,
                    )
                )
            zs.append(z.detach().to(dtype=torch.bfloat16).cpu())
            natives.append(native.detach().cpu())
            zsbd["z"].append(pred_z.detach().cpu())
            zsbd["z_unnorm"].append(pred_zu.detach().cpu())
            zsbd["h"].append(pred_h.detach().cpu())
            gaps["z"].append(gap_z.detach().half().cpu())
            gaps["z_unnorm"].append(gap_zu.detach().half().cpu())
            gaps["h"].append(gap_h.detach().half().cpu())
            margins.append(margin.detach().half().cpu())
            remaining -= bs
            chunk_i += 1
            done = n_local - remaining
            _log(f"generated {done}/{n_local} elapsed={time.time() - t0:.0f}s")

    native_t = torch.cat(natives, dim=0)
    z_t = torch.cat(zs, dim=0)
    margin_t = torch.cat(margins, dim=0)
    zsbd_t = {k: torch.cat(v, dim=0) for k, v in zsbd.items()}
    gap_t = {k: torch.cat(v, dim=0) for k, v in gaps.items()}
    types = classify_types(
        native_t,
        tokenizer,
        pad_id=layout.pad_token_id,
        eos_id=layout.eos_token_id,
        bos_id=layout.bos_token_id,
    )
    valid = types["valid"]
    agree = {k: agreement(native_t, zsbd_t[k], valid) for k in ZSBD_VARIANTS}
    tail = {k: frac_wrong(native_t, zsbd_t[k], valid) for k in ZSBD_VARIANTS}
    n_valid = float(valid.float().sum().item())
    type_frac = {
        "digit": float((types["digit"] & valid).float().sum().item()) / max(n_valid, 1.0),
        "subword": float((types["subword"] & valid).float().sum().item()) / max(n_valid, 1.0),
        "rare": float((types["rare"] & valid).float().sum().item()) / max(n_valid, 1.0),
    }
    best_var = max(agree, key=lambda k: agree[k])
    payload = {
        "z": z_t,
        "native": native_t,
        "margin": margin_t,
        "valid": valid,
        "digit": types["digit"],
        "subword": types["subword"],
        "rare": types["rare"],
        **{f"zsbd_{k}": zsbd_t[k] for k in ZSBD_VARIANTS},
        **{f"gap_{k}": gap_t[k] for k in ZSBD_VARIANTS},
    }
    shard_path = out_dir / f"shard{args.shard}.pt"
    torch.save(payload, shard_path)
    summary = {
        "shard": args.shard,
        "n": int(native_t.shape[0]),
        "step": int(step),
        "run": args.run,
        "best_zsbd": best_var,
        "agreement": agree,
        "tail_frac": tail,
        "type_frac": type_frac,
        "elapsed_sec": time.time() - t0,
        "shard_path": str(shard_path),
    }
    (out_dir / f"shard{args.shard}_m0.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(
        f"M0 shard{args.shard} agree={agree} tail={tail} types={type_frac} "
        f"best={best_var}"
    )
    del model, ema_state, emb_norm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "summary": summary,
        "payload": payload,
        "tokenizer": tokenizer,
        "device": device,
        "dtype": dtype,
        "best_var": best_var,
    }


def run_m1_shard(
    *,
    args: argparse.Namespace,
    m0: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    payload = m0["payload"]
    tokenizer = m0["tokenizer"]
    device = m0["device"]
    native = payload["native"]
    best = m0["best_var"]
    zsbd = payload[f"zsbd_{best}"]
    margin = payload["margin"].float()
    types = {
        "valid": payload["valid"],
        "digit": payload["digit"],
        "subword": payload["subword"],
        "rare": payload["rare"],
    }
    valid = types["valid"]
    rng = torch.Generator()
    rng.manual_seed(int(args.seed) + 17 + int(args.shard) * 1009)

    columns: dict[str, torch.Tensor] = {
        "native": native,
        "zsbd": zsbd,
    }
    gate_frac: dict[str, float] = {}
    for rho in RHO_GRID:
        gates = gate_masks(
            native=native, zsbd=zsbd, margin=margin, types=types, rho=rho, rng=rng,
        )
        for name, g in gates.items():
            key = f"{name}@{rho:.2f}"
            columns[key] = hybrid_tokens(native, zsbd, g)
            den = float(valid.float().sum().item())
            gate_frac[key] = float((g & valid).float().sum().item()) / max(den, 1.0)

    agree_vs_native = {
        name: agreement(native, tok, valid) for name, tok in columns.items()
    }

    _log(f"M1 shard{args.shard} load gpt2-large for PPL")
    from models import get_hf_model

    gpt2 = get_hf_model("gpt2-large", torch_dtype=torch.bfloat16, device=str(device))
    gpt2.eval()
    for p in gpt2.parameters():
        p.requires_grad_(False)

    ppl: dict[str, float] = {}
    for name, tok in columns.items():
        texts = decode_texts(tok, tokenizer)
        _per, _ent, corpus = score_texts(
            texts,
            gpt2_model=gpt2,
            max_length=args.num_tokens,
            device=device,
            amp_dtype=torch.bfloat16,
        )
        ppl[name] = float(corpus)
        _log(f"M1 shard{args.shard} {name} ppl={corpus:.4f} agree={agree_vs_native[name]:.4f}")

    del gpt2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = {
        "shard": args.shard,
        "n": int(native.shape[0]),
        "best_zsbd": best,
        "agreement_vs_native": agree_vs_native,
        "gen_ppl": ppl,
        "gate_frac": gate_frac,
    }
    (out_dir / f"shard{args.shard}_m1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def weighted_mean(items: list[tuple[int, float]]) -> float:
    num = sum(n * v for n, v in items)
    den = sum(n for n, _v in items)
    if den <= 0:
        return float("nan")
    return num / den


def merge_m0(out_dir: Path, shards: int) -> dict[str, Any]:
    rows = []
    for i in range(shards):
        path = out_dir / f"shard{i}_m0.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    n_tot = sum(r["n"] for r in rows)
    agree = {}
    tail = {}
    types = {}
    for key in ZSBD_VARIANTS:
        agree[key] = weighted_mean([(r["n"], r["agreement"][key]) for r in rows])
        tail[key] = weighted_mean([(r["n"], r["tail_frac"][key]) for r in rows])
    for key in ("digit", "subword", "rare"):
        types[key] = weighted_mean([(r["n"], r["type_frac"][key]) for r in rows])
    best = max(agree, key=lambda k: agree[k])
    best_agree = agree[best]
    tail_best = tail[best]
    pass_agree = (0.91 <= best_agree <= 0.98)
    pass_tail = (0.03 <= tail_best <= 0.12)
    has_types = all(types[k] > 0 for k in ("digit", "subword", "rare"))
    passed = bool(pass_agree and pass_tail and has_types)
    merged = {
        "n": n_tot,
        "shards": shards,
        "best_zsbd": best,
        "agreement": agree,
        "tail_frac": tail,
        "type_frac": types,
        "pass": {
            "agreement_93_96_pm2": pass_agree,
            "tail_0.05_0.10_pm2": pass_tail,
            "type_hist": has_types,
            "m0": passed,
        },
        "shards_detail": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(f"M0 merge n={n_tot} agree={agree} tail={tail} types={types} pass={passed}")
    return merged


def merge_m1(out_dir: Path, shards: int) -> dict[str, Any]:
    rows = []
    for i in range(shards):
        path = out_dir / f"shard{i}_m1.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    n_tot = sum(r["n"] for r in rows)
    keys = list(rows[0]["agreement_vs_native"].keys())
    agree = {
        k: weighted_mean([(r["n"], r["agreement_vs_native"][k]) for r in rows])
        for k in keys
    }
    ppl = {
        k: weighted_mean([(r["n"], r["gen_ppl"][k]) for r in rows])
        for k in keys
    }
    # C2：同 ρ 下结构门 agreement 高于随机门、PPL 更接近 native
    c2 = {}
    native_ppl = ppl["native"]
    for rho in RHO_GRID:
        s_key = f"struct@{rho:.2f}"
        r_key = f"rand@{rho:.2f}"
        t_key = f"type@{rho:.2f}"
        s_ag, r_ag = agree[s_key], agree[r_key]
        s_ppl, r_ppl = ppl[s_key], ppl[r_key]
        struct_better_ag = s_ag > r_ag + 0.005
        struct_closer_ppl = abs(s_ppl - native_ppl) + 0.05 < abs(r_ppl - native_ppl)
        rand_near_struct = (s_ag - r_ag) < 0.005 and abs(s_ppl - r_ppl) < 0.3
        c2[f"{rho:.2f}"] = {
            "struct_agree": s_ag,
            "rand_agree": r_ag,
            "type_agree": agree[t_key],
            "struct_ppl": s_ppl,
            "rand_ppl": r_ppl,
            "type_ppl": ppl[t_key],
            "struct_better": bool(struct_better_ag and struct_closer_ppl),
            "rand_near_struct": bool(rand_near_struct),
        }
    hard = all(v["struct_better"] for v in c2.values())
    degraded = any(v["rand_near_struct"] for v in c2.values()) and not hard
    merged = {
        "n": n_tot,
        "agreement_vs_native": agree,
        "gen_ppl": ppl,
        "c2_by_rho": c2,
        "c2_hard": hard,
        "c2_degraded": degraded,
        "shards_detail": rows,
    }
    m1_dir = out_dir.parent / "m1"
    m1_dir.mkdir(parents=True, exist_ok=True)
    (m1_dir / "summary.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(f"M1 merge n={n_tot} c2_hard={hard} c2_degraded={degraded} ppl={ppl}")
    return merged


def cuda_ms(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def _mean_std(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "std": float("nan")}
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
    return {"mean": m, "std": var ** 0.5}


def run_m2_shard(
    *,
    args: argparse.Namespace,
    n_local: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    """墙钟拆解：去噪 / DiT decode / unembed / ZSBD / BGEE。"""
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_checkpoint(checkpoint=None, run=args.run)
    _log(f"M2 load {ckpt} n={n_local} seed={seed} device={device}")
    model, model_meta, step, train_cfg = load_model_from_checkpoint(ckpt, device)
    dtype = resolve_dtype(device, train_cfg)
    if model_meta["name"] == "elf":
        from models.lm.elf.ace import attach_ace_identity

        attach_ace_identity(
            model,
            model_hash=Path(args.run).name,
            step=step,
            tokenizer=model_meta["config"].get("tokenizer") or "t5-small",
        )
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ema_raw = ck.get("ema")
    ema_state = None
    if isinstance(ema_raw, dict) and ema_raw:
        ema_state = {k: v.to(device=device) for k, v in ema_raw.items()}
    del ck
    sampling_cfg = get_generate(
        model_meta["name"], args.generate, overrides=None,
    ).to_sampling_cfg()
    bb = getattr(model, "backbone", model)
    emb_norm = load_t5_emb(device, dtype)
    repeats = max(int(getattr(args, "repeats", 5)), 2)
    warmup = 1
    bs = min(args.micro_bs, n_local)
    keys = (
        "e2e_native", "denoise", "decode_g", "decode_no_g",
        "unembed", "zsbd", "e2e_zsbd", "e2e_bgee",
    )
    buckets: dict[str, list[float]] = {k: [] for k in keys}

    def gen(**kw):
        return elf_generate_latent(
            bb,
            num_samples=bs,
            seqlen=args.num_tokens,
            sampling_cfg=sampling_cfg,
            **kw,
        )

    with swap_ema_weights(model, ema_state):
        model.eval()
        set_seed(seed)
        _log(f"M2 warmup bs={bs} repeats={repeats}")
        with torch.no_grad(), torch.amp.autocast(
            device.type, dtype=dtype, enabled=device.type == "cuda",
        ):
            for w in range(warmup):
                set_seed(seed + w)
                gen()
            for r in range(repeats):
                set_seed(seed + 1000 + r)
                (_tok, _nfe, z), ms = cuda_ms(lambda: gen())
                buckets["e2e_native"].append(ms / bs)

                set_seed(seed + 2000 + r)
                _, ms = cuda_ms(lambda: gen(skip_decode=True))
                buckets["denoise"].append(ms / bs)

                sc = sc_cfg_tensor(bb, sampling_cfg, bs, device, z.dtype)
                (logits, hidden), ms = cuda_ms(
                    lambda: elf_decode_probe(bb, z, sc)
                )
                buckets["decode_g"].append(ms / bs)

                t = torch.ones(bs, device=device, dtype=z.dtype)
                model_in = (
                    torch.cat([z, torch.zeros_like(z)], dim=-1)
                    if bb.self_cond_prob > 0 else z
                )

                def _dit_no_g():
                    return bb.net_forward(
                        model_in, t, decoder_step_active=False,
                        deterministic=True, self_cond_cfg_scale=sc,
                    )

                _, ms = cuda_ms(_dit_no_g)
                buckets["decode_no_g"].append(ms / bs)

                def _unembed():
                    lg = hidden @ bb.unembed_kernel + bb.unembed_bias
                    return lg.argmax(dim=-1)

                _, ms = cuda_ms(_unembed)
                buckets["unembed"].append(ms / bs)

                _, ms = cuda_ms(lambda: zsbd_topk(z, emb_norm))
                buckets["zsbd"].append(ms / bs)

                set_seed(seed + 3000 + r)

                def _e2e_zsbd():
                    _t, _n, zz = gen(skip_decode=True)
                    zsbd_topk(zz, emb_norm)
                    return zz

                _, ms = cuda_ms(_e2e_zsbd)
                buckets["e2e_zsbd"].append(ms / bs)

                set_seed(seed + 4000 + r)
                _, ms = cuda_ms(lambda: gen(skip_decode=True, bgee=True))
                buckets["e2e_bgee"].append(ms / bs)
                _log(
                    f"M2 r={r+1}/{repeats} e2e={buckets['e2e_native'][-1]:.2f} "
                    f"denoise={buckets['denoise'][-1]:.2f} "
                    f"g={buckets['decode_g'][-1]:.2f} "
                    f"unemb={buckets['unembed'][-1]:.2f} "
                    f"zsbd={buckets['zsbd'][-1]:.2f} ms/sample"
                )

    stats = {k: _mean_std(v) for k, v in buckets.items()}
    e2e = stats["e2e_native"]["mean"]
    den = stats["denoise"]["mean"]
    unemb = stats["unembed"]["mean"]
    zsbd_t = stats["zsbd"]["mean"]
    skip_g = stats["e2e_zsbd"]["mean"]
    unembed_share = unemb / e2e if e2e > 0 else float("nan")
    speedup_skip_g = e2e / skip_g if skip_g > 0 else float("nan")
    denoise_share = den / e2e if e2e > 0 else float("nan")
    # C1 前置：即便门免费，端到端也要能到 ~1.3×；否则 unembed 不是瓶颈
    bottleneck = bool(speedup_skip_g >= 1.25 or unembed_share >= 0.15)
    summary = {
        "shard": args.shard,
        "n_per_repeat": bs,
        "repeats": repeats,
        "ms_per_sample": stats,
        "unembed_share": unembed_share,
        "denoise_share": denoise_share,
        "speedup_if_skip_native_g": speedup_skip_g,
        "unembed_is_e2e_bottleneck": bottleneck,
        "step": int(step),
        "run": args.run,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"shard{args.shard}_m2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(
        f"M2 shard{args.shard} e2e={e2e:.3f} denoise_share={denoise_share:.3f} "
        f"unembed_share={unembed_share:.4f} skip_g_speedup={speedup_skip_g:.3f} "
        f"bottleneck={bottleneck}"
    )
    del model, ema_state, emb_norm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def merge_m2(out_dir: Path, shards: int) -> dict[str, Any]:
    rows = []
    for i in range(shards):
        rows.append(json.loads((out_dir / f"shard{i}_m2.json").read_text(encoding="utf-8")))
    keys = rows[0]["ms_per_sample"].keys()
    ms = {
        k: _mean_std([r["ms_per_sample"][k]["mean"] for r in rows])
        for k in keys
    }
    e2e = ms["e2e_native"]["mean"]
    skip = ms["e2e_zsbd"]["mean"]
    unemb = ms["unembed"]["mean"]
    den = ms["denoise"]["mean"]
    speedup = e2e / skip if skip > 0 else float("nan")
    unembed_share = unemb / e2e if e2e > 0 else float("nan")
    denoise_share = den / e2e if e2e > 0 else float("nan")
    bottleneck = bool(speedup >= 1.25 or unembed_share >= 0.15)
    merged = {
        "shards": shards,
        "ms_per_sample": ms,
        "unembed_share": unembed_share,
        "denoise_share": denoise_share,
        "speedup_if_skip_native_g": speedup,
        "unembed_is_e2e_bottleneck": bottleneck,
        "c1_prefail": (not bottleneck),
        "verdict": (
            "unembed(+softmax) 是端到端可测瓶颈"
            if bottleneck
            else "unembed(+softmax) 不是端到端瓶颈（去噪 NFE 占主导）"
        ),
        "shards_detail": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(f"M2 merge {merged['verdict']} speedup={speedup:.3f} denoise_share={denoise_share:.3f}")
    return merged


def spawn_workers(args: argparse.Namespace, gpus: list[int]) -> int:
    counts = shard_counts(args.num_samples, len(gpus))
    procs: list[subprocess.Popen] = []
    script = Path(__file__).resolve()
    for i, (gid, n_loc) in enumerate(zip(gpus, counts)):
        if n_loc <= 0:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        cmd = [
            sys.executable,
            str(script),
            "--run", args.run,
            "--stage", args.stage,
            "--num-samples", str(n_loc),
            "--num-tokens", str(args.num_tokens),
            "--micro-bs", str(args.micro_bs),
            "--seed", str(args.seed + i * 1_000_003),
            "--generate", args.generate,
            "--repeats", str(getattr(args, "repeats", 5)),
            "--out", args.out,
            "--shard", str(i),
            "--shards", str(len(gpus)),
            "--worker",
        ]
        _log(f"spawn shard={i} gpu={gid} n={n_loc} seed={args.seed + i * 1_000_003}")
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(ROOT)))
    rc = 0
    for p in procs:
        c = p.wait()
        if c != 0:
            rc = c
    return rc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="I-1 hybrid readout M0/M1 probe")
    p.add_argument("--run", default="full/elf/official-owt-b")
    p.add_argument("--stage", default="m0m1", choices=("m0", "m1", "m0m1", "m2"))
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--num-tokens", type=int, default=1024)
    p.add_argument("--micro-bs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generate", default="eval")
    p.add_argument("--out", default="temp/auto-research/hybrid-i1")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--shards", type=int, default=None)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--worker", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    m0_dir = out / "m0"
    m2_dir = out / "m2"
    gpus = visible_gpu_ids()
    if args.stage == "m2":
        m2_dir.mkdir(parents=True, exist_ok=True)
    else:
        m0_dir.mkdir(parents=True, exist_ok=True)

    if not args.worker and args.shard is None and len(gpus) > 1:
        _log(f"parent spawn on gpus={gpus} stage={args.stage}")
        rc = spawn_workers(args, gpus)
        if rc != 0:
            _log(f"worker failed rc={rc}")
            return rc
        if args.stage == "m2":
            merged = merge_m2(m2_dir, len(gpus))
            (out / "STATUS.md").write_text(
                f"# hybrid-i1 status {_now()}\n\n"
                f"M2: {merged['verdict']}\n"
                f"speedup_if_skip_g={merged['speedup_if_skip_native_g']:.3f}\n"
                f"denoise_share={merged['denoise_share']:.3f}\n"
                f"c1_prefail={merged['c1_prefail']}\n",
                encoding="utf-8",
            )
            return 0
        m0 = merge_m0(m0_dir, len(gpus))
        if args.stage in ("m1", "m0m1"):
            merge_m1(m0_dir, len(gpus))
        (out / "STATUS.md").write_text(
            f"# hybrid-i1 status {_now()}\n\n"
            f"M0 pass: {m0['pass']['m0']}\n"
            f"best ZSBD: {m0['best_zsbd']} agree={m0['agreement'][m0['best_zsbd']]:.4f}\n"
            f"tail={m0['tail_frac'][m0['best_zsbd']]:.4f}\n",
            encoding="utf-8",
        )
        return 0 if m0["pass"]["m0"] else 2

    if args.shard is None:
        args.shard = 0
        args.shards = 1
    if args.stage == "m2":
        run_m2_shard(
            args=args, n_local=args.num_samples, seed=args.seed, out_dir=m2_dir,
        )
        if not args.worker:
            merged = merge_m2(m2_dir, 1)
            (out / "STATUS.md").write_text(
                f"# hybrid-i1 status {_now()}\n\n"
                f"M2: {merged['verdict']}\n"
                f"c1_prefail={merged['c1_prefail']}\n",
                encoding="utf-8",
            )
        return 0
    m0 = run_m0_shard(
        args=args,
        n_local=args.num_samples,
        seed=args.seed,
        out_dir=m0_dir,
    )
    if args.stage in ("m1", "m0m1"):
        run_m1_shard(args=args, m0=m0, out_dir=m0_dir)
    if not args.worker:
        merged = merge_m0(m0_dir, 1)
        if args.stage in ("m1", "m0m1"):
            merge_m1(m0_dir, 1)
        (out / "STATUS.md").write_text(
            f"# hybrid-i1 status {_now()}\n\n"
            f"M0 pass: {merged['pass']['m0']}\n"
            f"best ZSBD: {merged['best_zsbd']} "
            f"agree={merged['agreement'][merged['best_zsbd']]:.4f}\n"
            f"tail={merged['tail_frac'][merged['best_zsbd']]:.4f}\n",
            encoding="utf-8",
        )
        return 0 if merged["pass"]["m0"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
