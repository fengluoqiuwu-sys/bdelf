"""TriFluency 三轴汇总：A/B/C′ + Gen.PPL/entropy。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from eval.cola_score import COLA_MODEL_ID, load_cola, score_cola
from eval.gen_ppl import score_texts
from eval.nonword import score_nonwords
from eval.repetition import TAU_H, score_repetition

N_MIN_ACCEPT = 50
GEN_EVAL_MODEL = "gpt2-large"


def _nanmean(xs: list[float]) -> float:
    vals = [x for x in xs if isinstance(x, float) and math.isfinite(x)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def compute_trifluency(
    texts: list[str],
    *,
    device: torch.device | None = None,
    max_length: int = 1024,
    tau_h: float = TAU_H,
    n_min_accept: int = N_MIN_ACCEPT,
    gen_eval_model: str = GEN_EVAL_MODEL,
    gen_eval_dtype: str = "bf16",
    skip_cola: bool = False,
    skip_gen_ppl: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对 ``texts`` 跑完整协议；返回 (per_sample 行, summary)。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reps, accepts, rep_sum = score_repetition(texts, tau_h=tau_h)
    nw_fracs, nw_counts, nw_sum = score_nonwords(texts)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    amp_dtype = dtype_map[gen_eval_dtype]

    per_ppl: list[float]
    per_ent: list[float]
    corpus_ppl: float
    clean_texts = [t for t, ok in zip(texts, accepts) if ok]
    n_accept = len(clean_texts)

    if skip_gen_ppl:
        per_ppl = [float("nan")] * len(texts)
        per_ent = [float("nan")] * len(texts)
        corpus_ppl = float("nan")
        clean_ppl = float("nan")
        clean_status = "skipped"
    else:
        import hf_config  # noqa: F401
        from models import get_hf_model

        print(f"[trifluency] loading {gen_eval_model} on {device}", flush=True)
        gpt2 = get_hf_model(
            gen_eval_model, torch_dtype=amp_dtype, device=str(device),
        )
        gpt2.eval()
        for p in gpt2.parameters():
            p.requires_grad_(False)
        per_ppl, per_ent, corpus_ppl = score_texts(
            texts,
            gpt2_model=gpt2,
            max_length=max_length,
            device=device,
            amp_dtype=amp_dtype,
        )
        if n_accept < n_min_accept:
            clean_ppl = float("nan")
            clean_status = "invalid"
        else:
            print(
                f"[trifluency] clean_ppl on {n_accept} accepted samples",
                flush=True,
            )
            _, _, clean_ppl = score_texts(
                clean_texts,
                gpt2_model=gpt2,
                max_length=max_length,
                device=device,
                amp_dtype=amp_dtype,
            )
            clean_status = "ok"
        del gpt2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cola_per: list[float]
    cola_sum: dict[str, float]
    cola_id = COLA_MODEL_ID
    if skip_cola:
        cola_per = [float("nan")] * len(texts)
        cola_sum = {"cola_g": float("nan")}
    else:
        print(f"[trifluency] loading CoLA {COLA_MODEL_ID}", flush=True)
        cola_model, cola_tok, cola_id = load_cola(device=device, torch_dtype=amp_dtype)
        cola_per, cola_sum = score_cola(
            texts, model=cola_model, tokenizer=cola_tok, device=device,
        )
        del cola_model, cola_tok
        if device.type == "cuda":
            torch.cuda.empty_cache()

    nonempty = sum(1 for t in texts if isinstance(t, str) and t.strip())
    summary: dict[str, Any] = {
        "n": len(texts),
        "accept_at_human": rep_sum["accept_at_human"],
        "median_rep": rep_sum["median_rep"],
        "nonword_word_pct": nw_sum["nonword_word_pct"],
        "nonword_sample_pct": nw_sum["nonword_sample_pct"],
        "clean_ppl": clean_ppl,
        "clean_ppl_status": clean_status,
        "n_accept": n_accept,
        "cola_g": cola_sum.get("cola_g", float("nan")),
        "raw_gen_ppl": corpus_ppl,
        "mean_entropy": _nanmean(per_ent),
        "nonempty_frac": float(nonempty) / float(max(len(texts), 1)),
        "meta": {
            "protocol": "trifluency-v1",
            "cola_model": cola_id,
            "gen_eval_model": gen_eval_model,
            "tau_h": tau_h,
            "n_min_accept": n_min_accept,
            "max_length": max_length,
        },
    }
    per_sample: list[dict[str, Any]] = []
    for i in range(len(texts)):
        per_sample.append(
            {
                "seq_rep_4": reps[i],
                "accept": bool(accepts[i]),
                "nonword_word_frac": nw_fracs[i],
                "nonword_count": nw_counts[i],
                "gen_ppl": per_ppl[i],
                "entropy": per_ent[i],
                "cola_g": cola_per[i],
            }
        )
    return per_sample, summary


def run_trifluency(
    texts: list[str],
    *,
    out_dir: Path | str,
    device: torch.device | None = None,
    max_length: int = 1024,
    tau_h: float = TAU_H,
    n_min_accept: int = N_MIN_ACCEPT,
    gen_eval_model: str = GEN_EVAL_MODEL,
    gen_eval_dtype: str = "bf16",
    skip_cola: bool = False,
    skip_gen_ppl: bool = False,
) -> dict[str, Any]:
    """对 ``texts`` 跑完整协议，写入 ``out_dir`` 的 csv/json，返回 summary。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample, summary = compute_trifluency(
        texts,
        device=device,
        max_length=max_length,
        tau_h=tau_h,
        n_min_accept=n_min_accept,
        gen_eval_model=gen_eval_model,
        gen_eval_dtype=gen_eval_dtype,
        skip_cola=skip_cola,
        skip_gen_ppl=skip_gen_ppl,
    )
    clean_status = str(summary.get("clean_ppl_status", ""))

    csv_path = out_dir / "per_sample.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "index",
                "seq_rep_4",
                "accept",
                "nonword_word_frac",
                "nonword_count",
                "gen_ppl",
                "entropy",
                "cola_g",
                "text_preview",
            ]
        )
        for i, (text, row) in enumerate(zip(texts, per_sample)):
            w.writerow(
                [
                    i + 1,
                    f"{row['seq_rep_4']:.6g}",
                    int(row["accept"]),
                    f"{row['nonword_word_frac']:.6g}",
                    row["nonword_count"],
                    _fmt(row["gen_ppl"]),
                    _fmt(row["entropy"]),
                    _fmt(row["cola_g"]),
                    re_preview(text),
                ]
            )

    sum_path = out_dir / "summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(summary), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

    print(f"[trifluency] wrote {csv_path}", flush=True)
    print(f"[trifluency] wrote {sum_path}", flush=True)
    print(
        f"[trifluency] accept={summary['accept_at_human']:.3f} "
        f"nonword%={summary['nonword_word_pct']:.2f} "
        f"clean_ppl={_fmt(summary['clean_ppl'])}({clean_status}) "
        f"cola_g={_fmt(summary['cola_g'])} "
        f"raw_ppl={_fmt(summary['raw_gen_ppl'])}",
        flush=True,
    )
    return summary


def _fmt(x: float) -> str:
    if isinstance(x, float) and math.isfinite(x):
        return f"{x:.6g}"
    return "nan"


def _sanitize(obj: Any) -> Any:
    """将 NaN/Inf 收成 None，保证 summary.json 合法。"""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def re_preview(text: str, n: int = 80) -> str:
    s = " ".join((text or "").split())
    if len(s) > n:
        return s[: n - 1] + "…"
    return s
