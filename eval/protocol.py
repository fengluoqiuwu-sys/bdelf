"""TriFluency 三轴汇总：A/B/C′ + Gen.PPL/entropy。"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from eval.cola_score import COLA_MODEL_ID, load_cola, score_cola
from eval.gen_ppl import score_texts
from eval.nonword import score_nonwords
from eval.repetition import TAU_H, score_repetition

N_MIN_ACCEPT = 50
GEN_EVAL_MODEL = "gpt2-large"

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass
class TrifluencyScorers:
    """一次加载、可复用于多组 generate 的打分器（gpt2 + CoLA）。"""

    device: torch.device
    amp_dtype: torch.dtype
    gen_eval_model: str
    gpt2: Any | None
    cola_model: Any | None
    cola_tok: Any | None
    cola_id: str
    skip_gen_ppl: bool = False
    skip_cola: bool = False

    def close(self) -> None:
        self.gpt2 = None
        self.cola_model = None
        self.cola_tok = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def load_trifluency_scorers(
    *,
    device: torch.device,
    gen_eval_model: str = GEN_EVAL_MODEL,
    gen_eval_dtype: str = "bf16",
    skip_cola: bool = False,
    skip_gen_ppl: bool = False,
) -> TrifluencyScorers:
    """加载 gpt2 / CoLA；多组评测时只调一次。"""
    amp_dtype = _DTYPE_MAP[gen_eval_dtype]
    gpt2 = None
    if not skip_gen_ppl:
        import hf_config  # noqa: F401
        from models import get_hf_model

        print(f"[trifluency] loading {gen_eval_model} on {device}", flush=True)
        gpt2 = get_hf_model(
            gen_eval_model, torch_dtype=amp_dtype, device=str(device),
        )
        gpt2.eval()
        for p in gpt2.parameters():
            p.requires_grad_(False)

    cola_model = cola_tok = None
    cola_id = COLA_MODEL_ID
    if not skip_cola:
        print(f"[trifluency] loading CoLA {COLA_MODEL_ID}", flush=True)
        cola_model, cola_tok, cola_id = load_cola(
            device=device, torch_dtype=amp_dtype,
        )

    return TrifluencyScorers(
        device=device,
        amp_dtype=amp_dtype,
        gen_eval_model=gen_eval_model,
        gpt2=gpt2,
        cola_model=cola_model,
        cola_tok=cola_tok,
        cola_id=cola_id,
        skip_gen_ppl=skip_gen_ppl,
        skip_cola=skip_cola,
    )


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
    scorers: TrifluencyScorers | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对 ``texts`` 跑完整协议；返回 (per_sample 行, summary)。

    传入 ``scorers`` 时复用已加载的 gpt2/CoLA（不卸载）；否则自行加载并在结束时释放。
    """
    own_scorers = scorers is None
    if device is None:
        device = (
            scorers.device
            if scorers is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    if scorers is None:
        scorers = load_trifluency_scorers(
            device=device,
            gen_eval_model=gen_eval_model,
            gen_eval_dtype=gen_eval_dtype,
            skip_cola=skip_cola,
            skip_gen_ppl=skip_gen_ppl,
        )
    elif skip_cola != scorers.skip_cola or skip_gen_ppl != scorers.skip_gen_ppl:
        raise ValueError(
            "scorers skip flags must match skip_cola/skip_gen_ppl "
            f"(scorers=({scorers.skip_cola},{scorers.skip_gen_ppl}) "
            f"vs args=({skip_cola},{skip_gen_ppl}))"
        )

    try:
        reps, accepts, rep_sum = score_repetition(texts, tau_h=tau_h)
        nw_fracs, nw_counts, nw_sum = score_nonwords(texts)

        amp_dtype = scorers.amp_dtype
        per_ppl: list[float]
        per_ent: list[float]
        corpus_ppl: float
        clean_texts = [t for t, ok in zip(texts, accepts) if ok]
        n_accept = len(clean_texts)

        if scorers.skip_gen_ppl:
            per_ppl = [float("nan")] * len(texts)
            per_ent = [float("nan")] * len(texts)
            corpus_ppl = float("nan")
            clean_ppl = float("nan")
            clean_status = "skipped"
        else:
            assert scorers.gpt2 is not None
            per_ppl, per_ent, corpus_ppl = score_texts(
                texts,
                gpt2_model=scorers.gpt2,
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
                    gpt2_model=scorers.gpt2,
                    max_length=max_length,
                    device=device,
                    amp_dtype=amp_dtype,
                )
                clean_status = "ok"

        cola_per: list[float]
        cola_sum: dict[str, float]
        cola_id = scorers.cola_id
        if scorers.skip_cola:
            cola_per = [float("nan")] * len(texts)
            cola_sum = {"cola_g": float("nan")}
        else:
            assert scorers.cola_model is not None and scorers.cola_tok is not None
            cola_per, cola_sum = score_cola(
                texts,
                model=scorers.cola_model,
                tokenizer=scorers.cola_tok,
                device=device,
            )

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
                "gen_eval_model": scorers.gen_eval_model,
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
    finally:
        if own_scorers:
            scorers.close()


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
    scorers: TrifluencyScorers | None = None,
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
        scorers=scorers,
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
