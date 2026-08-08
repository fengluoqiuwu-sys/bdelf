"""TriFluency 人类可读报告：参数说明 + 等级对照 + 逐样本 / 汇总。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence


METRIC_GUIDE = """\
# 指标说明（TriFluency-v1）
#
# A 重复（轨迹/反馈轴；与 ACE 同口径）
#   seq_rep_4     样本 4-gram 重复率 = 1 - |unique|/|total|（Welleck）
#   accept        1 当 seq_rep_4 < τ_H（默认 τ_H=0.0192，人类 XSum 95 分位）
#   accept@human  语料级：accept 比例（↑好）
#   median_rep    语料级：seq_rep_4 中位数（↓好）
#
# B 假词（解码轴；ACE App.C）
#   nonword_word_frac   本样本：len≥4 字母词中 wordfreq zipf=0 占比（↓好）
#   nonword_count       本样本假词个数
#   nonword_word_pct    语料级假词率 %（↓好）
#   nonword_sample_pct  语料级：含 ≥1 假词的样本比例 %（↓好）
#
# C′ 通顺（禁止用裸 Gen.PPL 排序）
#   gen_ppl / raw_gen_ppl  GPT-2 Large 因果 PPL；raw=全库（可被重复 hack）
#   clean_ppl              仅在 accept 样本上的 Gen.PPL（↓好；接受数不足则 invalid）
#   entropy                GPT-2 token unigram Shannon 熵（nat）；过低疑塌缩
#   cola_g                 开源 CoLA 可接受概率均值（↑好；易饱和，作护栏）
#
# 等级参考（无条件 OWT、L≈1024；S≈人类 / A≈ACE 修好 / B=半可用 / C=病态）
#   accept@human : S≥0.80  A∈[0.40,0.80)  B∈[0.15,0.40)  C<0.15
#   median_rep   : S≤0.010 A∈(0.010,0.025] B∈(0.025,0.05] C>0.05
#   nonword%     : S≤0.3   A∈(0.3,0.8]     B∈(0.8,1.5]    C>1.5
#   clean_ppl    : S≤16    A∈(16,24]       B∈(24,32]      C>32 或 invalid
#   cola_g       : S≥0.95  A∈[0.90,0.95)   B∈[0.80,0.90)  C<0.80
#   raw_gen_ppl  : 仅对照；<15 且 accept 很低 → 高度可疑（naive hack）
"""


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if not math.isfinite(x):
            return "nan"
        return f"{x:.6g}"
    if isinstance(x, bool):
        return "1" if x else "0"
    return str(x)


def _grade_accept(v: float) -> str:
    if not math.isfinite(v):
        return "?"
    if v >= 0.80:
        return "S"
    if v >= 0.40:
        return "A"
    if v >= 0.15:
        return "B"
    return "C"


def _grade_median_rep(v: float) -> str:
    if not math.isfinite(v):
        return "?"
    if v <= 0.010:
        return "S"
    if v <= 0.025:
        return "A"
    if v <= 0.05:
        return "B"
    return "C"


def _grade_nonword_pct(v: float) -> str:
    if not math.isfinite(v):
        return "?"
    if v <= 0.3:
        return "S"
    if v <= 0.8:
        return "A"
    if v <= 1.5:
        return "B"
    return "C"


def _grade_clean_ppl(v: float, status: str) -> str:
    if status != "ok" or not math.isfinite(v):
        return "C"
    if v <= 16:
        return "S"
    if v <= 24:
        return "A"
    if v <= 32:
        return "B"
    return "C"


def _grade_cola(v: float) -> str:
    if not math.isfinite(v):
        return "?"
    if v >= 0.95:
        return "S"
    if v >= 0.90:
        return "A"
    if v >= 0.80:
        return "B"
    return "C"


def write_run_header(
    f,
    *,
    params: Mapping[str, Any],
) -> None:
    """写参数块 + 指标说明。"""
    f.write("# ======== TriFluency 离线评测报告 ========\n")
    f.write("#\n")
    f.write("# --- 运行参数 ---\n")
    for key in sorted(params.keys(), key=str):
        val = params[key]
        if isinstance(val, (dict, list)):
            import json

            val_s = json.dumps(val, ensure_ascii=False, sort_keys=True)
        else:
            val_s = val
        f.write(f"# {key}={val_s}\n")
    f.write("#\n")
    f.write(METRIC_GUIDE)
    if not METRIC_GUIDE.endswith("\n"):
        f.write("\n")
    f.write("#\n")


def write_samples_report(
    path: Path | str,
    *,
    params: Mapping[str, Any],
    texts: Sequence[str],
    per_sample: Sequence[Mapping[str, Any]],
) -> None:
    """文件1：参数/指标说明 + 逐样本（多指标）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    with open(path, "w", encoding="utf-8") as f:
        write_run_header(f, params=params)
        f.write("# --- 逐样本 ---\n\n")
        for i, (text, row) in enumerate(zip(texts, per_sample), start=1):
            f.write("=" * 72 + "\n")
            f.write(
                f"### sample {i}/{n}"
                f"  gen_ppl={_fmt(row.get('gen_ppl'))}"
                f"  entropy={_fmt(row.get('entropy'))}"
                f"  seq_rep_4={_fmt(row.get('seq_rep_4'))}"
                f"  accept={_fmt(row.get('accept'))}"
                f"  nonword_frac={_fmt(row.get('nonword_word_frac'))}"
                f"  nonword_n={_fmt(row.get('nonword_count'))}"
                f"  cola_g={_fmt(row.get('cola_g'))}\n"
            )
            f.write("=" * 72 + "\n")
            f.write(text if isinstance(text, str) else "")
            if not (isinstance(text, str) and text.endswith("\n")):
                f.write("\n")
            f.write("\n")


def write_summary_report(
    path: Path | str,
    *,
    params: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    """文件2：参数/指标说明 + 语料级汇总与等级。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    acc = float(summary.get("accept_at_human", float("nan")))
    med = float(summary.get("median_rep", float("nan")))
    nw = float(summary.get("nonword_word_pct", float("nan")))
    clean = float(summary.get("clean_ppl", float("nan")))
    status = str(summary.get("clean_ppl_status", ""))
    cola = float(summary.get("cola_g", float("nan")))
    raw = float(summary.get("raw_gen_ppl", float("nan")))
    ent = float(summary.get("mean_entropy", float("nan")))

    with open(path, "w", encoding="utf-8") as f:
        write_run_header(f, params=params)
        f.write("# --- 语料汇总 ---\n")
        f.write(f"# n={summary.get('n')}\n")
        f.write(f"# nonempty_frac={_fmt(summary.get('nonempty_frac'))}\n")
        f.write(f"# n_accept={summary.get('n_accept')}\n")
        f.write("#\n")
        f.write("# 主表（含等级）\n")
        f.write(
            f"# accept@human={_fmt(acc)}  grade={_grade_accept(acc)}\n"
        )
        f.write(
            f"# median_rep={_fmt(med)}  grade={_grade_median_rep(med)}\n"
        )
        f.write(
            f"# nonword_word_pct={_fmt(nw)}  grade={_grade_nonword_pct(nw)}\n"
        )
        f.write(
            f"# nonword_sample_pct={_fmt(summary.get('nonword_sample_pct'))}\n"
        )
        f.write(
            f"# clean_ppl={_fmt(clean)}  status={status}  "
            f"grade={_grade_clean_ppl(clean, status)}\n"
        )
        f.write(f"# cola_g={_fmt(cola)}  grade={_grade_cola(cola)}\n")
        f.write("#\n")
        f.write("# 对照（不参与主排序）\n")
        f.write(f"# raw_gen_ppl={_fmt(raw)}\n")
        f.write(f"# mean_entropy={_fmt(ent)}\n")
        f.write("#\n")
        f.write("# 一行摘要（便于复制）\n")
        f.write(
            f"accept@human={_fmt(acc)}({_grade_accept(acc)}) "
            f"median_rep={_fmt(med)}({_grade_median_rep(med)}) "
            f"nonword%={_fmt(nw)}({_grade_nonword_pct(nw)}) "
            f"clean_ppl={_fmt(clean)}({_grade_clean_ppl(clean, status)}) "
            f"cola_g={_fmt(cola)}({_grade_cola(cola)}) "
            f"raw_gen_ppl={_fmt(raw)} entropy={_fmt(ent)}\n"
        )
