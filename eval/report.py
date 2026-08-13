"""TriFluency 人类可读报告：参数说明 + 等级对照 + 逐样本 / 汇总。"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

# ``{step}/results.csv`` 列顺序：name 后接语料级指标
CSV_METRIC_KEYS: tuple[str, ...] = (
    "accept_at_human",
    "median_rep",
    "nonword_word_pct",
    "nonword_sample_pct",
    "clean_ppl",
    "clean_ppl_status",
    "n_accept",
    "cola_g",
    "raw_gen_ppl",
    "mean_entropy",
    "n",
    "nonempty_frac",
)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
RESULTS_CSV_NAME = "results.csv"
RESULTS_CHART_NAME = "results.png"
RESULTS_TABLE_NAME = "results_table.png"

# 人类可读表 / 图中优先展示的主指标（CSV 仍写全量列）
_DISPLAY_METRICS: tuple[tuple[str, str], ...] = (
    ("accept_at_human", "accept@human"),
    ("median_rep", "median_rep"),
    ("nonword_word_pct", "nonword%"),
    ("clean_ppl", "clean_ppl"),
    ("cola_g", "cola_g"),
    ("raw_gen_ppl", "raw_gen_ppl"),
    ("mean_entropy", "entropy"),
    ("n_accept", "n_accept"),
)

# 这些列按整数写出，不做四位小数
_INT_METRIC_KEYS = frozenset({"n", "n_accept"})
# 这些列保持原文字符串
_STR_METRIC_KEYS = frozenset({"clean_ppl_status"})


def _fmt4(val: Any) -> str:
    """浮点保留四位小数；非有限 → nan；其余原样。"""
    if val is None or val == "":
        return ""
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, int) and not isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        if not math.isfinite(val):
            return "nan"
        return f"{val:.4f}"
    if isinstance(val, str):
        # 已是格式化字符串或 status
        try:
            f = float(val)
        except ValueError:
            return val
        if not math.isfinite(f):
            return "nan"
        return f"{f:.4f}"
    return str(val)


def _as_float(val: Any) -> float:
    if val is None or val == "":
        return float("nan")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return float("nan")
    return float("nan")


def _csv_cell(key: str, val: Any) -> Any:
    if key == "name":
        return val
    if key in _STR_METRIC_KEYS:
        return "" if val is None else str(val)
    if key in _INT_METRIC_KEYS:
        if val is None or val == "":
            return ""
        try:
            return int(val)
        except (TypeError, ValueError):
            return val
    return _fmt4(val)


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


def validate_run_name(name: str) -> str:
    """校验评测名（不进 generate-hash；用于 CSV 首列）。"""
    raw = str(name).strip()
    if not raw:
        raise ValueError("run name must be non-empty")
    if not _NAME_RE.fullmatch(raw):
        raise ValueError(
            f"invalid run name {name!r}; use [A-Za-z0-9._+-], "
            "starting with alphanumeric"
        )
    return raw


def _fmt_name_token(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if math.isfinite(v) and v == int(v):
            return str(int(v))
        return str(v)
    return str(v)


def suggest_run_name(
    overrides: Mapping[str, Any] | None,
    *,
    sampling: Mapping[str, Any] | None = None,
) -> str:
    """从 generate overrides（必要时补 sampling）推导可读名。"""
    ov = dict(overrides or {})
    samp = dict(sampling or {})
    parts: list[str] = []

    ace = ov.get("ace", None)
    if ace is True or ace == 1:
        parts.append("ace")

    sc = ov.get("self_cond_cfg_scale", None)
    if sc is None:
        sc = samp.get("self_cond_cfg_scale")
    if sc is not None:
        parts.append(f"sc{_fmt_name_token(sc)}")

    if ov.get("dma") is False or ov.get("dma") == 0:
        parts.append("nodma")

    order = ov.get("dma_ace_order")
    if order is not None:
        parts.append(f"dma-{_fmt_name_token(order)}")

    known = {"ace", "self_cond_cfg_scale", "dma", "dma_ace_order"}
    for key in sorted(ov.keys(), key=str):
        if key in known or str(key).startswith("_"):
            continue
        parts.append(f"{key}{_fmt_name_token(ov[key])}")

    if not parts:
        return "default"
    return validate_run_name("-".join(parts))


def rewrite_step_results_csv(step_dir: Path | str) -> Path | None:
    """扫描 ``{step}/{generate-hash}/``，写出 ``results.csv`` + 图表。

    同时生成：
    - ``results.csv``：全量指标，浮点四位小数
    - ``results.png``：主指标柱状图
    - ``results_table.png``：给人看的汇总表（四位小数）

    仅收录同时具备 ``summary.json`` 与 fingerprint ``name`` 的子目录。
    同名多条时保留 generate-hash 字典序最后一条。
    无有效行则删除旧产物并返回 None。
    """
    step_dir = Path(step_dir)
    if not step_dir.is_dir():
        return None

    by_name: dict[str, dict[str, Any]] = {}
    for child in sorted(step_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        fp_path = child / "fingerprint.json"
        if not summary_path.is_file() or not fp_path.is_file():
            continue
        try:
            fp = json.loads(fp_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(fp, dict) or not isinstance(summary, dict):
            continue
        name_raw = fp.get("name")
        if not name_raw:
            continue
        try:
            name = validate_run_name(str(name_raw))
        except ValueError:
            continue
        row: dict[str, Any] = {"name": name}
        for key in CSV_METRIC_KEYS:
            val = summary.get(key, "")
            if isinstance(val, float) and not math.isfinite(val):
                val = float("nan")
            row[key] = val
        by_name[name] = row

    out_path = step_dir / RESULTS_CSV_NAME
    chart_path = step_dir / RESULTS_CHART_NAME
    table_path = step_dir / RESULTS_TABLE_NAME
    if not by_name:
        for p in (out_path, chart_path, table_path):
            if p.is_file():
                p.unlink()
        return None

    fieldnames = ["name", *CSV_METRIC_KEYS]
    rows = [by_name[k] for k in sorted(by_name.keys())]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(k, row.get(k, "")) for k in fieldnames})

    _write_results_chart(chart_path, rows, title=str(step_dir))
    _write_results_table_fig(table_path, rows, title=str(step_dir))
    return out_path


def _write_results_chart(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
) -> None:
    """主指标柱状图（按 name）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(r["name"]) for r in rows]
    plot_keys = [
        ("accept_at_human", "accept@human ↑"),
        ("median_rep", "median_rep ↓"),
        ("nonword_word_pct", "nonword% ↓"),
        ("clean_ppl", "clean_ppl ↓"),
        ("cola_g", "cola_g ↑"),
    ]
    n_panel = len(plot_keys)
    fig_h = max(2.4 * n_panel, 6.0)
    fig, axes = plt.subplots(
        n_panel, 1, figsize=(max(8.0, 0.55 * len(names) + 3.0), fig_h), sharex=True,
    )
    if n_panel == 1:
        axes = [axes]
    x = list(range(len(names)))
    for ax, (key, ylabel) in zip(axes, plot_keys):
        vals = [_as_float(r.get(key)) for r in rows]
        colors = ["#4C72B0" if math.isfinite(v) else "#CCCCCC" for v in vals]
        plot_vals = [0.0 if not math.isfinite(v) else v for v in vals]
        bars = ax.bar(x, plot_vals, color=colors, width=0.72)
        for bar, v in zip(bars, vals):
            if not math.isfinite(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        finite = [v for v in vals if math.isfinite(v)]
        if finite:
            lo, hi = min(finite), max(finite)
            pad = (hi - lo) * 0.15 if hi > lo else (abs(hi) * 0.1 + 0.05)
            ax.set_ylim(max(0.0, lo - pad) if lo >= 0 else lo - pad, hi + pad)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    short = title if len(title) <= 80 else "…" + title[-79:]
    fig.suptitle(f"TriFluency  {short}", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_results_table_fig(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
) -> None:
    """给人阅读的汇总表图（数值四位小数）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers = ["name", *[lab for _, lab in _DISPLAY_METRICS]]
    cell_rows: list[list[str]] = []
    for r in rows:
        line = [str(r["name"])]
        for key, _lab in _DISPLAY_METRICS:
            if key in _INT_METRIC_KEYS:
                line.append(_csv_cell(key, r.get(key, "")))
            else:
                line.append(_fmt4(r.get(key, "")))
        cell_rows.append(line)

    n_row = len(cell_rows)
    n_col = len(headers)
    fig_w = max(10.0, 1.15 * n_col + 2.0)
    fig_h = max(1.8, 0.42 * n_row + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    short = title if len(title) <= 90 else "…" + title[-89:]
    ax.set_title(f"TriFluency results  {short}", fontsize=11, pad=12)

    table = ax.table(
        cellText=cell_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (r_i, c_i), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r_i == 0:
            cell.set_facecolor("#EEF2F7")
            cell.set_text_props(weight="bold")
        elif r_i % 2 == 0:
            cell.set_facecolor("#FAFAFA")
        if c_i == 0:
            cell.set_text_props(ha="left")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
