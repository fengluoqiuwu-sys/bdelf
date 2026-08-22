"""Training metrics CSV I/O, PPL plots, and log formatting."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

TRAIN_CSV_FIELDS = [
    "step",
    "tokens",
    "train_loss",
    "train_ppl",
    "loss_branch",
    "denoise_mse",
    "decode_ce",
    "late_ce",
    "lex_ce",
    "attr",
    "chart_ce",
    "lr",
    "tokens_per_sec",
]
EVAL_CSV_FIELDS = [
    "step",
    "tokens",
    "eval_loss",
    "eval_ppl",
    "gen_loss",
    "gen_ppl",
    "gen_uniq_mean",
    "gen_nonempty_frac",
    "lr",
]

# owt+elf train 子集抽样 32768 条、经 gpt2-large 打分的参考水平（写死）。
# 来源：temp/score_owt_train_ref.py（seed=42）；gen_ppl 对齐在线 gen-eval 口径。
REF_TRAIN_GEN_PPL = 16.9413
REF_TRAIN_GEN_UNIQ_MEAN = 435.6844

_TRAIN_LOG = "[train]"


def _train_log(msg: str, *, file: Any = None) -> None:
    if file is None:
        file = sys.stdout
    print(f"{_TRAIN_LOG} {msg}", file=file, flush=True)


def loss_to_ppl(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def append_csv_row(csv_path: Path, fields: list[str], row: dict[str, Any]) -> None:
    if csv_path.exists():
        ensure_csv_schema(csv_path, fields)
        write_header = False
    else:
        write_header = True
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def init_csv_header(csv_path: Path, fields: list[str]) -> None:
    """若尚无文件则只写表头，保证 plot/schema 迁移有落盘目标。

    train_log 每步都会 append；eval_log 要等到首次 eval。
    ``log_plot_step`` 常小于 ``eval_step``，若不预先建表头，
    首次 ``update_ppl_plots`` 会对尚不存在的 eval_log 调 ``ensure_csv_schema`` 崩溃。
    """
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def ensure_csv_schema(csv_path: Path, fields: list[str]) -> None:
    """Rewrite CSV if the on-disk header is missing newly added columns.

    调用方须保证文件已存在（新 run 用 ``init_csv_header``；旧 run 由 append 创建）。
    """
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        if old_fields == fields:
            return
        rows = list(reader)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def truncate_csv_for_resume(csv_path: Path, start_step: int) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        if not old_fields:
            return 0
        # Prefer the canonical schema so resume can introduce new columns.
        fieldnames = EVAL_CSV_FIELDS if csv_path.name == "eval_log.csv" else old_fields
        if csv_path.name == "train_log.csv":
            fieldnames = TRAIN_CSV_FIELDS
        rows_by_step: dict[int, dict[str, str]] = {}
        for row in reader:
            step = int(row["step"])
            if step < start_step:
                rows_by_step[step] = row
    rows = [rows_by_step[s] for s in sorted(rows_by_step)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return len(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    return float(raw)


def _row_tokens(
    row: dict[str, str],
    *,
    tokens_per_micro_step: int | None,
) -> int | None:
    """累计数据 token：优先读 CSV ``tokens``，否则由 step 回推。"""
    raw = _parse_float(row.get("tokens"))
    if raw is not None:
        return int(raw)
    if tokens_per_micro_step is None or tokens_per_micro_step < 1:
        return None
    step_raw = row.get("step")
    if step_raw is None or step_raw == "":
        return None
    return (int(step_raw) + 1) * tokens_per_micro_step


def _backfill_tokens_column(
    csv_path: Path,
    fields: list[str],
    *,
    tokens_per_micro_step: int,
) -> None:
    """为缺少 ``tokens`` 的旧行按 step 回填，便于 CSV 与曲线一致。"""
    if not csv_path.exists() or tokens_per_micro_step < 1:
        return
    rows = _read_csv_rows(csv_path)
    if not rows:
        return
    changed = False
    for row in rows:
        if row.get("tokens"):
            continue
        tok = _row_tokens(row, tokens_per_micro_step=tokens_per_micro_step)
        if tok is None:
            continue
        row["tokens"] = str(tok)
        changed = True
    if not changed:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _decode_ce_train_series(
    train_rows: list[dict[str, str]],
    *,
    tokens_per_micro_step: int | None,
) -> tuple[list[int], list[float], list[float]]:
    """Train decode-CE points for plotting (BDELF exclusive / ELF mixed)."""
    xs: list[int] = []
    ppls: list[float] = []
    lrs: list[float] = []
    for row in train_rows:
        branch = row.get("loss_branch") or ""
        if branch not in ("decode", "mixed"):
            continue
        # 旧日志：无 decode 样本时 ce 被记成 0 → ppl=1.0，绘图跳过。
        ce = _parse_float(row.get("decode_ce"))
        if branch == "mixed" and (ce is None or ce == 0.0):
            continue
        ppl = _parse_float(row.get("train_ppl"))
        if ppl is None:
            if ce is not None and ce > 0.0:
                ppl = loss_to_ppl(ce)
        if ppl is None:
            continue
        if branch == "mixed" and ppl == 1.0 and (ce is None or ce == 0.0):
            continue
        tok = _row_tokens(row, tokens_per_micro_step=tokens_per_micro_step)
        if tok is None:
            continue
        xs.append(tok)
        ppls.append(ppl)
        lrs.append(float(row["lr"]))
    return xs, ppls, lrs


def update_ppl_plots(
    train_csv: Path,
    eval_csv: Path,
    out_dir: Path,
    *,
    tokens_per_micro_step: int | None = None,
) -> None:
    # tokens 列迁移只作用于已落盘的 CSV；缺文件说明调用方未 init / 路径错误。
    if tokens_per_micro_step is not None and tokens_per_micro_step >= 1:
        ensure_csv_schema(train_csv, TRAIN_CSV_FIELDS)
        ensure_csv_schema(eval_csv, EVAL_CSV_FIELDS)
        _backfill_tokens_column(
            train_csv, TRAIN_CSV_FIELDS,
            tokens_per_micro_step=tokens_per_micro_step,
        )
        _backfill_tokens_column(
            eval_csv, EVAL_CSV_FIELDS,
            tokens_per_micro_step=tokens_per_micro_step,
        )

    train_rows = _read_csv_rows(train_csv)
    eval_rows = _read_csv_rows(eval_csv)
    if not train_rows:
        return

    dual_branch = any(
        r.get("loss_branch") in ("denoise", "decode", "mixed") for r in train_rows
    )
    if dual_branch:
        train_plot_x, train_ppl, _ = _decode_ce_train_series(
            train_rows, tokens_per_micro_step=tokens_per_micro_step,
        )
    else:
        train_plot_x = []
        train_ppl = []
        for r in train_rows:
            tok = _row_tokens(r, tokens_per_micro_step=tokens_per_micro_step)
            ppl = _parse_float(r.get("train_ppl"))
            if tok is None or ppl is None:
                continue
            train_plot_x.append(tok)
            train_ppl.append(ppl)

    eval_x: list[int] = []
    eval_ppl: list[float | None] = []
    gen_ppl: list[float | None] = []
    gen_uniq: list[float | None] = []
    for r in eval_rows:
        tok = _row_tokens(r, tokens_per_micro_step=tokens_per_micro_step)
        if tok is None:
            continue
        eval_x.append(tok)
        eval_ppl.append(_parse_float(r.get("eval_ppl") or r.get("gpt2_ppl")))
        gen_ppl.append(_parse_float(r.get("gen_ppl")))
        gen_uniq.append(_parse_float(r.get("gen_uniq_mean")))

    train_lr_x: list[int] = []
    train_lr: list[float] = []
    for r in train_rows:
        tok = _row_tokens(r, tokens_per_micro_step=tokens_per_micro_step)
        if tok is None:
            continue
        train_lr_x.append(tok)
        train_lr.append(float(r["lr"]))

    train_label = (
        "train decode ppl (exp ce)" if dual_branch else "train ppl (exp loss)"
    )
    eval_label = (
        "eval decode ppl (exp ce)" if dual_branch else "eval ppl (exp loss)"
    )

    def _filter_xy(
        xs: list[int],
        ys: list[float | None] | list[float],
        *,
        x_min: float | None = None,
        y_max: float | None = None,
    ) -> tuple[list[int], list[float]]:
        out_x: list[int] = []
        out_y: list[float] = []
        for x, y in zip(xs, ys):
            if y is None:
                continue
            if x_min is not None and x < x_min:
                continue
            if y_max is not None and y > y_max:
                continue
            out_x.append(x)
            out_y.append(float(y))
        return out_x, out_y

    def _draw_one(
        *,
        filename: str,
        title: str,
        x_min: float | None = None,
        x_max: float | None = None,
        ppl_cap: float | None = None,
        ppl_ylim: tuple[float, float] | None = None,
        auto_ppl_ylim_from: tuple[list[float], ...] | None = None,
    ) -> None:
        t_xs, t_ppls = _filter_xy(train_plot_x, train_ppl, x_min=x_min, y_max=ppl_cap)
        e_xs, e_ppls = _filter_xy(eval_x, eval_ppl, x_min=x_min, y_max=ppl_cap)
        g_xs, g_ppls = _filter_xy(eval_x, gen_ppl, x_min=x_min, y_max=ppl_cap)
        u_xs, u_vals = _filter_xy(eval_x, gen_uniq, x_min=x_min)
        lr_xs, lr_vals = _filter_xy(
            train_lr_x,
            [lr if lr > 0 else None for lr in train_lr],
            x_min=x_min,
        )

        if not t_xs and not e_xs and not g_xs and not u_xs:
            return

        ylim = ppl_ylim
        if ylim is None and auto_ppl_ylim_from is not None:
            peak = 0.0
            for series in auto_ppl_ylim_from:
                for v in series:
                    if v > peak:
                        peak = v
            if peak <= 0:
                return
            ylim = (0.0, peak * 1.05)

        fig, ax_ppl = plt.subplots(figsize=(11, 5.2))

        if t_xs:
            ax_ppl.plot(
                t_xs, t_ppls, color="#4C72B0", alpha=0.55, linewidth=1.2,
                label=train_label, zorder=1,
            )
        if e_xs:
            ax_ppl.plot(
                e_xs, e_ppls, color="#D62728", linewidth=2.8, marker="o",
                markersize=4, label=eval_label, zorder=5,
            )
        if g_xs:
            ax_ppl.plot(
                g_xs, g_ppls, color="#2CA02C", linewidth=2.4, marker="s",
                markersize=4, label="gen ppl (gpt2-large)", zorder=6,
            )
        # 训练语料参考：同色虚线（owt+elf 抽样写死）。
        ax_ppl.axhline(
            REF_TRAIN_GEN_PPL,
            color="#2CA02C",
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=f"train-data gen ppl ({REF_TRAIN_GEN_PPL:g})",
            zorder=3,
        )

        ax_lr = ax_ppl.twinx()
        if lr_xs:
            ax_lr.plot(
                lr_xs, lr_vals, color="#7F7F7F", linestyle="--",
                linewidth=1.0, alpha=0.9, label="lr", zorder=2,
            )
            ax_lr.set_ylabel("learning rate")
            ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        ax_uniq = ax_ppl.twinx()
        ax_uniq.spines["right"].set_position(("outward", 55))
        ax_uniq.set_ylim(0, 500)
        ax_uniq.set_ylabel("gen_uniq_mean")
        if u_xs:
            ax_uniq.plot(
                u_xs, u_vals, color="#9467BD", linewidth=2.0, marker="^",
                markersize=3.5, label="gen_uniq_mean", zorder=4,
            )
        ax_uniq.axhline(
            REF_TRAIN_GEN_UNIQ_MEAN,
            color="#9467BD",
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=f"train-data uniq ({REF_TRAIN_GEN_UNIQ_MEAN:g})",
            zorder=3,
        )

        if x_min is not None or x_max is not None:
            left = x_min if x_min is not None else ax_ppl.get_xlim()[0]
            right = x_max if x_max is not None else ax_ppl.get_xlim()[1]
            ax_ppl.set_xlim(left, right)
        if ylim is not None:
            ax_ppl.set_ylim(*ylim)

        ax_ppl.set_xlabel("tokens")
        ax_ppl.set_ylabel("perplexity")
        ax_ppl.set_title(title)
        ax_ppl.grid(True, alpha=0.25)
        ax_ppl.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        handles, labels = ax_ppl.get_legend_handles_labels()
        h2, l2 = ax_lr.get_legend_handles_labels()
        h3, l3 = ax_uniq.get_legend_handles_labels()
        # 图例按列填充；显式排序，让 lr 与 gen ppl 对调位置。
        by_label = {lab: h for h, lab in zip(handles + h2 + h3, labels + l2 + l3)}
        legend_order = [
            train_label,
            eval_label,
            "lr",
            f"train-data gen ppl ({REF_TRAIN_GEN_PPL:g})",
            "gen ppl (gpt2-large)",
            "gen_uniq_mean",
            f"train-data uniq ({REF_TRAIN_GEN_UNIQ_MEAN:g})",
        ]
        all_h = [by_label[lab] for lab in legend_order if lab in by_label]
        all_l = [lab for lab in legend_order if lab in by_label]
        # 图例放在坐标区下方横排，避免挡曲线；系列变多时自动换行。
        n_items = max(len(all_h), 1)
        ncol = min(3, n_items)
        fig.legend(
            all_h,
            all_l,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncol,
            frameon=False,
            fontsize=8,
            columnspacing=1.2,
            handlelength=2.0,
        )

        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
        fig.savefig(out_dir / filename, dpi=120, bbox_inches="tight")
        plt.close(fig)

    for cap, filename in ((1000.0, "ppl_under_1000.png"), (100.0, "ppl_under_100.png")):
        _draw_one(
            filename=filename,
            title=f"PPL & LR (ppl ≤ {cap:g})",
            ppl_cap=cap,
        )

    # 最近 5B token 窗口：忽略 ppl>1000（与 under_1000 同口径），
    # 纵轴 = 窗口内保留点的 gen+train 最高值 × 1.05。
    recent_span = 5_000_000_000
    recent_ppl_cap = 1000.0
    all_xs = train_plot_x + eval_x + train_lr_x
    if all_xs:
        x_right = max(all_xs)
        x_left = max(0, x_right - recent_span)
        t_win, t_win_ppl = _filter_xy(
            train_plot_x, train_ppl, x_min=x_left, y_max=recent_ppl_cap,
        )
        g_win, g_win_ppl = _filter_xy(
            eval_x, gen_ppl, x_min=x_left, y_max=recent_ppl_cap,
        )
        _draw_one(
            filename="ppl_recent_5b.png",
            title=f"PPL & LR (recent 5B tokens, ppl ≤ {recent_ppl_cap:g})",
            x_min=x_left,
            x_max=x_right,
            ppl_cap=recent_ppl_cap,
            auto_ppl_ylim_from=(t_win_ppl, g_win_ppl),
        )


def _as_optional_float(raw: Any) -> float | None:
    """把 tensor / 数值转成 float；nan/None → None。"""
    if raw is None:
        return None
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "item"):
        try:
            raw = raw.item()
        except (RuntimeError, ValueError):
            return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # nan
        return None
    return value


def build_train_row(
    step: int,
    tokens: int,
    train_loss: float,
    lr: float,
    tokens_per_sec: float,
    *,
    dual_branch: bool,
    loss_branch: str = "",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    denoise_mse = metrics.get("denoise_mse")
    decode_ce = metrics.get("decode_ce")
    late_ce = metrics.get("late_ce")
    lex_ce = metrics.get("lex_ce")
    attr = metrics.get("attr")
    chart_ce = metrics.get("chart_ce")
    row: dict[str, Any] = {
        "step": step,
        "tokens": tokens,
        "train_loss": round(train_loss, 6) if train_loss == train_loss else "",
        "train_ppl": "",
        "loss_branch": "",
        "denoise_mse": "",
        "decode_ce": "",
        "late_ce": "",
        "lex_ce": "",
        "attr": "",
        "chart_ce": "",
        "lr": lr,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
    if dual_branch:
        row["loss_branch"] = loss_branch
        mse = _as_optional_float(denoise_mse)
        ce = _as_optional_float(decode_ce)
        lce = _as_optional_float(late_ce)
        lxce = _as_optional_float(lex_ce)
        attr_v = _as_optional_float(attr)
        chart_v = _as_optional_float(chart_ce)
        if loss_branch == "mixed":
            # Official-style combined loss; still record both branch metrics.
            if mse is not None:
                row["denoise_mse"] = round(mse, 6)
            # 无 decode 样本时 ce 为 nan，跳过，不写假 ppl=1.0。
            if ce is not None:
                row["decode_ce"] = round(ce, 6)
                row["train_ppl"] = round(loss_to_ppl(ce), 4)
            if lce is not None:
                row["late_ce"] = round(lce, 6)
            if lxce is not None:
                row["lex_ce"] = round(lxce, 6)
            if attr_v is not None:
                row["attr"] = round(attr_v, 6)
            if chart_v is not None:
                row["chart_ce"] = round(chart_v, 6)
        elif loss_branch == "denoise":
            # MSE is not a CE; leave train_ppl empty.
            row["denoise_mse"] = round(train_loss, 6) if train_loss == train_loss else ""
            if lce is not None:
                row["late_ce"] = round(lce, 6)
            if lxce is not None:
                row["lex_ce"] = round(lxce, 6)
        elif loss_branch == "decode":
            # PPL only from decode CE (exp(ce)), never from denoise MSE.
            if train_loss == train_loss:
                row["decode_ce"] = round(train_loss, 6)
                row["train_ppl"] = round(loss_to_ppl(train_loss), 4)
                row["train_loss"] = row["decode_ce"]
        else:
            raise ValueError(
                f"dual_branch logging requires loss_branch "
                f"'denoise', 'decode', or 'mixed', got {loss_branch!r}"
            )
    else:
        if train_loss == train_loss:
            row["train_ppl"] = round(loss_to_ppl(train_loss), 4)
    return row


def _train_metrics_text(row: dict[str, Any]) -> str:
    branch = row.get("loss_branch") or ""
    if branch == "mixed":
        mse = row.get("denoise_mse")
        ce = row.get("decode_ce")
        parts = [f"[mixed] loss {row['train_loss']:.4f}"]
        if mse not in ("", None):
            parts.append(f"mse {mse}")
        if ce not in ("", None):
            parts.append(f"ce {ce} ppl {row.get('train_ppl', '')}")
        attr = row.get("attr")
        if attr not in ("", None):
            parts.append(f"attr {attr}")
        parts.append(f"lr {row['lr']:.2e}")
        parts.append(f"{row['tokens_per_sec']:.0f} tok/s")
        return " | ".join(parts)
    if branch == "denoise":
        return (
            f"[denoise] mse {row['train_loss']:.4f} | "
            f"lr {row['lr']:.2e} | {row['tokens_per_sec']:.0f} tok/s"
        )
    if branch == "decode":
        return (
            f"[decode] ce {row['train_loss']:.4f} ppl {row['train_ppl']} | "
            f"lr {row['lr']:.2e} | {row['tokens_per_sec']:.0f} tok/s"
        )
    return (
        f"loss {row['train_loss']:.4f} ppl {row['train_ppl']} | "
        f"lr {row['lr']:.2e} | {row['tokens_per_sec']:.0f} tok/s"
    )


def format_interval_summary(
    step: int,
    max_steps: int,
    row: dict[str, Any],
) -> list[str]:
    pct = 100.0 * (step + 1) / max_steps
    tokens = row.get("tokens")
    tok_part = f" tokens={tokens:,}" if isinstance(tokens, int) else ""
    return [
        f"[{step + 1}/{max_steps} ({pct:.1f}%){tok_part}] {_train_metrics_text(row)}"
    ]


def _rank0_log(msg: str, pbar: tqdm | None) -> None:
    line = f"{_TRAIN_LOG} {msg}"
    if pbar is not None:
        tqdm.write(line)
    else:
        print(line, flush=True)
