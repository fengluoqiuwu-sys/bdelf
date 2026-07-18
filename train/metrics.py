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
    "train_loss",
    "train_ppl",
    "loss_branch",
    "denoise_mse",
    "decode_ce",
    "lr",
    "tokens_per_sec",
]
EVAL_CSV_FIELDS = ["step", "eval_loss", "eval_ppl", "gen_loss", "gen_ppl", "lr"]

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


def ensure_csv_schema(csv_path: Path, fields: list[str]) -> None:
    """Rewrite CSV if the on-disk header is missing newly added columns."""
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


def _decode_ce_train_series(
    train_rows: list[dict[str, str]],
) -> tuple[list[int], list[float], list[float]]:
    """Train decode-CE points for plotting (BDELF/ELF dual-branch)."""
    steps: list[int] = []
    ppls: list[float] = []
    lrs: list[float] = []
    for row in train_rows:
        if row.get("loss_branch") != "decode":
            continue
        ppl = _parse_float(row.get("train_ppl"))
        if ppl is None:
            ce = _parse_float(row.get("decode_ce"))
            if ce is not None:
                ppl = loss_to_ppl(ce)
        if ppl is None:
            continue
        steps.append(int(row["step"]))
        ppls.append(ppl)
        lrs.append(float(row["lr"]))
    return steps, ppls, lrs


def update_ppl_plots(
    train_csv: Path,
    eval_csv: Path,
    out_dir: Path,
) -> None:
    train_rows = _read_csv_rows(train_csv)
    eval_rows = _read_csv_rows(eval_csv)
    if not train_rows:
        return

    train_steps = [int(r["step"]) for r in train_rows]
    train_lr = [float(r["lr"]) for r in train_rows]

    dual_branch = any(r.get("loss_branch") in ("denoise", "decode") for r in train_rows)
    if dual_branch:
        train_plot_steps, train_ppl, _ = _decode_ce_train_series(train_rows)
    else:
        train_plot_steps = train_steps
        train_ppl = [_parse_float(r.get("train_ppl")) for r in train_rows]

    eval_steps = [int(r["step"]) for r in eval_rows]
    eval_ppl = [
        _parse_float(r.get("eval_ppl") or r.get("gpt2_ppl")) for r in eval_rows
    ]
    gen_ppl = [_parse_float(r.get("gen_ppl")) for r in eval_rows]

    for cap, filename in ((1000.0, "ppl_under_1000.png"), (100.0, "ppl_under_100.png")):
        t_steps, t_ppls = zip(
            *[
                (s, p)
                for s, p in zip(train_plot_steps, train_ppl)
                if p is not None and p <= cap
            ]
        ) if any(p is not None and p <= cap for p in train_ppl) else ([], [])

        e_steps, e_ppls = zip(
            *[(s, p) for s, p in zip(eval_steps, eval_ppl) if p is not None and p <= cap]
        ) if any(p is not None and p <= cap for p in eval_ppl) else ([], [])

        g_steps, g_ppls = zip(
            *[(s, p) for s, p in zip(eval_steps, gen_ppl) if p is not None and p <= cap]
        ) if any(p is not None and p <= cap for p in gen_ppl) else ([], [])

        if not t_steps and not e_steps and not g_steps:
            continue

        fig, ax_ppl = plt.subplots(figsize=(10, 4.5))

        if t_steps:
            train_label = (
                "train decode ppl (exp ce)"
                if dual_branch
                else "train ppl (exp loss)"
            )
            ax_ppl.plot(
                t_steps, t_ppls, color="#4C72B0", alpha=0.55, linewidth=1.2,
                label=train_label, zorder=1,
            )
        if e_steps:
            ax_ppl.plot(
                e_steps, e_ppls, color="#D62728", linewidth=2.8, marker="o",
                markersize=4, label="eval ppl (exp loss)", zorder=5,
            )
        if g_steps:
            ax_ppl.plot(
                g_steps, g_ppls, color="#2CA02C", linewidth=2.4, marker="s",
                markersize=4, label="gen ppl (gpt2-large)", zorder=6,
            )

        ax_lr = ax_ppl.twinx()
        lr_steps, lr_vals = zip(
            *[(s, lr) for s, lr in zip(train_steps, train_lr) if lr > 0]
        ) if train_lr else ([], [])
        if lr_steps:
            ax_lr.plot(
                lr_steps, lr_vals, color="#7F7F7F", linestyle="--",
                linewidth=1.0, alpha=0.9, label="lr", zorder=2,
            )
            ax_lr.set_ylabel("learning rate")
            ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        ax_ppl.set_xlabel("data step" if dual_branch else "step")
        ax_ppl.set_ylabel("perplexity")
        ax_ppl.set_title(f"PPL & LR (ppl ≤ {cap:g})")
        ax_ppl.grid(True, alpha=0.25)

        handles, labels = ax_ppl.get_legend_handles_labels()
        h2, l2 = ax_lr.get_legend_handles_labels()
        ax_ppl.legend(handles + h2, labels + l2, loc="upper right")

        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)


def build_train_row(
    step: int,
    train_loss: float,
    lr: float,
    tokens_per_sec: float,
    *,
    dual_branch: bool,
    loss_branch: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": step,
        "train_loss": round(train_loss, 6),
        "train_ppl": "",
        "loss_branch": "",
        "denoise_mse": "",
        "decode_ce": "",
        "lr": lr,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
    if dual_branch:
        row["loss_branch"] = loss_branch
        if loss_branch == "denoise":
            row["denoise_mse"] = round(train_loss, 6)
        elif loss_branch == "decode":
            ppl = loss_to_ppl(train_loss)
            row["decode_ce"] = round(train_loss, 6)
            row["train_ppl"] = round(ppl, 4)
            row["train_loss"] = row["decode_ce"]
    else:
        row["train_ppl"] = round(loss_to_ppl(train_loss), 4)
    return row


def _train_metrics_text(row: dict[str, Any]) -> str:
    branch = row.get("loss_branch") or ""
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
    return [f"[{step + 1}/{max_steps} ({pct:.1f}%)] {_train_metrics_text(row)}"]


def _rank0_log(msg: str, pbar: tqdm | None) -> None:
    line = f"{_TRAIN_LOG} {msg}"
    if pbar is not None:
        tqdm.write(line)
    else:
        print(line, flush=True)
