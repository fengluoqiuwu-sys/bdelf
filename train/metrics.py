"""训练指标 CSV I/O 与日志格式（主表核心列；卫星表字段常量）。"""

from __future__ import annotations

import atexit
import csv
import math
import sys
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm

from models import kind_of

# lm 主表：仅核心列
TRAIN_CSV_FIELDS_LM = [
    "step",
    "tokens",
    "train_loss",
    "train_ppl",
    "lr",
    "tokens_per_sec",
]
TRAIN_CSV_FIELDS_LATENT = [
    "curriculum_stage",
    "step",
    "tokens",
    "train_loss",
    "lr",
    "tokens_per_sec",
    "recon_ce",
    "kl",
    "mask",
    "token_acc",
    "mask_acc",
    "beta_kl",
    "lambda_mask",
]
TRAIN_CSV_FIELDS = TRAIN_CSV_FIELDS_LM

EVAL_CSV_FIELDS_LM = [
    "step",
    "tokens",
    "lr",
    "eval_loss",
    "eval_ppl",
]
EVAL_CSV_FIELDS_LATENT = [
    "curriculum_stage",
    "step",
    "tokens",
    "lr",
    "eval_loss",
]
EVAL_CSV_FIELDS = EVAL_CSV_FIELDS_LM

# lm 官方卫星表（对齐键仅 step；不含 VAE 列）
TRAIN_OFFICIAL_FIELDS_LM = [
    "step",
    "loss_branch",
    "denoise_mse",
    "decode_ce",
    "late_ce",
    "lex_ce",
    "attr",
    "attr_rho",
    "chart_ce",
    "commit",
]
TRAIN_OFFICIAL_FIELDS = TRAIN_OFFICIAL_FIELDS_LM

# Eval 官方卫星表
EVAL_OFFICIAL_FIELDS = [
    "step",
    "gen_loss",
    "gen_ppl",
    "gen_uniq_mean",
    "gen_nonempty_frac",
    "entropy",
    "dist1",
]

# 逐样本落盘基础列（外部可加列）
EVAL_SAMPLE_BASE_FIELDS = [
    "id",
    "text",
    "gen_ppl",
    "entropy",
]

_TRAIN_LOG = "[train]"


def train_csv_fields(model: str, cfg: Any | None = None) -> list[str]:
    if kind_of(model) == "latent":
        fields = list(TRAIN_CSV_FIELDS_LATENT)
        if cfg is not None and cfg.extra.get("curriculum"):
            if "observation_window" not in fields:
                fields.append("observation_window")
        return fields
    return list(TRAIN_CSV_FIELDS_LM)


def eval_csv_fields(model: str, cfg: Any | None = None) -> list[str]:
    if kind_of(model) == "latent":
        fields = list(EVAL_CSV_FIELDS_LATENT)
        if cfg is not None and cfg.extra.get("curriculum"):
            from train.latent_eval import latent_curriculum_eval_fields

            for name in latent_curriculum_eval_fields():
                if name not in fields:
                    fields.append(name)
        return fields
    return list(EVAL_CSV_FIELDS_LM)


def train_official_fields(model: str) -> list[str]:
    if kind_of(model) == "latent":
        return []
    return list(TRAIN_OFFICIAL_FIELDS_LM)


def _train_log(msg: str, *, file: Any = None) -> None:
    if file is None:
        file = sys.stdout
    print(f"{_TRAIN_LOG} {msg}", file=file, flush=True)


def loss_to_ppl(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def _csv_header(csv_path: Path) -> list[str]:
    """只读首行表头，不载入数据行。"""
    with open(csv_path, encoding="utf-8", newline="") as f:
        return list(next(csv.reader(f), []))


# 进程内：表头已核对的路径、以及追加用常开句柄（避免每步 open/读表头/close）。
_csv_schema_ok: dict[str, tuple[tuple[str, ...], bool]] = {}
_csv_append_files: dict[str, TextIO] = {}
_csv_atexit_registered = False


def _close_csv_append_files() -> None:
    for fh in _csv_append_files.values():
        try:
            fh.close()
        except OSError:
            pass
    _csv_append_files.clear()


def _register_csv_atexit() -> None:
    global _csv_atexit_registered
    if _csv_atexit_registered:
        return
    atexit.register(_close_csv_append_files)
    _csv_atexit_registered = True


def _drop_csv_append_handle(csv_path: Path) -> None:
    key = str(csv_path)
    fh = _csv_append_files.pop(key, None)
    if fh is not None:
        fh.close()
    _csv_schema_ok.pop(key, None)


def _mark_csv_schema_ok(
    csv_path: Path, fields: list[str], *, extend: bool = False,
) -> None:
    _csv_schema_ok[str(csv_path)] = (tuple(fields), extend)


def prepare_csv_for_append(
    csv_path: Path,
    fields: list[str],
    *,
    extend: bool = False,
) -> None:
    """训练启动时调用：没有文件则建表头；已有则对齐新列（必要时整表改写一次）。

    之后 ``append_csv_row`` 只追加，不再读/写整文件。
    """
    _register_csv_atexit()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        init_csv_header(csv_path, fields)
    else:
        ensure_csv_schema(csv_path, fields, extend=extend)
    _mark_csv_schema_ok(csv_path, fields, extend=extend)


def append_csv_row(
    csv_path: Path,
    fields: list[str],
    row: dict[str, Any],
    *,
    extend: bool = False,
) -> None:
    """只追加一行。新列必须已在 ``prepare_csv_for_append`` / 启动时加好。"""
    _register_csv_atexit()
    key = str(csv_path)
    spec = (tuple(fields), extend)
    write_header = False
    if not csv_path.exists():
        # 未走启动 prepare 的兜底：只写表头，不读已有数据。
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = True
    _csv_schema_ok[key] = spec
    fh = _csv_append_files.get(key)
    if fh is None:
        # 行缓冲：每行一次 write，不再每步重新打开或读表头。
        fh = open(csv_path, "a", newline="", encoding="utf-8", buffering=1)
        _csv_append_files[key] = fh
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    writer.writerow({k: row.get(k, "") for k in fields})


def init_csv_header(csv_path: Path, fields: list[str]) -> None:
    """若尚无文件则只写表头。"""
    if csv_path.exists():
        _mark_csv_schema_ok(csv_path, fields)
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _drop_csv_append_handle(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    _mark_csv_schema_ok(csv_path, fields)


def ensure_csv_schema(
    csv_path: Path,
    fields: list[str],
    *,
    extend: bool = False,
) -> None:
    """对齐表头。仅训练启动时调用；表头已一致则只读首行。

    有新列才整表改写一次。训练循环里请用 ``append_csv_row``，不要走这里。

    - ``extend=False``：表头严格为 ``fields``（主表 / 官方卫星）。
    - ``extend=True``：保留已有列并追加 ``fields`` 中的新列（外部 / samples）。
    已有行对新列留空，不回填。
    """
    if not csv_path.exists():
        return
    old_fields = _csv_header(csv_path)
    if extend:
        out_fields = list(old_fields)
        for name in fields:
            if name not in out_fields:
                out_fields.append(name)
    else:
        out_fields = list(fields)
    if out_fields == old_fields:
        return
    _drop_csv_append_handle(csv_path)
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in out_fields})
    _mark_csv_schema_ok(csv_path, out_fields, extend=extend)


def truncate_csv_for_resume(csv_path: Path, start_step: int, fields: list[str]) -> int:
    """保留 step < start_step 的行；表头用 ``fields``（缺列留空）。"""
    if not csv_path.exists():
        return 0
    _drop_csv_append_handle(csv_path)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_by_step: dict[int, dict[str, str]] = {}
        for row in reader:
            step = int(row["step"])
            if step < start_step:
                rows_by_step[step] = row
    rows = [rows_by_step[s] for s in sorted(rows_by_step)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    _mark_csv_schema_ok(csv_path, fields)
    return len(rows)


def _curriculum_stage_rank(name: str) -> int:
    raw = (name or "").strip().lower()
    if len(raw) >= 2 and raw[0] == "s" and raw[1:].isdigit():
        return int(raw[1:])
    return 0


def truncate_csv_for_curriculum_resume(
    csv_path: Path,
    *,
    resume_stage: str,
    resume_step: int,
    fields: list[str],
) -> int:
    """保留 (curriculum_stage, step) 严格早于 resume 点的行（step 为阶段内计数）。"""
    if not csv_path.exists():
        return 0
    _drop_csv_append_handle(csv_path)
    resume_key = (_curriculum_stage_rank(resume_stage), int(resume_step))
    kept_rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = str(row.get("curriculum_stage") or "")
            try:
                step = int(row.get("step") or 0)
            except ValueError:
                continue
            if (_curriculum_stage_rank(stage), step) < resume_key:
                kept_rows.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in kept_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    _mark_csv_schema_ok(csv_path, fields)
    return len(kept_rows)


def _as_optional_float(raw: Any) -> float | None:
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


def _fmt_cell(value: Any) -> Any:
    if value is None:
        return ""
    f = _as_optional_float(value)
    if f is None and value != 0 and value != 0.0:
        if isinstance(value, str):
            return value
        return ""
    if f is None:
        return ""
    return f


def build_train_core_row(
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
    """主表行：仅核心列。"""
    metrics = metrics or {}
    row: dict[str, Any] = {
        "step": step,
        "tokens": tokens,
        "train_loss": round(train_loss, 6) if train_loss == train_loss else "",
        "train_ppl": "",
        "lr": lr,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
    if dual_branch:
        if loss_branch == "mixed":
            ce = _as_optional_float(metrics.get("decode_ce"))
            if ce is not None:
                row["train_ppl"] = round(loss_to_ppl(ce), 4)
        elif loss_branch == "denoise":
            pass  # MSE：不写 ppl
        elif loss_branch == "decode":
            if train_loss == train_loss:
                row["train_ppl"] = round(loss_to_ppl(train_loss), 4)
                row["train_loss"] = round(train_loss, 6)
        else:
            raise ValueError(
                f"dual_branch logging requires loss_branch "
                f"'denoise', 'decode', or 'mixed', got {loss_branch!r}"
            )
    else:
        if train_loss == train_loss:
            row["train_ppl"] = round(loss_to_ppl(train_loss), 4)
    return row


def build_latent_train_row(
    step: int,
    tokens: int,
    train_loss: float,
    lr: float,
    tokens_per_sec: float,
    *,
    metrics: dict[str, Any] | None = None,
    curriculum_stage: str = "",
    observation_window: bool | None = None,
) -> dict[str, Any]:
    """latent 宽主表行（无 train_ppl / official 卫星）。"""
    metrics = dict(metrics or {})
    row: dict[str, Any] = {
        "curriculum_stage": curriculum_stage,
        "step": step,
        "tokens": tokens,
        "train_loss": round(train_loss, 6) if train_loss == train_loss else "",
        "lr": lr,
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
    if observation_window is not None:
        row["observation_window"] = int(observation_window)
    for key in (
        "recon_ce",
        "kl",
        "mask",
        "token_acc",
        "mask_acc",
        "beta_kl",
        "lambda_mask",
    ):
        val = _as_optional_float(metrics.get(key))
        if val is None:
            row[key] = ""
        elif key in ("token_acc", "mask_acc"):
            row[key] = round(val, 4)
        else:
            row[key] = round(val, 6)
    return row


def build_train_official_row(
    step: int,
    *,
    dual_branch: bool,
    loss_branch: str = "",
    train_loss: float = float("nan"),
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """官方卫星行；无任何扩展时返回 None（不写）。"""
    metrics = dict(metrics or {})
    row: dict[str, Any] = {k: "" for k in TRAIN_OFFICIAL_FIELDS_LM}
    row["step"] = step

    has_any = False
    if dual_branch and loss_branch:
        row["loss_branch"] = loss_branch
        has_any = True
        mse = _as_optional_float(metrics.get("denoise_mse"))
        ce = _as_optional_float(metrics.get("decode_ce"))
        if loss_branch == "mixed":
            if mse is not None:
                row["denoise_mse"] = round(mse, 6)
            if ce is not None:
                row["decode_ce"] = round(ce, 6)
        elif loss_branch == "denoise":
            row["denoise_mse"] = (
                round(train_loss, 6) if train_loss == train_loss else ""
            )
        elif loss_branch == "decode":
            if train_loss == train_loss:
                row["decode_ce"] = round(train_loss, 6)

    for key in (
        "late_ce",
        "lex_ce",
        "attr",
        "attr_rho",
        "chart_ce",
        "commit",
        "denoise_mse",
        "decode_ce",
    ):
        if dual_branch and key in ("denoise_mse", "decode_ce") and loss_branch:
            continue  # 已在分支逻辑写入
        val = _as_optional_float(metrics.get(key))
        if val is not None:
            row[key] = round(val, 6)
            has_any = True

    if not has_any:
        return None
    return row


def _train_metrics_text(core: dict[str, Any], official: dict[str, Any] | None) -> str:
    branch = (official or {}).get("loss_branch") or ""
    tok_s = core.get("tokens_per_sec", "")
    lr = core.get("lr", "")
    if branch == "mixed":
        parts = [f"[mixed] loss {core.get('train_loss', '')}"]
        mse = (official or {}).get("denoise_mse")
        ce = (official or {}).get("decode_ce")
        if mse not in ("", None):
            parts.append(f"mse {mse}")
        if ce not in ("", None):
            parts.append(f"ce {ce} ppl {core.get('train_ppl', '')}")
        attr = (official or {}).get("attr")
        if attr not in ("", None):
            parts.append(f"attr {attr}")
        parts.append(f"lr {lr:.2e}" if isinstance(lr, float) else f"lr {lr}")
        parts.append(f"{tok_s:.0f} tok/s" if isinstance(tok_s, float) else f"{tok_s} tok/s")
        return " | ".join(parts)
    if branch == "denoise":
        return (
            f"[denoise] mse {core.get('train_loss', '')} | "
            f"lr {lr:.2e} | {float(tok_s):.0f} tok/s"
            if isinstance(lr, float) and isinstance(tok_s, (int, float))
            else f"[denoise] mse {core.get('train_loss', '')} | lr {lr} | {tok_s} tok/s"
        )
    if branch == "decode":
        return (
            f"[decode] ce {core.get('train_loss', '')} ppl {core.get('train_ppl', '')} | "
            f"lr {lr:.2e} | {float(tok_s):.0f} tok/s"
            if isinstance(lr, float) and isinstance(tok_s, (int, float))
            else (
                f"[decode] ce {core.get('train_loss', '')} "
                f"ppl {core.get('train_ppl', '')} | lr {lr} | {tok_s} tok/s"
            )
        )
    return (
        f"loss {core.get('train_loss', '')} ppl {core.get('train_ppl', '')} | "
        f"lr {lr:.2e} | {float(tok_s):.0f} tok/s"
        if isinstance(lr, float) and isinstance(tok_s, (int, float))
        else (
            f"loss {core.get('train_loss', '')} ppl {core.get('train_ppl', '')} | "
            f"lr {lr} | {tok_s} tok/s"
        )
    )


def format_interval_summary(
    step: int,
    max_steps: int,
    core: dict[str, Any],
    official: dict[str, Any] | None = None,
) -> list[str]:
    pct = 100.0 * (step + 1) / max_steps
    tokens = core.get("tokens")
    tok_part = f" tokens={tokens:,}" if isinstance(tokens, int) else ""
    return [
        f"[{step + 1}/{max_steps} ({pct:.1f}%){tok_part}] "
        f"{_train_metrics_text(core, official)}"
    ]


def _rank0_log(msg: str, pbar: tqdm | None) -> None:
    line = f"{_TRAIN_LOG} {msg}"
    if pbar is not None:
        tqdm.write(line)
    else:
        print(line, flush=True)
