"""TriFluency A 轴：n-gram 重复率与 human-clean 接受。"""

from __future__ import annotations

import re
from statistics import median

# ACE / XSum 人类 seq-rep-4 的 95 分位
TAU_H = 0.0192

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^A-Za-z0-9'\s]")


def tokenize(text: str) -> list[str]:
    """小写空白切分：字母数字/' 为词，其余标点单独成 token。"""
    return _TOKEN_RE.findall(text.lower())


def seq_rep_4(text: str) -> float:
    """样本内 4-gram 重复率（Welleck / ACE）：``1 - |unique|/|total|``。"""
    toks = tokenize(text)
    if len(toks) < 4:
        return 0.0
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    if not grams:
        return 0.0
    return 1.0 - float(len(set(grams))) / float(len(grams))


def accept_human(rep: float, *, tau_h: float = TAU_H) -> bool:
    return rep < tau_h


def score_repetition(
    texts: list[str],
    *,
    tau_h: float = TAU_H,
) -> tuple[list[float], list[bool], dict[str, float]]:
    """返回 (per_rep, per_accept, summary)。"""
    reps = [seq_rep_4(t) for t in texts]
    accepts = [accept_human(r, tau_h=tau_h) for r in reps]
    n = max(len(texts), 1)
    summary = {
        "accept_at_human": float(sum(accepts)) / float(n),
        "median_rep": float(median(reps)) if reps else float("nan"),
        "mean_rep": float(sum(reps) / n) if reps else float("nan"),
        "tau_h": float(tau_h),
    }
    return reps, accepts, summary
