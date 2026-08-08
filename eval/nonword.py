"""TriFluency B 轴：假词率（ACE App.C：wordfreq zipf==0）。"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def _is_nonword(word: str) -> bool:
    from wordfreq import zipf_frequency

    return float(zipf_frequency(word.lower(), "en")) == 0.0


def score_nonwords(texts: list[str]) -> tuple[list[float], list[int], dict[str, float]]:
    """返回 (per_sample_frac, per_sample_count, summary)。"""
    fracs: list[float] = []
    counts: list[int] = []
    total_words = 0
    total_nw = 0
    samples_with = 0

    for text in texts:
        words = _WORD_RE.findall(text or "")
        nw = sum(1 for w in words if _is_nonword(w))
        n = len(words)
        frac = float(nw) / float(n) if n > 0 else 0.0
        fracs.append(frac)
        counts.append(nw)
        total_words += n
        total_nw += nw
        if nw > 0:
            samples_with += 1

    n_samples = max(len(texts), 1)
    summary = {
        "nonword_word_pct": (
            100.0 * float(total_nw) / float(total_words) if total_words > 0 else 0.0
        ),
        "nonword_sample_pct": 100.0 * float(samples_with) / float(n_samples),
        "total_words": float(total_words),
        "total_nonwords": float(total_nw),
    }
    return fracs, counts, summary
