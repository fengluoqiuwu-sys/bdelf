"""Hacking-Gen-PPL 式零参数负对照：Periodic / Phrase-bank。"""

from __future__ import annotations

import itertools


def make_periodic(
    n: int,
    *,
    num_tokens: int = 1024,
    phrase: str = "apple table cloud river",
) -> list[str]:
    """周期拼接高频短语直到约 ``num_tokens`` 词。"""
    words = phrase.split()
    if not words:
        words = ["the", "a"]
    cycle = itertools.cycle(words)
    texts: list[str] = []
    for _ in range(n):
        toks = [next(cycle) for _ in range(num_tokens)]
        texts.append(" ".join(toks))
    return texts


def make_phrase_bank(
    n: int,
    *,
    num_tokens: int = 1024,
    bank: list[str] | None = None,
) -> list[str]:
    """从短语库均匀抽 5-gram 风格块拼接（非周期）。"""
    if bank is None:
        bank = [
            "the president said that the",
            "according to a new study",
            "in the middle of the",
            "it is important to note",
            "over the course of the",
            "one of the most important",
            "at the end of the",
            "as a matter of fact",
        ]
    texts: list[str] = []
    for i in range(n):
        parts: list[str] = []
        wcount = 0
        j = 0
        while wcount < num_tokens:
            phrase = bank[(i + j) % len(bank)]
            parts.append(phrase)
            wcount += len(phrase.split())
            j += 1
        toks = " ".join(parts).split()[:num_tokens]
        texts.append(" ".join(toks))
    return texts


def make_naive(kind: str, n: int, *, num_tokens: int = 1024) -> list[str]:
    kind = kind.lower().strip()
    if kind in ("periodic", "period"):
        return make_periodic(n, num_tokens=num_tokens)
    if kind in ("phrase_bank", "phrase-bank", "phrasebank"):
        return make_phrase_bank(n, num_tokens=num_tokens)
    raise ValueError(f"Unknown naive kind: {kind!r} (use periodic|phrase_bank)")
