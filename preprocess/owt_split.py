"""OWT 句段切分（GPT-2 BPE）。

切点落在分隔符最后一个 token 之后。lookback 内优先段落，再换行，再句末。
找不到则在 content 上限硬切，余段从切点继续（不跳到下一句）。

判定在原文 + offset_mapping 上做（GPT-2 是 byte BPE，按单 token decode
会对不齐 CJK）。需要 Fast tokenizer。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# 档位：数字越小越优先。P3 默认不参与。
# ---------------------------------------------------------------------------
P0_PARAGRAPH = 0  # 连续空行
P1_NEWLINE = 1  # 单个换行
P2_SENTENCE = 2  # 句末标点 + 可选闭合引号/括号 + 尾随换行
P3_WEAK = 3  # 分号；默认关闭

USE_P3 = False

# 闭合符（句末核后面允许跟的）
_CLOSERS = "\"'”’」』)）\\]】"

_PATTERNS: list[tuple[int, re.Pattern[str]]] = [
    (P0_PARAGRAPH, re.compile(r"(?:\r\n|\n){2,}")),
    (P1_NEWLINE, re.compile(r"\r\n|\n")),
    # 省略号必须写在单字符 `.` 前面，否则 '...' 会被切成第一个点
    (
        P2_SENTENCE,
        re.compile(rf"(?:\.\.\.+|…+|[.!?。！？])[{_CLOSERS}]*(?:\r\n|\n)*"),
    ),
    (P3_WEAK, re.compile(rf"[;；][{_CLOSERS}]*(?:\r\n|\n)*")),
]


def lookback_tokens(d: int) -> int:
    """向前看的 token 数。d=512 → 64；d=2048 → 256。"""
    if d < 2:
        raise ValueError("d must be >= 2")
    return min(256, max(1, d // 8))


@dataclass(frozen=True)
class DelimHit:
    priority: int
    char_start: int
    char_end: int
    token_end: int  # 左段 token 切片的右开端


@dataclass(frozen=True)
class EncodedDoc:
    text: str
    token_ids: list[int]
    offsets: list[tuple[int, int]]  # 每个 token 在原文上的 [start, end)


def encode_doc(tokenizer: Any, text: str) -> EncodedDoc:
    """必须用 Fast tokenizer，才能拿 offset_mapping。"""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = list(enc["offset_mapping"])
    if len(ids) != len(offs):
        raise RuntimeError("input_ids / offset_mapping 长度不一致")
    return EncodedDoc(text=text, token_ids=ids, offsets=offs)


def _char_end_to_token_end(offsets: list[tuple[int, int]], char_end: int) -> int:
    """分隔符在原文 [..., char_end) 结束 → 包含该字符的最后一个 token 之后。

    GPT-2 对 CJK 常是多个 byte token 共用同一字符区间，必须取 s < char_end 的最大 i+1。
    """
    token_end = 0
    for i, (s, _e) in enumerate(offsets):
        if s < char_end:
            token_end = i + 1
        elif s >= char_end:
            break
    return token_end


def find_delimiters(doc: EncodedDoc, *, use_p3: bool = USE_P3) -> list[DelimHit]:
    """全文所有分隔符（含互相重叠，例如 \\n\\n 既是 P0 也覆盖两个 P1）。"""
    hits: list[DelimHit] = []
    for priority, pat in _PATTERNS:
        if priority == P3_WEAK and not use_p3:
            continue
        for m in pat.finditer(doc.text):
            cs, ce = m.start(), m.end()
            if ce <= cs:
                continue
            token_end = _char_end_to_token_end(doc.offsets, ce)
            if token_end <= 0:
                continue
            hits.append(
                DelimHit(
                    priority=priority,
                    char_start=cs,
                    char_end=ce,
                    token_end=token_end,
                )
            )
    return hits


def choose_split(
    hits: list[DelimHit],
    *,
    start: int,
    target: int,
    lookback: int,
) -> int | None:
    """在 tokens[start:target] 的末尾 lookback 段里选切点。

    候选：token_end ∈ (max(start, target-lookback), target]。
    先取档位最高（priority 最小），同档取最靠右（最接近 target）。
    没有则返回 None（调用方硬切在 target）。
    """
    if target <= start:
        return None
    lo = max(start, target - lookback)
    best: DelimHit | None = None
    for h in hits:
        if not (lo < h.token_end <= target):
            continue
        if best is None:
            best = h
            continue
        if h.priority < best.priority:
            best = h
        elif h.priority == best.priority and h.token_end > best.token_end:
            best = h
    return None if best is None else best.token_end


def split_token_ranges(
    doc: EncodedDoc,
    *,
    d: int,
    hits: list[DelimHit] | None = None,
    use_p3: bool = USE_P3,
) -> list[tuple[int, int]]:
    """把文档切成若干 content 区间 [left, right)，不含 BOS/EOS。

    content 上限 = d-2（预留 BOS/EOS）。整篇 content <= d-2 则一段。
    """
    if d < 2:
        raise ValueError("d must be >= 2")
    n = len(doc.token_ids)
    if n == 0:
        return []
    content_cap = d - 2
    lb = lookback_tokens(d)
    if hits is None:
        hits = find_delimiters(doc, use_p3=use_p3)

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < n:
        remaining = n - start
        if remaining <= content_cap:
            ranges.append((start, n))
            break
        target = start + content_cap
        cut = choose_split(hits, start=start, target=target, lookback=lb)
        if cut is None:
            cut = target  # 硬切；下一圈从 cut 继续，不跳句
        if cut <= start or cut > target:
            cut = target
        ranges.append((start, cut))
        start = cut
    return ranges


def wrap_chunk(
    content_ids: list[int],
    *,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    return [bos_id, *content_ids, eos_id]


def chunk_document(
    tokenizer: Any,
    text: str,
    *,
    d: int,
    bos_id: int,
    eos_id: int,
    min_len: int = 128,
    use_p3: bool = USE_P3,
) -> list[list[int]]:
    """切一篇：加 BOS/EOS 后仍 <= d；丢掉包装后长度 < min_len 的段。"""
    doc = encode_doc(tokenizer, text)
    hits = find_delimiters(doc, use_p3=use_p3)
    out: list[list[int]] = []
    for left, right in split_token_ranges(doc, d=d, hits=hits, use_p3=use_p3):
        wrapped = wrap_chunk(doc.token_ids[left:right], bos_id=bos_id, eos_id=eos_id)
        if len(wrapped) < min_len:
            continue
        if len(wrapped) > d:
            raise RuntimeError(f"chunk length {len(wrapped)} > d={d}")
        out.append(wrapped)
    return out


def pad_to(ids: Sequence[int], length: int, pad_id: int) -> list[int]:
    if len(ids) > length:
        raise ValueError(f"len={len(ids)} > pad_to={length}")
    return list(ids) + [pad_id] * (length - len(ids))


def bucket_pad_length(n: int) -> int | None:
    """变长桶：有效长度 n（含 BOS/EOS）。[128,256]→256，其后右闭。"""
    if n < 128:
        return None
    if n <= 256:
        return 256
    if n <= 512:
        return 512
    if n <= 1024:
        return 1024
    if n <= 2048:
        return 2048
    return None


def bucket_counts_from_lengths(
    lengths: Sequence[int],
    bucket_lengths: Sequence[int],
) -> dict[int, int]:
    """按有效长度统计各 pad 桶样本数。"""
    counts = {int(b): 0 for b in bucket_lengths}
    for n in lengths:
        pad_len = bucket_pad_length(int(n))
        if pad_len is not None and pad_len in counts:
            counts[pad_len] += 1
    return counts
