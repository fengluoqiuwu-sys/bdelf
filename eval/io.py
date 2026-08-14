"""Annotated 样本文件读写与纯文本列表。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_HEADER_RE = re.compile(r"^#\s*(.+)$")
_SAMPLE_HDR_RE = re.compile(
    r"^###\s*sample\s+(\d+)/(\d+)(?:\s+gen_ppl=(\S+))?(?:\s+entropy=(\S+))?"
)


@dataclass
class SampleDoc:
    """一批无条件生成样本。"""

    texts: list[str]
    headers: dict[str, str] = field(default_factory=dict)
    # annotated 中可选旧分数字段（协议默认忽略、重算）
    annotated_gen_ppl: list[float | None] = field(default_factory=list)
    annotated_entropy: list[float | None] = field(default_factory=list)


def _parse_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def read_annotated(path: Path | str) -> SampleDoc:
    """解析 ``gen_samples_annotated`` 风格或纯文本（空行分隔）。"""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if "### sample" in raw:
        return _read_annotated_blocks(raw)
    # 纯文本：按空行分块，否则整文件一条
    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]
    if not chunks:
        chunks = [raw.strip()] if raw.strip() else []
    return SampleDoc(texts=chunks)


def _read_annotated_blocks(raw: str) -> SampleDoc:
    headers: dict[str, str] = {}
    texts: list[str] = []
    ppls: list[float | None] = []
    ents: list[float | None] = []

    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        hm = _HEADER_RE.match(line)
        if hm and not line.startswith("###"):
            body = hm.group(1).strip()
            if "=" in body:
                # 允许多个 key=value
                for part in body.split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        headers[k] = v
            i += 1
            continue
        if line.startswith("===") and i + 1 < len(lines):
            hdr = lines[i + 1]
            sm = _SAMPLE_HDR_RE.match(hdr.strip())
            if sm:
                ppls.append(_parse_float(sm.group(3)))
                ents.append(_parse_float(sm.group(4)))
                # 跳过 === / ### / ===
                i += 3
                buf: list[str] = []
                while i < len(lines) and not lines[i].startswith("==="):
                    buf.append(lines[i])
                    i += 1
                texts.append("\n".join(buf).rstrip("\n"))
                continue
        i += 1

    return SampleDoc(
        texts=texts,
        headers=headers,
        annotated_gen_ppl=ppls,
        annotated_entropy=ents,
    )


def write_texts(path: Path | str, texts: list[str], *, meta: dict[str, str] | None = None) -> None:
    """写出简单 annotated（无分数字段）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in (meta or {}).items():
            f.write(f"# {k}={v}\n")
        f.write("\n")
        for i, text in enumerate(texts, start=1):
            f.write("=" * 72 + "\n")
            f.write(f"### sample {i}/{len(texts)}\n")
            f.write("=" * 72 + "\n")
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            f.write("\n")
