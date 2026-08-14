"""TriFluency B 轴补充：glue token 占固定长度（默认 1024）的比例。

一条假词（``[A-Za-z]{4,}`` 且 wordfreq zipf=0）若由 ≥2 个 T5/BPE piece
拼成（词首 ``▁``/``Ġ`` + continuation），这些 piece 计为 glue token。

语料级 ``glue_token_pct`` = 100 × (glue piece 总数) / (N × seq_len)，
``seq_len`` 默认 1024（生成长度 / OWT 切片宽），与 re-encode 后实际
piece 数无关。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

GLUE_SEQ_LEN = 1024
GLUE_SUMMARY_KEY = "glue_token_pct"

_WORD_RE = re.compile(r"^[A-Za-z]{4,}$")
_SPECIAL_EXACT = frozenset({"<pad>", "</s>", "<unk>", "<s>", "<|endoftext|>"})


def _is_special_piece(p: str) -> bool:
    if p in _SPECIAL_EXACT:
        return True
    if p.startswith("<extra_id_"):
        return True
    return False


def _is_word_start(p: str) -> bool:
    return p.startswith("▁") or p.startswith("Ġ")


def _surface(p: str) -> str:
    if _is_special_piece(p):
        return ""
    if p.startswith("▁") or p.startswith("Ġ"):
        return p[1:]
    return p


@lru_cache(maxsize=65536)
def _zipf0(word: str) -> bool:
    from wordfreq import zipf_frequency

    return float(zipf_frequency(word.lower(), "en")) == 0.0


def _group_words(pieces: Sequence[str]) -> list[tuple[str, list[str]]]:
    """按词首标记切成 (surface, content_pieces)。"""
    words: list[tuple[str, list[str]]] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        surf = "".join(_surface(p) for p in buf)
        content = [p for p in buf if _surface(p)]
        if content:
            words.append((surf, content))
        buf.clear()

    for p in pieces:
        if _is_special_piece(p):
            continue
        if _is_word_start(p):
            flush()
            buf = [p]
        else:
            buf.append(p)
    flush()
    return words


def count_glue_pieces(pieces: Sequence[str]) -> int:
    """一条序列里属于 glue 假词的 piece 数。"""
    n = 0
    for surf, content in _group_words(pieces):
        if len(content) < 2:
            continue
        if not _WORD_RE.fullmatch(surf):
            continue
        if not _zipf0(surf):
            continue
        n += len(content)
    return n


def _get_tokenizer(name: str) -> Any:
    from tokenizer import get_tokenizer

    return get_tokenizer(name)


def texts_to_piece_lists(
    texts: Sequence[str],
    *,
    tokenizer_name: str = "t5-small",
) -> list[list[str]]:
    tok = _get_tokenizer(tokenizer_name)
    out: list[list[str]] = []
    for text in texts:
        ids = tok.encode_preprocess(text or "")
        out.append(list(tok.convert_ids_to_tokens(ids)))
    return out


def score_glue(
    texts: Sequence[str],
    *,
    tokenizer_name: str = "t5-small",
    seq_len: int = GLUE_SEQ_LEN,
    piece_lists: Sequence[Sequence[str]] | None = None,
) -> tuple[list[float], list[int], dict[str, float]]:
    """返回 (per_sample_frac, per_sample_count, summary)。

    ``frac`` = n_glue / seq_len；``glue_token_pct`` 为语料级百分数。
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if piece_lists is None:
        piece_lists = texts_to_piece_lists(texts, tokenizer_name=tokenizer_name)
    elif len(piece_lists) != len(texts):
        raise ValueError(
            f"piece_lists length {len(piece_lists)} != texts {len(texts)}"
        )

    fracs: list[float] = []
    counts: list[int] = []
    total = 0
    for pcs in piece_lists:
        n = count_glue_pieces(pcs)
        counts.append(n)
        fracs.append(float(n) / float(seq_len))
        total += n

    n_samples = max(len(texts), 1)
    summary = {
        "glue_token_pct": 100.0 * float(total) / float(n_samples * seq_len),
        "glue_token_mean": float(total) / float(n_samples),
        "glue_seq_len": float(seq_len),
        "total_glue_tokens": float(total),
    }
    return fracs, counts, summary


def summary_has_glue(summary: MappingOrPath) -> bool:
    data = _load_summary(summary)
    if not isinstance(data, dict):
        return False
    val = data.get(GLUE_SUMMARY_KEY)
    if val is None or val == "":
        return False
    try:
        float(val)
    except (TypeError, ValueError):
        return False
    return True


MappingOrPath = dict[str, Any] | Path | str


def _load_summary(summary: MappingOrPath) -> dict[str, Any] | None:
    if isinstance(summary, dict):
        return summary
    path = Path(summary)
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def patch_glue_summary(
    out_dir: Path | str,
    *,
    tokenizer_name: str = "t5-small",
    seq_len: int = GLUE_SEQ_LEN,
) -> dict[str, float]:
    """从已有 ``samples.txt`` 补 ``summary.json`` 的 glue 字段（不重跑 gpt2）。"""
    from eval.io import read_annotated

    out_dir = Path(out_dir)
    samples = out_dir / "samples.txt"
    if not samples.is_file():
        raise FileNotFoundError(f"missing samples: {samples}")
    texts = read_annotated(samples).texts
    if not texts:
        raise ValueError(f"no samples parsed from {samples}")
    _fracs, counts, gsum = score_glue(
        texts, tokenizer_name=tokenizer_name, seq_len=seq_len,
    )
    sum_path = out_dir / "summary.json"
    data: dict[str, Any] = {}
    if sum_path.is_file():
        try:
            loaded = json.loads(sum_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
    data.update(gsum)
    sum_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _patch_per_sample_csv(out_dir / "per_sample.csv", counts, seq_len)
    return gsum


def rescore_glue_step_dir(
    step_dir: Path | str,
    *,
    tokenizer_name: str = "t5-small",
    seq_len: int = GLUE_SEQ_LEN,
    force: bool = False,
) -> list[Path]:
    """扫描 ``{step}/{generate-hash}/``，缺 glue 列则从 samples 补打分。"""
    step_dir = Path(step_dir)
    if not step_dir.is_dir():
        return []
    patched: list[Path] = []
    for child in sorted(step_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if not (child / "samples.txt").is_file():
            continue
        if not (child / "summary.json").is_file():
            continue
        sl = seq_len
        fp_path = child / "fingerprint.json"
        if fp_path.is_file():
            try:
                fp = json.loads(fp_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                fp = None
            if isinstance(fp, dict) and fp.get("num_tokens"):
                try:
                    sl = int(fp["num_tokens"])
                except (TypeError, ValueError):
                    sl = seq_len
        if summary_has_glue(child) and not force:
            continue
        gsum = patch_glue_summary(
            child, tokenizer_name=tokenizer_name, seq_len=sl,
        )
        name = child.name
        fp_path = child / "fingerprint.json"
        if fp_path.is_file():
            try:
                fp2 = json.loads(fp_path.read_text(encoding="utf-8"))
                if isinstance(fp2, dict) and fp2.get("name"):
                    name = str(fp2["name"])
            except (OSError, json.JSONDecodeError):
                pass
        print(
            f"[glue] {name}  {child.name[:16]}  "
            f"glue_token_pct={gsum['glue_token_pct']:.4f}",
            flush=True,
        )
        patched.append(child)
    return patched


def _patch_per_sample_csv(
    path: Path, counts: list[int], seq_len: int,
) -> None:
    if not path.is_file():
        return
    import csv

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        return
    if "glue_token_n" not in fieldnames:
        fieldnames.extend(["glue_token_n", "glue_token_frac"])
    for i, row in enumerate(rows):
        n = counts[i] if i < len(counts) else 0
        row["glue_token_n"] = str(n)
        row["glue_token_frac"] = f"{n / float(seq_len):.6g}"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
