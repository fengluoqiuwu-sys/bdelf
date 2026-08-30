"""训练在线 gen-eval 的固定 GPT-2 前缀。

开训前生成并缓存：同一起始 seed，按递增 offset 采样，直到凑满 ``n`` 条
**token 序列互异**的前缀（保证一次 gen-eval 的 64/256 条都不重复）。
跨 run / eval step / micro-batch 共用 ``cache/gen_eval_prefixes/``。
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

import hf_config  # noqa: F401 — 须先于 huggingface / transformers

from eval.gen_ppl import get_gpt2_tokenizer, get_src_tokenizer
from train.metrics import _train_log

CACHE_DIR = Path("cache/gen_eval_prefixes")


class RaggedPrefixError(ValueError):
    """同一 micro-batch 内前缀长度不齐，须改为逐条 generate。"""


def gen_eval_prefix_spec(cfg: Any) -> tuple[int, int, str]:
    """``(前缀 token 数, 种子, HF 模型名)``；``0`` 表示无条件生成。"""
    n = int(getattr(cfg, "gen_eval_prefix_tokens", 0) or 0)
    seed = int(getattr(cfg, "gen_eval_prefix_seed", 42) or 42)
    model = str(getattr(cfg, "gen_eval_prefix_model", "gpt2") or "gpt2").strip()
    if n < 0:
        raise ValueError(f"gen_eval_prefix_tokens must be >= 0, got {n}")
    if not model:
        model = "gpt2"
    return n, seed, model


def prefix_cache_path(*, model: str, length: int, seed: int) -> Path:
    safe = str(model).replace("/", "--")
    return CACHE_DIR / f"{safe}_l{length}_s{seed}.json"


def _row_key(row: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in row)


def _dedupe_keep_first(
    texts: Sequence[str],
    token_ids: Sequence[Sequence[int]],
) -> tuple[list[str], list[list[int]]]:
    """按 token 序列去重，保留首次出现（批次内互异）。"""
    seen: set[tuple[int, ...]] = set()
    out_t: list[str] = []
    out_i: list[list[int]] = []
    for text, row in zip(texts, token_ids):
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        out_t.append(text)
        out_i.append([int(x) for x in row])
    return out_t, out_i


def _load_cache(path: Path) -> tuple[list[str], list[list[int]], int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    texts = list(raw.get("texts") or [])
    ids = list(raw.get("token_ids") or [])
    if len(texts) != len(ids):
        raise ValueError(f"{path}: texts/token_ids 长度不一致")
    ids = [[int(x) for x in row] for row in ids]
    next_offset = int(raw.get("next_offset", len(ids)))
    return texts, ids, next_offset


def _save_cache(
    path: Path,
    *,
    model: str,
    length: int,
    seed: int,
    texts: Sequence[str],
    token_ids: Sequence[Sequence[int]],
    next_offset: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "length": int(length),
        "seed": int(seed),
        "next_offset": int(next_offset),
        "texts": list(texts),
        "token_ids": [list(row) for row in token_ids],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(path)


def _bos_id(tok: Any) -> int:
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return int(bos if bos is not None else 50256)


def _sample_prefix_batch(
    model: Any,
    tok: Any,
    length: int,
    *,
    seeds: Sequence[int],
    device: torch.device,
) -> tuple[list[str], list[list[int]]]:
    """每条用独立 ``Generator(seed_i)``，batch 前向；与逐条 ``manual_seed`` 同构。"""
    n = len(seeds)
    if n < 1:
        return [], []
    bos = _bos_id(tok)
    gens = []
    for s in seeds:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(s))
        gens.append(g)
    ids = torch.full((n, 1), bos, dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(int(length)):
            logits = model(ids).logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            cols = [
                torch.multinomial(probs[i], num_samples=1, generator=gens[i])
                for i in range(n)
            ]
            nxt = torch.stack(cols, dim=0).to(device=device)
            ids = torch.cat([ids, nxt], dim=1)
    out_ids = [[int(x) for x in row[1:].tolist()] for row in ids]
    texts = [tok.decode(row, skip_special_tokens=True) for row in out_ids]
    return texts, out_ids


def _build_prefixes(
    n: int,
    length: int,
    *,
    seed: int,
    model_name: str,
    existing_texts: list[str],
    existing_ids: list[list[int]],
    next_offset: int,
    log: bool,
) -> tuple[list[str], list[list[int]], int]:
    """从 ``seed + next_offset`` 起递增采样，直到 ``n`` 条 token 序列互异。"""
    texts, token_ids = _dedupe_keep_first(existing_texts, existing_ids)
    offset = max(int(next_offset), len(token_ids))
    if len(token_ids) >= n:
        return texts[:n], token_ids[:n], offset
    from models import get_hf_model

    need0 = n - len(token_ids)
    if log:
        _train_log(
            f"eval/gen prefix: sampling {need0}+ unique x {length} "
            f"from {model_name} (cpu, seed={seed}+offset) ..."
        )
    device = torch.device("cpu")
    model = get_hf_model(model_name, torch_dtype=torch.float32, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = get_gpt2_tokenizer() if model_name == "gpt2" else None
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    seen = {_row_key(row) for row in token_ids}
    max_offset = offset + max(n * 32, n + 2048)
    skipped = 0
    try:
        while len(token_ids) < n:
            if offset >= max_offset:
                raise RuntimeError(
                    f"gen-eval prefixes: 无法在 seed={seed} offset<{max_offset} "
                    f"内凑满 {n} 条互异前缀（已有 {len(token_ids)}）"
                )
            need = n - len(token_ids)
            seeds = [seed + offset + j for j in range(need)]
            offset += need
            new_texts, new_ids = _sample_prefix_batch(
                model, tok, length, seeds=seeds, device=device,
            )
            for text, ids in zip(new_texts, new_ids):
                if len(ids) != length:
                    raise RuntimeError(
                        f"prefix: expected {length} tokens, got {len(ids)}"
                    )
                key = _row_key(ids)
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)
                texts.append(text)
                token_ids.append(ids)
                if len(token_ids) >= n:
                    break
    finally:
        del model
        gc.collect()
    if log and skipped:
        _train_log(f"eval/gen prefix: skipped {skipped} duplicate draws")
    return texts[:n], token_ids[:n], offset


def ensure_prefix_pool(
    n: int,
    length: int,
    *,
    seed: int,
    model_name: str,
    rank: int,
    is_distributed: bool,
    log: bool,
) -> tuple[list[str], list[list[int]]]:
    """开训前调用：返回 ``n`` 条互异前缀；仅 rank0 读盘/生成，再 broadcast。"""
    if n < 1 or length < 1:
        return [], []
    box: list[Any] = [None]
    if rank == 0:
        path = prefix_cache_path(model=model_name, length=length, seed=seed)
        texts: list[str] = []
        token_ids: list[list[int]] = []
        next_offset = 0
        if path.is_file():
            try:
                texts, token_ids, next_offset = _load_cache(path)
                texts, token_ids = _dedupe_keep_first(texts, token_ids)
                if log and len(token_ids) >= n:
                    _train_log(f"eval/gen prefix: cache {path} n={n} unique")
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                texts, token_ids, next_offset = [], [], 0
        if len(token_ids) < n:
            texts, token_ids, next_offset = _build_prefixes(
                n,
                length,
                seed=seed,
                model_name=model_name,
                existing_texts=texts,
                existing_ids=token_ids,
                next_offset=next_offset,
                log=log,
            )
            _save_cache(
                path,
                model=model_name,
                length=length,
                seed=seed,
                texts=texts,
                token_ids=token_ids,
                next_offset=next_offset,
            )
        n_uniq = len({_row_key(row) for row in token_ids[:n]})
        if n_uniq != n:
            raise RuntimeError(
                f"gen-eval prefixes: 期望 {n} 条互异，实际 {n_uniq}"
            )
        box[0] = {"texts": texts[:n], "token_ids": token_ids[:n]}
    if is_distributed:
        dist.broadcast_object_list(box, src=0)
    payload = box[0]
    if not isinstance(payload, dict):
        raise RuntimeError("prefix pool broadcast failed")
    return list(payload["texts"]), list(payload["token_ids"])


def encode_prefix_batch(
    texts: Sequence[str],
    token_ids: Sequence[Sequence[int]],
    *,
    src_tok_name: str,
    device: torch.device,
) -> torch.Tensor:
    """编成 ``(B, L)``；同源 GPT-2 时直接用缓存 id + BOS，避免重编码漂移。"""
    if len(texts) != len(token_ids):
        raise ValueError("texts/token_ids 条数须一致")
    if not texts:
        raise ValueError("empty prefix batch")
    src_tok = get_src_tokenizer(src_tok_name)
    layout = src_tok.get_token_layout()
    bos = int(layout.bos_token_id)
    rows: list[list[int]] = []
    if src_tok_name == "gpt2":
        for ids in token_ids:
            rows.append([bos, *[int(x) for x in ids]])
    else:
        for text in texts:
            ids = src_tok.encode_preprocess(text)
            rows.append([bos, *list(ids)])
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise RaggedPrefixError(
            f"prefix lengths differ in batch: {sorted(widths)}"
        )
    return torch.tensor(rows, dtype=torch.long, device=device)
