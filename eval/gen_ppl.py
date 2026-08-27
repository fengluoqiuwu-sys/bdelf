"""Gen.PPL 与 unigram entropy 权威实现（论文 / ELF metrics_utils 口径）。

训练在线 gen-eval（``train.eval.eval_one_batch_gen_ppl``）与离线 TriFluency
均从此模块取重分词与打分原语。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from train.metrics import loss_to_ppl

# 进程内 tokenizer 缓存
_SRC_TOKENIZER_CACHE: dict[str, Any] = {}
_GPT2_TOKENIZER: Any | None = None


def content_ids_for_eval(
    token_ids: Sequence[int],
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> list[int]:
    """首个 EOS 截断（不含 EOS）；丢掉 BOS / PAD。

    在线 gen-eval 的 uniq / 解码 / GPT-2 打分都走这条内容序列，
    避免停机后的 PAD 或首尾特殊符灌进指标。
    """
    skip = {int(bos_id), int(pad_id)}
    eos = int(eos_id)
    out: list[int] = []
    for raw in token_ids:
        tid = int(raw)
        if tid == eos:
            break
        if tid in skip:
            continue
        out.append(tid)
    return out


def decode_eval_texts(
    rows: Sequence[Any],
    src_tok: Any,
) -> tuple[list[str], list[float], list[int]]:
    """按内容 token 解码；返回 uniq 与内容长度（均不含 BOS/EOS/PAD）。"""
    layout = src_tok.get_token_layout()
    bos_id = int(layout.bos_token_id)
    eos_id = int(layout.eos_token_id)
    pad_id = int(layout.pad_token_id)
    texts: list[str] = []
    uniq: list[float] = []
    lengths: list[int] = []
    for row in rows:
        ids = row.tolist() if hasattr(row, "tolist") else list(row)
        content = content_ids_for_eval(
            ids, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
        )
        lengths.append(len(content))
        uniq.append(float(len(set(content))) if content else 0.0)
        texts.append(
            src_tok.decode(content, skip_special_tokens=True) if content else ""
        )
    return texts, uniq, lengths


def get_src_tokenizer(name: str) -> Any:
    """缓存并返回训练侧 tokenizer（按 preprocess / 模型配置名）。"""
    tok = _SRC_TOKENIZER_CACHE.get(name)
    if tok is None:
        from tokenizer import get_tokenizer

        tok = get_tokenizer(name)
        _SRC_TOKENIZER_CACHE[name] = tok
    return tok


def get_gpt2_tokenizer() -> Any:
    """与 ``config/tokenizers/gpt2`` 对齐的 GPT-2 tokenizer（本地 cache）。"""
    global _GPT2_TOKENIZER
    if _GPT2_TOKENIZER is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir="cache/tokenizers/gpt2",
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        _GPT2_TOKENIZER = tok
    return _GPT2_TOKENIZER


def prepare_gpt2_eval_texts(
    texts: list[str],
    *,
    gpt2_vocab_size: int,
    fill_token_id: int,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将解码文本用 GPT-2 重分词，供 Gen.PPL 打分。

    调用方应已去掉源侧 BOS/EOS/PAD。此处 ``add_special_tokens=False``，
    且 pad 位 label=-100（GPT-2 的 pad_id 等于 eos_id，不能按 id 掩）。
    返回 ``(input_ids, labels, attention_mask)``。
    """
    gpt2_tok = get_gpt2_tokenizer()
    encoded = gpt2_tok(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    # GPT-2 pad_id == eos_id，不能用 id 相等掩 pad。
    labels[attention_mask == 0] = -100
    oov = input_ids >= gpt2_vocab_size
    input_ids[oov] = fill_token_id
    return input_ids, labels, attention_mask


def prepare_gpt2_eval_batch(
    batch: torch.Tensor,
    *,
    src_tokenizer_name: str,
    gpt2_vocab_size: int,
    fill_token_id: int,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """训练 tokenizer 解码后再 GPT-2 编码（在线 gen-eval 路径）。"""
    src_tok = get_src_tokenizer(src_tokenizer_name)
    texts, _uniq, _lens = decode_eval_texts(batch.detach().cpu(), src_tok)
    return prepare_gpt2_eval_texts(
        texts,
        gpt2_vocab_size=gpt2_vocab_size,
        fill_token_id=fill_token_id,
        device=device,
        max_length=max_length,
    )


def unigram_entropy(token_ids: np.ndarray) -> float:
    """valid GPT-2 token id 上的经验 unigram Shannon 熵（nat）。"""
    if token_ids.size == 0:
        return float("nan")
    _, counts = np.unique(token_ids, return_counts=True)
    probs = counts.astype(np.float64) / float(counts.sum())
    return float(-np.sum(probs * np.log(probs + 1e-10)))


@torch.no_grad()
def score_texts(
    texts: list[str],
    *,
    gpt2_model: torch.nn.Module,
    max_length: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[list[float], list[float], float]:
    """逐条 Gen.PPL + unigram entropy；返回 (per_ppl, per_ent, corpus_ppl)。"""
    gpt2_vocab_size = int(getattr(gpt2_model.config, "vocab_size", 50257))
    fill_token_id = int(getattr(gpt2_model.config, "eos_token_id", None) or 50256)
    use_amp = device.type == "cuda"

    per_ppl: list[float] = []
    per_ent: list[float] = []
    loss_sum = 0.0
    token_sum = 0

    for text in texts:
        if not (isinstance(text, str) and text.strip()):
            per_ppl.append(float("nan"))
            per_ent.append(float("nan"))
            continue

        input_ids, labels, attention_mask = prepare_gpt2_eval_texts(
            [text],
            gpt2_vocab_size=gpt2_vocab_size,
            fill_token_id=fill_token_id,
            device=device,
            max_length=max_length,
        )
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = gpt2_model(
                input_ids, attention_mask=attention_mask, labels=labels,
            )
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)
            valid = shift_labels != -100
            n_tok = int(valid.sum().item())
            if n_tok > 0:
                sample_loss = float((nll * valid).sum().item() / n_tok)
                per_ppl.append(loss_to_ppl(sample_loss))
                loss_sum += sample_loss * n_tok
                token_sum += n_tok
            else:
                per_ppl.append(float("nan"))

        ids_np = input_ids[0].detach().cpu().numpy()
        mask_np = attention_mask[0].detach().cpu().numpy()
        valid_len = int(mask_np.sum())
        per_ent.append(unigram_entropy(ids_np[:valid_len]))
        del outputs, logits, shift_logits, shift_labels, nll, valid, loss

    corpus_ppl = (
        loss_to_ppl(loss_sum / token_sum) if token_sum > 0 else float("nan")
    )
    return per_ppl, per_ent, corpus_ppl
