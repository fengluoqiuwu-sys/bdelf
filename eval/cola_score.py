"""TriFluency C′ 护栏：开源 CoLA 分类器可接受概率。"""

from __future__ import annotations

import re
from typing import Any

import torch

COLA_MODEL_ID = "textattack/bert-base-uncased-CoLA"
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """粗切句；过短片段跳过；无句则整段作一句。"""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    parts = [p for p in parts if len(p) >= 3]
    return parts if parts else [text]


def load_cola(
    *,
    device: torch.device,
    torch_dtype: torch.dtype | None = None,
) -> tuple[Any, Any, str]:
    """下载并加载 CoLA；返回 (model, tokenizer, repo_id)。"""
    import os

    import hf_config  # noqa: F401
    from huggingface_hub import snapshot_download
    from models.hf_model import is_hf_model_cached, resolve_hf_model_cache_path
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local = resolve_hf_model_cache_path(COLA_MODEL_ID)
    # 分类器需 tokenizer 文件；比 causal-LM 的 allow_patterns 更宽。
    need_tok = not (local / "tokenizer_config.json").is_file() and not (
        local / "vocab.txt"
    ).is_file()
    if not is_hf_model_cached(COLA_MODEL_ID, local) or need_tok:
        local.mkdir(parents=True, exist_ok=True)
        proxy_keys = (
            "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
            "all_proxy", "ALL_PROXY",
        )
        saved = {k: os.environ.pop(k) for k in proxy_keys if k in os.environ}
        try:
            snapshot_download(
                repo_id=COLA_MODEL_ID,
                local_dir=str(local),
                allow_patterns=[
                    "config.json",
                    "tokenizer*",
                    "vocab.txt",
                    "special_tokens_map.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                    "*.json",
                ],
            )
        finally:
            os.environ.update(saved)

    tok = AutoTokenizer.from_pretrained(str(local))
    kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype
    model = AutoModelForSequenceClassification.from_pretrained(str(local), **kwargs)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, COLA_MODEL_ID


@torch.no_grad()
def score_cola(
    texts: list[str],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 32,
) -> tuple[list[float], dict[str, float]]:
    """每样本：各句「可接受」类概率均值；无句 → nan。

    CoLA label：通常 0=unacceptable, 1=acceptable。
    """
    # 找出 acceptable 类下标
    id2label = getattr(model.config, "id2label", None) or {}
    accept_idx = 1
    for i, name in id2label.items():
        if str(name).lower() in ("acceptable", "1", "label_1"):
            accept_idx = int(i)
            break

    per: list[float] = []
    all_probs: list[float] = []

    for text in texts:
        sents = split_sentences(text)
        if not sents:
            per.append(float("nan"))
            continue
        probs: list[float] = []
        for i in range(0, len(sents), batch_size):
            chunk = sents[i : i + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits.float(), dim=-1)[:, accept_idx]
            probs.extend(float(x) for x in p.detach().cpu())
        mean_p = float(sum(probs) / len(probs))
        per.append(mean_p)
        all_probs.extend(probs)

    import math

    finite = [x for x in per if isinstance(x, float) and math.isfinite(x)]
    summary = {
        "cola_g": float(sum(finite) / len(finite)) if finite else float("nan"),
        "cola_n_scored_samples": float(len(finite)),
        "cola_n_sentences": float(len(all_probs)),
    }
    return per, summary
