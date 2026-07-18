"""Eval-split PPL and one-batch generative PPL scoring."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_hf_model
from models.tokens import FL_TokenLayout
from preprocess import get_preprocess
from train import FL_TrainConfig
from train.checkpoint import unwrap_model
from train.metrics import _TRAIN_LOG, _train_log, loss_to_ppl


def get_amp_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def uses_full_sequence(model: nn.Module) -> bool:
    return getattr(unwrap_model(model), "full_sequence_training", False)


def uses_dual_branch_logging(model: nn.Module) -> bool:
    return getattr(unwrap_model(model), "dual_branch_logging", False)


def forward_loss(
    model: nn.Module,
    batch: torch.Tensor,
    *,
    branch: str | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {}
    if branch is not None:
        if not uses_dual_branch_logging(model):
            raise ValueError(f"Model does not support branch={branch!r}")
        kwargs["branch"] = branch
    if uses_full_sequence(model):
        _, loss = model(batch, None, **kwargs)
    else:
        _, loss = model(batch[:, :-1], batch[:, 1:], **kwargs)
    return loss


def _eval_loss_branch(model: nn.Module) -> str | None:
    """BDELF/ELF eval uses decode CE; AR/BD3LM use the default training loss."""
    if uses_dual_branch_logging(model):
        return "decode"
    return None


@torch.no_grad()
def eval_model_ppl(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    pbar_parent: tqdm | None = None,
) -> tuple[float, float]:
    """Eval split loss and exp(loss) PPL from the training model."""
    was_training = model.training
    model.eval()
    branch = _eval_loss_branch(model)
    use_amp = device.type == "cuda"
    total_loss = 0.0
    batches = 0
    if len(loader) == 0:
        return float("nan"), float("nan")

    batch_iter: DataLoader | tqdm = loader
    if pbar_parent is not None:
        pbar_parent.clear()
        batch_iter = tqdm(
            loader,
            desc="eval",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            total=len(loader),
        )
    try:
        for eval_batch in batch_iter:
            eval_batch = eval_batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = forward_loss(model, eval_batch, branch=branch)
            total_loss += float(loss.item())
            batches += 1
    finally:
        if isinstance(batch_iter, tqdm):
            batch_iter.close()
        if pbar_parent is not None:
            pbar_parent.refresh()
        if was_training:
            model.train()

    avg_loss = total_loss / max(1, batches)
    avg_ppl = loss_to_ppl(avg_loss)
    if batches > 0:
        label = "decode ce" if branch == "decode" else "loss"
        summary = f"eval: {label} {avg_loss:.4f} ppl {avg_ppl:.2f}"
        if pbar_parent is not None:
            tqdm.write(f"{_TRAIN_LOG} {summary}")
        else:
            _train_log(summary)
    return avg_loss, avg_ppl


def prepare_gpt2_eval_batch(
    batch: torch.Tensor,
    layout: FL_TokenLayout,
    *,
    gpt2_vocab_size: int,
    fill_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map extended-vocab token ids into the GPT-2 baseline range for CE/PPL."""
    input_ids = batch.clone()
    labels = batch.clone()
    oov = input_ids >= gpt2_vocab_size
    input_ids[oov] = fill_token_id
    for token_id in (layout.bos_token_id, layout.eos_token_id, layout.pad_token_id):
        labels[labels == token_id] = -100
    return input_ids, labels


def prepare_gpt2_eval_batch_retokenize(
    batch: torch.Tensor,
    *,
    src_tokenizer_name: str,
    gpt2_vocab_size: int,
    fill_token_id: int,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode with the train tokenizer, then re-encode with GPT-2 for Gen. PPL.

    Required when the train model uses a non-GPT-2 vocabulary (e.g. ELF / T5).
    """
    from transformers import AutoTokenizer

    from tokenizer import get_tokenizer

    src_tok = get_tokenizer(src_tokenizer_name)
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    if gpt2_tok.pad_token_id is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token

    texts = [
        src_tok.decode(row.tolist(), skip_special_tokens=True)
        for row in batch.detach().cpu()
    ]
    encoded = gpt2_tok(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    labels = input_ids.clone()
    pad_id = int(gpt2_tok.pad_token_id)
    labels[labels == pad_id] = -100
    oov = input_ids >= gpt2_vocab_size
    input_ids[oov] = fill_token_id
    return input_ids, labels


def _gen_eval_sampling_cfg(cfg: FL_TrainConfig) -> dict[str, Any]:
    sampling_cfg: dict[str, Any] = {"use_fast_infer": cfg.eval_use_fast_infer}
    if cfg.model == "bd3lm":
        sampling_cfg["num_steps"] = cfg.eval_gen_steps
    elif cfg.model == "elf":
        # Keep eval sampling lighter than the default 32–64-step SDE.
        sampling_cfg["num_sampling_steps"] = min(16, cfg.eval_gen_steps)
        sampling_cfg["sampling_method"] = "ode"
        sampling_cfg["temperature"] = 0.0  # paper decode: argmax
    return sampling_cfg


def load_gen_eval_baseline(cfg: FL_TrainConfig) -> nn.Module:
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[cfg.gen_eval_model_dtype]
    device = cfg.gen_eval_model_device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("gen_eval_model_device=cuda but no CUDA device was found")
    model = get_hf_model(cfg.gen_eval_model, torch_dtype=torch_dtype, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def eval_one_batch_gen_ppl(
    train_model: nn.Module,
    gpt2_model: nn.Module,
    *,
    cfg: FL_TrainConfig,
    train_device: torch.device,
    train_amp_dtype: torch.dtype,
    token_layout: FL_TokenLayout,
    seed: int,
    pbar_parent: tqdm | None = None,
) -> tuple[float, float]:
    """Unconditional one-batch gen. PPL: sample with train model, score via gpt2-large."""
    was_training = train_model.training
    train_model.eval()
    gpt2_model.eval()
    gpt2_device = next(gpt2_model.parameters()).device
    gpt2_vocab_size = int(getattr(gpt2_model.config, "vocab_size", 50257))
    fill_token_id = int(
        getattr(gpt2_model.config, "eos_token_id", None) or 50256,
    )
    seqlen = int(cfg.extra.get("chunk_length", 1024))
    use_train_amp = train_device.type == "cuda"
    use_gpt2_amp = gpt2_device.type == "cuda"
    gpt2_amp_dtype = get_amp_dtype(cfg.gen_eval_model_dtype)

    if pbar_parent is not None:
        pbar_parent.clear()
        tqdm.write(
            f"{_TRAIN_LOG} eval/gen: sampling {cfg.batch_size} x {seqlen} "
            f"(seed={seed}) ...",
        )

    # Isolate sampling RNG from the training loop.
    devices = [train_device] if train_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if train_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        gen_model = unwrap_model(train_model)
        with torch.amp.autocast(
            "cuda", dtype=train_amp_dtype, enabled=use_train_amp,
        ):
            generated, _nfe = gen_model.generate(
                num_samples=cfg.batch_size,
                seqlen=seqlen,
                for_eval=True,
                sampling_cfg=_gen_eval_sampling_cfg(cfg),
            )

    if cfg.model == "elf":
        src_tok_name = get_preprocess(cfg.preprocess).tokenizer
        input_ids, labels = prepare_gpt2_eval_batch_retokenize(
            generated,
            src_tokenizer_name=src_tok_name,
            gpt2_vocab_size=gpt2_vocab_size,
            fill_token_id=fill_token_id,
            device=gpt2_device,
            max_length=seqlen,
        )
    else:
        generated = generated.to(gpt2_device, non_blocking=True)
        input_ids, labels = prepare_gpt2_eval_batch(
            generated,
            token_layout,
            gpt2_vocab_size=gpt2_vocab_size,
            fill_token_id=fill_token_id,
        )
    with torch.amp.autocast("cuda", dtype=gpt2_amp_dtype, enabled=use_gpt2_amp):
        outputs = gpt2_model(input_ids, labels=labels)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        gen_loss = float(loss.item())
    gen_ppl = loss_to_ppl(gen_loss)

    if was_training:
        train_model.train()
    if pbar_parent is not None:
        pbar_parent.refresh()

    summary = (
        f"eval/gen ({cfg.gen_eval_model}): loss {gen_loss:.4f} ppl {gen_ppl:.2f}"
    )
    if pbar_parent is not None:
        tqdm.write(f"{_TRAIN_LOG} {summary}")
    else:
        _train_log(summary)
    return gen_loss, gen_ppl
