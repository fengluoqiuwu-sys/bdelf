"""训练环内 held-out PPL 与在线 gen-eval 胶水。

Gen.PPL 重分词 / 打分原语已上收至 ``eval.gen_ppl``；本模块仅保留
DataLoader / DDP / 采样聚合逻辑，并 re-export 常用符号以兼容旧导入。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval.gen_ppl import (  # noqa: F401 — re-export
    get_gpt2_tokenizer as _get_gpt2_tokenizer,
    get_src_tokenizer as _get_src_tokenizer,
    prepare_gpt2_eval_batch,
    prepare_gpt2_eval_texts,
)
from models import get_hf_model
from preprocess import get_preprocess
from train import FL_TrainConfig
from train.checkpoint import unwrap_model
from train.metrics import _TRAIN_LOG, _train_log, loss_to_ppl


def _gen_eval_local_count(n_total: int, *, rank: int, world_size: int) -> int:
    """把 ``n_total`` 均分到各 rank；余数给前 ``rem`` 个 rank。"""
    if world_size <= 1:
        return n_total
    base, rem = divmod(n_total, world_size)
    return base + (1 if rank < rem else 0)


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
    is_distributed: bool = False,
    log: bool = True,
) -> tuple[float, float]:
    """Eval split loss and exp(loss) PPL from the training model.

    多卡时各 rank 只跑本地分片，再对 ``(sum_loss, n_batches)`` allreduce，
    得到与单卡全量相同的按 batch 均权平均。
    """
    was_training = model.training
    model.eval()
    branch = _eval_loss_branch(model)
    use_amp = device.type == "cuda"
    total_loss = 0.0
    batches = 0

    batch_iter: DataLoader | tqdm = loader
    show_pbar = log and pbar_parent is not None and len(loader) > 0
    if show_pbar:
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
        if show_pbar and pbar_parent is not None:
            pbar_parent.refresh()
        if was_training:
            model.train()

    if is_distributed:
        stats = torch.tensor(
            [total_loss, float(batches)], device=device, dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        batches = int(stats[1].item())

    if batches == 0:
        return float("nan"), float("nan")

    avg_loss = total_loss / batches
    avg_ppl = loss_to_ppl(avg_loss)
    if log:
        label = "decode ce" if branch == "decode" else "loss"
        summary = f"eval: {label} {avg_loss:.4f} ppl {avg_ppl:.2f}"
        if pbar_parent is not None:
            tqdm.write(f"{_TRAIN_LOG} {summary}")
        else:
            _train_log(summary)
    return avg_loss, avg_ppl


def _gen_eval_sampling_cfg(cfg: FL_TrainConfig) -> dict[str, Any]:
    """在线 gen-eval 采样参数来自 ``--generate`` 指定的 generate 配置。"""
    return dict(cfg.generate_sampling)


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
    seed: int,
    pbar_parent: tqdm | None = None,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False,
    log: bool = True,
) -> tuple[float, float, float, float]:
    """Unconditional gen. PPL: sample with train model, score via gpt2-large.

    全局共 ``cfg.gen_eval_samples`` 条；多卡时各 rank 分担采样与打分，再按
    token 加权聚合 loss、按样本聚合 uniq / nonempty。

    Returns ``(gen_loss, gen_ppl, gen_uniq_mean, gen_nonempty_frac)``. The last
    two catch mode-collapse that can fake a very low gen_ppl (e.g. repeated ``/``).
    """
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
    n_total = int(cfg.gen_eval_samples)
    n_local = _gen_eval_local_count(n_total, rank=rank, world_size=world_size)
    micro_bs = max(1, int(cfg.batch_size))
    local_seed = seed * max(1, world_size) + rank

    if log and pbar_parent is not None:
        pbar_parent.clear()
        tqdm.write(
            f"{_TRAIN_LOG} eval/gen: sampling {n_total} x {seqlen} "
            f"(world={world_size}, local={n_local}, micro_bs={micro_bs}, "
            f"seed={local_seed}) ...",
        )

    # Isolate sampling RNG from the training loop.
    devices = [train_device] if train_device.type == "cuda" else []
    uniq_sum = 0.0
    nonempty_count = 0
    loss_sum = 0.0
    token_sum = 0
    skipped_local = 0

    if n_local > 0:
        chunks: list[torch.Tensor] = []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(local_seed)
            if train_device.type == "cuda":
                torch.cuda.manual_seed_all(local_seed)
            gen_model = unwrap_model(train_model)
            sampling_cfg = _gen_eval_sampling_cfg(cfg)
            with torch.amp.autocast(
                "cuda", dtype=train_amp_dtype, enabled=use_train_amp,
            ):
                remaining = n_local
                while remaining > 0:
                    this_bs = min(micro_bs, remaining)
                    generated, _nfe = gen_model.generate(
                        num_samples=this_bs,
                        seqlen=seqlen,
                        for_eval=True,
                        sampling_cfg=sampling_cfg,
                    )
                    chunks.append(generated.detach())
                    remaining -= this_bs
        generated = torch.cat(chunks, dim=0)
        assert generated.size(0) == n_local

        # Collapse diagnostics on token ids (before skip_special decode).
        uniq_counts = [
            float(row.unique().numel()) for row in generated.detach().cpu()
        ]
        uniq_sum = float(sum(uniq_counts))

        src_tok_name = get_preprocess(cfg.preprocess).tokenizer
        src_tok = _get_src_tokenizer(src_tok_name)
        texts = [
            src_tok.decode(row.tolist(), skip_special_tokens=True)
            for row in generated.detach().cpu()
        ]
        # Match official ELF: score only nonempty decoded strings.
        nonempty = [t for t in texts if isinstance(t, str) and t.strip()]
        skipped_local = len(texts) - len(nonempty)
        nonempty_count = len(nonempty)

        if nonempty:
            # Score in micro-batches to keep gpt2 peak memory near one train batch.
            score_bs = max(1, int(cfg.batch_size))
            with torch.amp.autocast(
                "cuda", dtype=gpt2_amp_dtype, enabled=use_gpt2_amp,
            ):
                for i in range(0, len(nonempty), score_bs):
                    chunk = nonempty[i : i + score_bs]
                    input_ids, labels, attention_mask = prepare_gpt2_eval_texts(
                        chunk,
                        gpt2_vocab_size=gpt2_vocab_size,
                        fill_token_id=fill_token_id,
                        device=gpt2_device,
                        max_length=seqlen,
                    )
                    outputs = gpt2_model(
                        input_ids, attention_mask=attention_mask, labels=labels,
                    )
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    n_tok = int((labels != -100).sum().item())
                    if n_tok > 0:
                        loss_sum += float(loss.item()) * n_tok
                        token_sum += n_tok

    # [loss_sum, token_sum, uniq_sum, n_local, nonempty_count, skipped]
    stats = torch.tensor(
        [
            loss_sum,
            float(token_sum),
            uniq_sum,
            float(n_local),
            float(nonempty_count),
            float(skipped_local),
        ],
        device=train_device,
        dtype=torch.float64,
    )
    if is_distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    loss_sum = float(stats[0].item())
    token_sum = int(stats[1].item())
    uniq_sum = float(stats[2].item())
    n_generated = int(stats[3].item())
    nonempty_total = int(stats[4].item())
    skipped = int(stats[5].item())

    gen_uniq_mean = uniq_sum / max(n_generated, 1)
    gen_nonempty_frac = nonempty_total / max(n_generated, 1)

    if was_training:
        train_model.train()
    if log and pbar_parent is not None:
        pbar_parent.refresh()

    if log and skipped > 0:
        msg = f"eval/gen: skipped {skipped}/{n_generated} empty samples"
        if pbar_parent is not None:
            tqdm.write(f"{_TRAIN_LOG} {msg}")
        else:
            _train_log(msg)

    if nonempty_total == 0 or token_sum == 0:
        gen_loss = float("nan")
        gen_ppl = float("nan")
        if nonempty_total == 0:
            reason = "all samples empty"
        else:
            reason = "no scorable tokens"
        summary = (
            f"eval/gen ({cfg.gen_eval_model}): {reason}; "
            f"loss nan ppl nan uniq_mean={gen_uniq_mean:.1f}"
        )
    else:
        gen_loss = loss_sum / token_sum
        gen_ppl = loss_to_ppl(gen_loss)
        summary = (
            f"eval/gen ({cfg.gen_eval_model}): loss {gen_loss:.4f} "
            f"ppl {gen_ppl:.2f} (n={nonempty_total} "
            f"uniq_mean={gen_uniq_mean:.1f} nonempty={gen_nonempty_frac:.2f})"
        )

    if log:
        if pbar_parent is not None:
            tqdm.write(f"{_TRAIN_LOG} {summary}")
        else:
            _train_log(summary)
    return gen_loss, gen_ppl, gen_uniq_mean, gen_nonempty_frac
