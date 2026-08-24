"""在线 Eval 管线：HeldOut 永写主表 + 可插组件 + 共享样本落盘。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from eval.gen_ppl import (
    get_src_tokenizer as _get_src_tokenizer,
    prepare_gpt2_eval_texts,
    score_texts,
)
from preprocess import get_preprocess
from models import kind_of
from train import FL_TrainConfig
from train.checkpoint import unwrap_model
from train.eval import (
    _gen_eval_local_count,
    _gen_eval_sampling_cfg,
    eval_model_ppl,
    get_amp_dtype,
    release_eval_cuda_scratch,
)
from train.metrics import (
    EVAL_OFFICIAL_FIELDS,
    EVAL_SAMPLE_BASE_FIELDS,
    _TRAIN_LOG,
    _train_log,
    append_csv_row,
    eval_csv_fields,
    loss_to_ppl,
)
from train.run_logs import (
    eval_official_csv,
    load_eval_tick,
    sample_step_dir,
    save_eval_tick,
)


@dataclass
class EvalComponentSpec:
    """模型代码写死的在线 eval 组件说明（不含 HeldOut）。"""

    name: str
    every_n_evals: int = 1
    needs_samples: bool = False
    official: bool = True


DEFAULT_ONLINE_EVAL_COMPONENTS: tuple[EvalComponentSpec, ...] = (
    EvalComponentSpec("gen_ppl", every_n_evals=1, needs_samples=True, official=True),
    EvalComponentSpec("entropy", every_n_evals=1, needs_samples=True, official=True),
    EvalComponentSpec("dist1", every_n_evals=1, needs_samples=True, official=True),
)


def resolve_online_eval_components(model: nn.Module) -> list[EvalComponentSpec]:
    """模型覆写 ``online_eval_components``；``None``/缺省 → 默认官方三项。"""
    raw = unwrap_model(model)
    fn = getattr(raw, "online_eval_components", None)
    if callable(fn):
        specs = fn()
        if specs is None:
            return list(DEFAULT_ONLINE_EVAL_COMPONENTS)
        out: list[EvalComponentSpec] = []
        for item in specs:
            if isinstance(item, EvalComponentSpec):
                out.append(item)
            elif isinstance(item, dict):
                out.append(EvalComponentSpec(**item))
            else:
                raise TypeError(
                    f"online_eval_components entries must be "
                    f"EvalComponentSpec or dict, got {type(item)}"
                )
        return out
    bb = getattr(raw, "backbone", None)
    if bb is not None:
        fn_bb = getattr(bb, "online_eval_components", None)
        if callable(fn_bb):
            specs = fn_bb()
            if specs is None:
                return list(DEFAULT_ONLINE_EVAL_COMPONENTS)
            return [
                s if isinstance(s, EvalComponentSpec) else EvalComponentSpec(**s)
                for s in specs
            ]
    return list(DEFAULT_ONLINE_EVAL_COMPONENTS)


def _component_due(spec: EvalComponentSpec, tick: int) -> bool:
    n = max(1, int(spec.every_n_evals))
    # tick 从 1 起：第 1、1+n、… 次到期
    return ((tick - 1) % n) == 0


@dataclass
class SharedGenBatch:
    """一次生成的共享样本（全局汇总后）。"""

    texts: list[str]
    uniq_counts: list[float]
    seed: int
    seqlen: int


@dataclass
class SampleScoreSheet:
    """逐条分数；可被外部组件加列。"""

    gen_ppl: list[float]
    entropy: list[float]
    gen_loss_corpus: float
    gen_ppl_corpus: float
    extra_columns: dict[str, list[Any]] = field(default_factory=dict)


def _gather_object_list(local: list[Any], *, is_distributed: bool) -> list[Any]:
    if not is_distributed:
        return list(local)
    gathered: list[list[Any]] | None = [None] * dist.get_world_size()  # type: ignore[list-item]
    dist.all_gather_object(gathered, local)
    out: list[Any] = []
    assert gathered is not None
    for part in gathered:
        out.extend(part)
    return out


@torch.compiler.disable
@torch.no_grad()
def generate_shared_samples(
    train_model: nn.Module,
    *,
    cfg: FL_TrainConfig,
    train_device: torch.device,
    train_amp_dtype: torch.dtype,
    seed: int,
    rank: int,
    world_size: int,
    is_distributed: bool,
    pbar_parent: tqdm | None,
    log: bool,
) -> SharedGenBatch:
    """各卡分担生成，再汇总 texts / uniq（顺序：rank0, rank1, …）。

    只切 unwrap 后原模块的 ``eval()``，不动 DDP/compile 外壳，避免 Dynamo
    为 eval 模式另编一张图。
    """
    raw = unwrap_model(train_model)
    was_training = raw.training
    raw.eval()
    raw = unwrap_model(train_model)
    was_training = raw.training
    raw.eval()
    try:
        return _generate_shared_samples_body(
            raw,
            cfg=cfg,
            train_device=train_device,
            train_amp_dtype=train_amp_dtype,
            seed=seed,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
            pbar_parent=pbar_parent,
            log=log,
        )
    finally:
        if was_training:
            raw.train()


def _generate_shared_samples_body(
    n_local = _gen_eval_local_count(n_total, rank=rank, world_size=world_size)
    micro_bs = max(1, int(cfg.batch_size))
    local_seed = seed * max(1, world_size) + rank
    use_train_amp = train_device.type == "cuda"

    if log and pbar_parent is not None:
        pbar_parent.clear()
        tqdm.write(
            f"{_TRAIN_LOG} eval/gen: sampling {n_total} x {seqlen} "
            f"(world={world_size}, local={n_local}, micro_bs={micro_bs}, "
            f"seed={local_seed}) ...",
        )

    devices = [train_device] if train_device.type == "cuda" else []
    local_texts: list[str] = []
    local_uniq: list[float] = []

    if n_local > 0:
        chunks: list[torch.Tensor] = []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(local_seed)
            if train_device.type == "cuda":
                torch.cuda.manual_seed_all(local_seed)
            gen_model = raw
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
        local_uniq = [
            float(row.unique().numel()) for row in generated.detach().cpu()
        ]
        src_tok_name = get_preprocess(cfg.preprocess).tokenizer
        src_tok = _get_src_tokenizer(src_tok_name)
        local_texts = [
            src_tok.decode(row.tolist(), skip_special_tokens=True)
            for row in generated.detach().cpu()
        ]

    texts = _gather_object_list(local_texts, is_distributed=is_distributed)
    uniq_counts = _gather_object_list(local_uniq, is_distributed=is_distributed)
    # 截到 n_total（余数分配可能导致恰好 n_total）
    texts = texts[:n_total]
    uniq_counts = uniq_counts[:n_total]

    if was_training:
        raw.train()
    if log and pbar_parent is not None:
        pbar_parent.refresh()

    return SharedGenBatch(
        texts=texts, uniq_counts=uniq_counts, seed=seed, seqlen=seqlen,
    )


@torch.no_grad()
def score_shared_samples(
    batch: SharedGenBatch,
    gpt2_model: nn.Module,
    *,
    cfg: FL_TrainConfig,
    amp_dtype: torch.dtype,
) -> SampleScoreSheet:
    """在共享文本上算逐条 gen_ppl / entropy 与语料级 loss/ppl。"""
    device = next(gpt2_model.parameters()).device
    per_ppl, per_ent, corpus_ppl = score_texts(
        batch.texts,
        gpt2_model=gpt2_model,
        max_length=batch.seqlen,
        device=device,
        amp_dtype=amp_dtype,
    )
    # corpus loss：从 nonempty 再聚一次 token 加权（与旧 eval_one_batch 对齐）
    gpt2_vocab_size = int(getattr(gpt2_model.config, "vocab_size", 50257))
    fill_token_id = int(getattr(gpt2_model.config, "eos_token_id", None) or 50256)
    loss_sum = 0.0
    token_sum = 0
    nonempty = [t for t in batch.texts if isinstance(t, str) and t.strip()]
    use_amp = device.type == "cuda"
    score_bs = max(1, int(cfg.batch_size))
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        for i in range(0, len(nonempty), score_bs):
            chunk = nonempty[i : i + score_bs]
            input_ids, labels, attention_mask = prepare_gpt2_eval_texts(
                chunk,
                gpt2_vocab_size=gpt2_vocab_size,
                fill_token_id=fill_token_id,
                device=device,
                max_length=batch.seqlen,
            )
            outputs = gpt2_model(
                input_ids, attention_mask=attention_mask, labels=labels,
            )
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            n_tok = int((labels != -100).sum().item())
            if n_tok > 0:
                loss_sum += float(loss.item()) * n_tok
                token_sum += n_tok
    gen_loss = loss_sum / token_sum if token_sum > 0 else float("nan")
    gen_ppl = loss_to_ppl(gen_loss) if token_sum > 0 else float("nan")
    # corpus_ppl from score_texts is equivalent when nonempty-only; prefer token-weighted
    _ = corpus_ppl
    return SampleScoreSheet(
        gen_ppl=per_ppl,
        entropy=per_ent,
        gen_loss_corpus=gen_loss,
        gen_ppl_corpus=gen_ppl,
    )


def _corpus_dist1(texts: Sequence[str], gpt2_max_length: int) -> float:
    """语料级 Distinct-1（GPT-2 分词 unigram）。"""
    from eval.gen_ppl import get_gpt2_tokenizer

    tok = get_gpt2_tokenizer()
    all_ids: list[int] = []
    for text in texts:
        if not (isinstance(text, str) and text.strip()):
            continue
        encoded = tok(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=gpt2_max_length,
            return_tensors=None,
        )
        ids = encoded["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        all_ids.extend(int(x) for x in ids)
    if not all_ids:
        return float("nan")
    return float(len(set(all_ids))) / float(len(all_ids))


def _mean_finite(values: Sequence[float]) -> float:
    arr = [v for v in values if v == v]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def write_sample_dir(
    run_dir: Path,
    step: int,
    batch: SharedGenBatch,
    scores: SampleScoreSheet,
    *,
    meta_extra: dict[str, Any] | None = None,
) -> Path:
    """落盘 ``eval_samples/step_*/{meta.json,samples.csv}``。"""
    out_dir = sample_step_dir(run_dir, step)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "step": step,
        "n": len(batch.texts),
        "seed": batch.seed,
        "seqlen": batch.seqlen,
        **(meta_extra or {}),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    extra_keys = list(scores.extra_columns.keys())
    fields = list(EVAL_SAMPLE_BASE_FIELDS) + [
        k for k in extra_keys if k not in EVAL_SAMPLE_BASE_FIELDS
    ]
    path = out_dir / "samples.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, text in enumerate(batch.texts):
            row: dict[str, Any] = {
                "id": i,
                "text": text,
                "gen_ppl": (
                    round(scores.gen_ppl[i], 4)
                    if i < len(scores.gen_ppl) and scores.gen_ppl[i] == scores.gen_ppl[i]
                    else ""
                ),
                "entropy": (
                    round(scores.entropy[i], 6)
                    if i < len(scores.entropy) and scores.entropy[i] == scores.entropy[i]
                    else ""
                ),
            }
            for k in extra_keys:
                col = scores.extra_columns[k]
                row[k] = col[i] if i < len(col) else ""
            writer.writerow(row)
    return out_dir


def run_online_eval(
    model: nn.Module,
    *,
    cfg: FL_TrainConfig,
    run_dir: Path,
    step: int,
    lr: float,
    eval_loader: Any,
    gpt2_model: nn.Module | None,
    device: torch.device,
    amp_dtype: torch.dtype,
    rank: int,
    world_size: int,
    is_distributed: bool,
    pbar_parent: tqdm | None,
    ema_state: Any,
    swap_ema_weights: Callable[..., Any],
    curriculum_eval_ctx: Any | None = None,
    curriculum_sampler: Any | None = None,
    eval_tokens: int | None = None,
    latent_probe_pool: Dataset | None = None,
    latent_pad_token_id: int | None = None,
    log_step: int | None = None,
    log_stage: str | None = None,
) -> None:
    """HeldOut 永写主表；按 tick 跑到期组件；共享生成至多一次。"""
    is_latent = kind_of(cfg.model) == "latent"
    eval_fields = eval_csv_fields(cfg.model, cfg)
    csv_step = step if log_step is None else int(log_step)
    csv_stage = "" if log_stage is None else str(log_stage)
    tick = load_eval_tick(run_dir) + 1
    components = resolve_online_eval_components(model)
    due = [c for c in components if _component_due(c, tick)]
    need_samples = not is_latent and any(c.needs_samples for c in due)
    data_tokens = (
        int(eval_tokens)
        if eval_tokens is not None
        else cfg.tokens_seen_after_step(step)
    )

    curriculum_eval_row: dict[str, Any] | None = None
    eval_loss = float("nan")
    eval_ppl = float("nan")
    batch: SharedGenBatch | None = None
    scores: SampleScoreSheet | None = None

    if (
        is_latent
        and cfg.extra.get("curriculum")
        and curriculum_eval_ctx is not None
        and curriculum_sampler is not None
    ):
        from train.latent_eval import run_latent_curriculum_eval

        # copy_ 进同一 Parameter，前向打编译模块（对齐训练 graph_l）。
        with swap_ema_weights(model, ema_state):
            curriculum_eval_row = run_latent_curriculum_eval(
                model,
                ctx=curriculum_eval_ctx,
                sampler=curriculum_sampler,
                step=csv_step,
                tokens=data_tokens,
                lr=lr,
                device=device,
                amp_dtype=amp_dtype,
                eval_sample_seed=cfg.eval_sample_seed,
                rank=rank,
                world_size=world_size,
                is_distributed=is_distributed,
                pbar_parent=pbar_parent,
                log=(rank == 0),
            )
        release_eval_cuda_scratch(
            model, log=(rank == 0), empty_cache=False,
        )
        if csv_stage:
            curriculum_eval_row["curriculum_stage"] = csv_stage
    else:
        from train.ema import using_ema_weights

        with using_ema_weights(model, ema_state):
            eval_loss, eval_ppl = eval_model_ppl(
                unwrap_model(model),
                eval_loader,
                device,
                amp_dtype,
                pbar_parent=pbar_parent,
                is_distributed=is_distributed,
                log=(rank == 0),
            )

            if need_samples and gpt2_model is not None:
                try:
                    batch = generate_shared_samples(
                        model,
                        cfg=cfg,
                        train_device=device,
                        train_amp_dtype=amp_dtype,
                        seed=cfg.seed + step,
                        rank=rank,
                        world_size=world_size,
                        is_distributed=is_distributed,
                        pbar_parent=pbar_parent,
                        log=(rank == 0),
                    )
                    if rank == 0:
                        gpt2_amp = get_amp_dtype(cfg.gen_eval_model_dtype)
                        scores = score_shared_samples(
                            batch, gpt2_model, cfg=cfg, amp_dtype=gpt2_amp,
                        )
                    if is_distributed:
                        payload: list[Any] = [scores]
                        dist.broadcast_object_list(payload, src=0)
                        scores = payload[0]
                except Exception as exc:  # noqa: BLE001 — 组件失败不中断训练
                    if rank == 0:
                        _train_log(f"eval/gen failed (skip samples): {exc}")
                    batch = None
                    scores = None
            elif need_samples and gpt2_model is None and rank == 0:
                _train_log("eval/gen skipped: no gpt2 baseline")

        release_eval_cuda_scratch(
            model, log=(rank == 0), empty_cache=False,
        )

    if is_distributed:
        dist.barrier()

    if rank == 0:
        if curriculum_eval_row is not None:
            append_csv_row(run_dir / "eval_log.csv", eval_fields, curriculum_eval_row)
            save_eval_tick(run_dir, tick)
        else:
            eval_row: dict[str, Any] = {
                "step": csv_step,
                "tokens": data_tokens,
                "lr": lr,
                "eval_loss": round(eval_loss, 6) if eval_loss == eval_loss else "",
            }
            if is_latent:
                eval_row["curriculum_stage"] = csv_stage
            if not is_latent:
                eval_row["eval_ppl"] = (
                    round(eval_ppl, 4) if eval_ppl == eval_ppl else ""
                )
            append_csv_row(run_dir / "eval_log.csv", eval_fields, eval_row)

            if is_latent:
                save_eval_tick(run_dir, tick)
            else:
                due_names = {c.name for c in due if c.official}
                off: dict[str, Any] = {k: "" for k in EVAL_OFFICIAL_FIELDS}
                off["step"] = step
                wrote_official = False

                if batch is not None and scores is not None:
                    n = len(batch.texts)
                    nonempty = sum(
                        1 for t in batch.texts if isinstance(t, str) and t.strip()
                    )
                    uniq_mean = (
                        float(sum(batch.uniq_counts) / max(n, 1))
                        if batch.uniq_counts
                        else float("nan")
                    )
                    nonempty_frac = nonempty / max(n, 1)

                    if "gen_ppl" in due_names:
                        if scores.gen_loss_corpus == scores.gen_loss_corpus:
                            off["gen_loss"] = round(scores.gen_loss_corpus, 6)
                        if scores.gen_ppl_corpus == scores.gen_ppl_corpus:
                            off["gen_ppl"] = round(scores.gen_ppl_corpus, 4)
                        if uniq_mean == uniq_mean:
                            off["gen_uniq_mean"] = round(uniq_mean, 2)
                        off["gen_nonempty_frac"] = round(nonempty_frac, 4)
                        wrote_official = True
                    if "entropy" in due_names:
                        ment = _mean_finite(scores.entropy)
                        if ment == ment:
                            off["entropy"] = round(ment, 6)
                            wrote_official = True
                    if "dist1" in due_names:
                        d1 = _corpus_dist1(batch.texts, batch.seqlen)
                        if d1 == d1:
                            off["dist1"] = round(d1, 6)
                            wrote_official = True

                    write_sample_dir(
                        run_dir, step, batch, scores,
                        meta_extra={"eval_tick": tick, "due": sorted(due_names)},
                    )
                    summary_bits = []
                    if off.get("gen_ppl") != "":
                        summary_bits.append(f"gen_ppl {off['gen_ppl']}")
                    if off.get("entropy") != "":
                        summary_bits.append(f"entropy {off['entropy']}")
                    if off.get("dist1") != "":
                        summary_bits.append(f"dist1 {off['dist1']}")
                    if summary_bits:
                        msg = "eval/official: " + " ".join(summary_bits)
                        if pbar_parent is not None:
                            tqdm.write(f"{_TRAIN_LOG} {msg}")
                        else:
                            _train_log(msg)

                if wrote_official:
                    append_csv_row(eval_official_csv(run_dir), EVAL_OFFICIAL_FIELDS, off)

                save_eval_tick(run_dir, tick)

        if is_latent and cfg.vae_probe_samples > 0 and latent_probe_pool is not None:
            from models.tokens import get_token_layout
            from train.vae_probe import maybe_write_vae_probe

            pad_id = latent_pad_token_id
            if pad_id is None:
                pad_id = get_token_layout("gpt2").pad_token_id
            probe_meta: dict[str, Any] = {"eval_tick": tick}
            if curriculum_eval_ctx is not None:
                probe_meta["eval_split"] = getattr(
                    curriculum_eval_ctx, "eval_split", "",
                )
            if curriculum_sampler is not None:
                probe_meta["curriculum_stage"] = curriculum_sampler.current_stage.name
            maybe_write_vae_probe(
                run_dir,
                step,
                model,
                latent_probe_pool,
                pad_token_id=pad_id,
                device=device,
                amp_dtype=amp_dtype,
                probe_samples=cfg.vae_probe_samples,
                probe_seed=cfg.eval_sample_seed + step,
                meta_extra=probe_meta,
            )

    if is_distributed:
        dist.barrier()
