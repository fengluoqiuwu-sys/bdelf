#!/usr/bin/env python3
"""LLM pretraining entry point for bdelf."""

from __future__ import annotations

import argparse
import atexit
import os
import sys
from pathlib import Path

# Do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here:
# on RTX 5080 + WSL2 + torch cu130 it segfaults at the first CUDA alloc.

import logging
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

import hf_config  # noqa: F401
from models import (
    build_model,
    kind_of,
    list_model_configs,
    list_models,
    resolve_model_config_path,
)
from dataset import list_datasets
from preprocess import get_preprocessed, list_preprocess
from train import (
    FL_TrainConfig,
    get_train_config,
    list_generate,
    list_train_configs,
    list_train_models,
    parse_train_overrides,
)
from train.batching import (
    TokenChunkDataset,
    build_eval_subset,
    collate_input_ids,
    shard_eval_dataset,
)
from train.dist import (
    _resolve_launch_world_size,
    setup_distributed,
)
from train.eval import load_gen_eval_baseline
from train.async_log import install as install_async_log
from train.async_log import shutdown as shutdown_async_log
from train.loop import set_seed, train_loop
from train.metrics import _train_log
from train.scratch import _cleanup_this_job_scratch

# TF32 + SDPA：全模型训练共用。Filters: residual Dynamo/HF noise.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo does not know how to trace the builtin.*posix\.(l?stat).*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.*",
)
logging.getLogger("torch._dynamo.backends.distributed").setLevel(logging.ERROR)


def _spawn_worker(
    local_rank: int,
    model_name: str,
    train_config: str,
    world_size: int,
    dataset: str,
    preprocess: str,
    generate: str,
    overrides: dict | None,
) -> None:
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    cfg = get_train_config(
        model_name,
        train_config,
        dataset=dataset,
        preprocess=preprocess,
        generate=generate,
        world_size=world_size,
        overrides=overrides,
    )
    size = train_config.rsplit("-", 1)[0]
    run_training(model_name, size, cfg)


def build_arg_parser() -> argparse.ArgumentParser:
    models = list_models() or ["<none>"]
    model_help = ", ".join(f"{m} ({kind_of(m)})" for m in models) if models != ["<none>"] else "<none>"
    datasets = list_datasets() or ["<none>"]
    preprocess_names = list_preprocess() or ["<none>"]
    parser = argparse.ArgumentParser(
        description="bdelf pretraining entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --model ar --config 100m-fast "
            "--dataset owt --preprocess default --generate eval\n"
            "  python train.py --model ar2 --config 100m-full "
            "--dataset owt --preprocess default --generate eval\n"
            "  python train.py --model ar1_5 --config 100m-full "
            "--dataset owt --preprocess default --generate eval\n"
            "  python train.py --model elf --config 100m-fast "
            "--dataset owt --preprocess elf --generate eval\n"
            "  python train.py --model elf --config 100m-full "
            "--dataset owt --preprocess elf --generate eval\n\n"
            f"Available models: {model_help}\n"
            f"Available datasets: {', '.join(datasets)}\n"
            f"Available preprocess configs: {', '.join(preprocess_names)}\n"
            "Train configs: {size}m-{fast,full} (from config/models/{kind}/<model>/)\n"
            "Generate configs: config/generate/{kind}/<model>/<name>.yaml\n"
            "Overrides: --set section.key=value "
            "(sections: optimizer, batch, schedule, eval, generate, model, extra)"
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model family name; options: {model_help}",
    )
    parser.add_argument(
        "--config",
        required=True,
        dest="train_config",
        metavar="CONFIG",
        help="Train config name, e.g. 100m-fast / 100m-full / 300m-full",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name (config/datasets/); options: {', '.join(datasets)}",
    )
    parser.add_argument(
        "--preprocess",
        required=True,
        help=f"Preprocess config name (config/preprocess/); options: {', '.join(preprocess_names)}",
    )
    parser.add_argument(
        "--generate",
        required=True,
        help=(
            "Generate config name under config/generate/<model>/ "
            "(train online gen-eval: eval; standalone generate.py: generate)"
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="SECTION.KEY=VALUE",
        help=(
            "Override a train hyperparameter after loading YAML "
            "(repeatable). Examples: optimizer.learning_rate=1e-3, "
            "batch.batch_size=16, schedule.target_tokens=1e9, "
            "eval.gen_eval_samples=64, generate.temperature=0.8, "
            "extra.init_ckpt=cache/checkpoints/full/elf/<hash>/checkpoint_latest.pt"
        ),
    )
    parser.add_argument(
        "--init-ckpt",
        default=None,
        metavar="PATH",
        help=(
            "Load model(+EMA) weights from another run and start a new hash "
            "(does not restore optimizer/step/RNG). Equivalent to "
            "--set extra.init_ckpt=PATH. Path relative to repo root."
        ),
    )
    parser.add_argument(
        "--gpus",
        default=None,
        metavar="IDS",
        help=(
            "Visible CUDA device indices (comma-separated physical IDs), "
            "e.g. 0,1 or 2,3. Sets CUDA_VISIBLE_DEVICES before launch. "
            "Required on common remotes where all GPUs are visible; "
            "optional elsewhere (Slurm usually already masks devices)."
        ),
    )
    return parser


def apply_cuda_visible_devices(gpus: str | None) -> list[int] | None:
    """解析 ``--gpus``，写入 ``CUDA_VISIBLE_DEVICES``；返回物理卡号列表。

    须在首次 ``torch.cuda.device_count()`` / 分配设备之前调用。
    """
    if gpus is None:
        return None
    raw = gpus.strip()
    if not raw:
        raise SystemExit("--gpus 不能为空（例: --gpus 0,1）")
    ids: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            raise SystemExit(f"--gpus 格式无效: {gpus!r}（例: 0,1）")
        try:
            idx = int(p)
        except ValueError as exc:
            raise SystemExit(f"--gpus 含非整数 {p!r}: {gpus!r}") from exc
        if idx < 0:
            raise SystemExit(f"--gpus 卡号须 >= 0，收到 {idx}")
        ids.append(idx)
    if len(ids) != len(set(ids)):
        raise SystemExit(f"--gpus 有重复卡号: {gpus!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)
    return ids


def _normalize_init_ckpt(spec: str) -> str:
    """把 ``--init-ckpt`` 收成相对仓库根的 posix 路径（进指纹）。

    相对路径 **不** ``resolve()`` 软链，避免 ``cache → /mnt/...`` 把绝对路径写进 hash。
    """
    repo = Path(__file__).resolve().parent
    raw = spec.strip().replace("\\", "/")
    if not raw:
        raise SystemExit("--init-ckpt 不能为空")
    p = Path(raw)
    if p.is_absolute():
        try:
            return p.resolve().relative_to(repo).as_posix()
        except ValueError:
            return str(p)
    cand = repo / raw
    if not cand.is_file():
        raise SystemExit(f"--init-ckpt 不存在: {cand}")
    return Path(raw).as_posix()


def validate_args(args: argparse.Namespace) -> tuple[str, str, FL_TrainConfig]:
    models = list_models()
    if args.model not in models:
        raise SystemExit(
            f"Unknown model {args.model!r}. Available: {', '.join(models) or '<none>'}\n"
            f"Model config: config/models/{{lm|latent}}/<model>/"
        )

    train_models = list_train_models()
    if args.model not in train_models:
        raise SystemExit(
            f"Model {args.model!r} has no train recipe. Available: {', '.join(train_models)}"
        )

    configs = list_train_configs(args.model)
    if args.train_config not in configs:
        raise SystemExit(
            f"Unknown train config {args.train_config!r}. {args.model} available: "
            f"{', '.join(configs)}\n"
            f"Naming format: {{size}}m-{{fast,full,curriculum}}"
        )

    if args.train_config.endswith("-curriculum"):
        size = args.train_config
    else:
        size = args.train_config.rsplit("-", 1)[0]
    try:
        resolve_model_config_path(args.model, size)
    except FileNotFoundError as exc:
        available = ", ".join(list_model_configs(args.model)) or "<none>"
        raise SystemExit(
            f"Model architecture config not found: "
            f"config/models/{kind_of(args.model)}/{args.model}/{size}.yaml\n"
            f"Available: {available}\n{exc}"
        ) from exc

    datasets = list_datasets()
    if args.dataset not in datasets:
        raise SystemExit(
            f"Unknown dataset {args.dataset!r}. Available: {', '.join(datasets) or '<none>'}\n"
            f"Config directory: config/datasets/"
        )

    preprocess_names = list_preprocess()
    if args.preprocess not in preprocess_names:
        raise SystemExit(
            f"Unknown preprocess config {args.preprocess!r}. Available: "
            f"{', '.join(preprocess_names) or '<none>'}\n"
            f"Config directory: config/preprocess/"
        )

    generate_names = list_generate(args.model)
    if args.generate not in generate_names:
        raise SystemExit(
            f"Unknown generate config {args.generate!r}. {args.model} available: "
            f"{', '.join(generate_names) or '<none>'}\n"
            f"Config directory: config/generate/{args.model}/"
        )

    try:
        overrides = parse_train_overrides(args.overrides)
    except ValueError as exc:
        raise SystemExit(f"Invalid --set override: {exc}") from exc

    if getattr(args, "init_ckpt", None):
        rel = _normalize_init_ckpt(args.init_ckpt)
        extra_ov = overrides.setdefault("extra", {})
        prev = extra_ov.get("init_ckpt")
        if prev not in (None, rel):
            raise SystemExit(
                f"--init-ckpt={rel!r} 与 --set extra.init_ckpt={prev!r} 冲突"
            )
        extra_ov["init_ckpt"] = rel

    try:
        launch_world_size = _resolve_launch_world_size()
        cfg = get_train_config(
            args.model,
            args.train_config,
            dataset=args.dataset,
            preprocess=args.preprocess,
            generate=args.generate,
            world_size=launch_world_size,
            overrides=overrides,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Failed to load train config: {exc}") from exc

    return args.model, size, cfg


def run_training(model_name: str, model_size: str, cfg: FL_TrainConfig) -> None:
    rank, world_size, device, is_distributed = setup_distributed(cfg)
    if rank == 0:
        install_async_log()
    set_seed(cfg.seed, rank)
    try:
        _run_training_body(
            model_name, model_size, cfg,
            rank=rank, world_size=world_size, device=device,
            is_distributed=is_distributed,
        )
    finally:
        if rank == 0:
            shutdown_async_log()
        if is_distributed:
            dist.destroy_process_group()


def _run_training_body(
    model_name: str,
    model_size: str,
    cfg: FL_TrainConfig,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    is_distributed: bool,
) -> None:

    if rank == 0:
        _train_log(f"Model: {model_name}/{model_size}")
        _train_log(
            f"Train run: {cfg.extra.get('run_relpath', cfg.name)} "
            f"(hash={cfg.name}, variant={cfg.variant})"
        )
        overrides = (cfg.extra.get("config_refs") or {}).get("overrides") or {}
        if overrides:
            flat = []
            for section, mapping in sorted(overrides.items()):
                for key, value in sorted(mapping.items()):
                    flat.append(f"{section}.{key}={value!r}")
            _train_log(f"Overrides: {', '.join(flat)}")
        if cfg.use_muon:
            _train_log(
                f"Optimizer: Muon+AdamW hybrid "
                f"(muon_lr={cfg.muon_learning_rate}, adam_lr={cfg.learning_rate})"
            )
        _train_log(
            f"Data: dataset={cfg.dataset}, preprocess={cfg.preprocess}, "
            f"generate={cfg.generate}"
        )
        _train_log(f"Device: {device}, world_size={world_size}")
        if cfg.target_tokens is not None:
            opt_steps = int(cfg.extra.get("max_optimizer_steps", 0)) or (
                cfg.max_steps // max(1, cfg.grad_accum_steps)
            )
            _train_log(
                f"Token budget: {cfg.target_tokens:,} data tokens "
                f"({cfg.tokens_per_optimizer_step:,}/opt-step) → "
                f"{opt_steps:,} optimizer steps "
                f"({cfg.max_steps:,} data micro-steps, accum={cfg.grad_accum_steps})",
            )

    curriculum_sampler = None
    curriculum_eval_ctx = None
    train_ds: TokenChunkDataset | None = None
    eval_ds_full = None
    latent_probe_pool = None
    latent_pad_token_id: int | None = None

    if cfg.preprocess == "latent-curriculum":
        from train.latent_curriculum import (
            LatentCurriculumSampler,
            load_curriculum_spec,
            resolve_curriculum_spec_name,
        )
        from train.eval_split import resolve_eval_split
        from train.latent_eval import LatentCurriculumEvalContext
        from models.tokens import get_token_layout

        if is_distributed and rank != 0:
            dist.barrier()
        cur_spec = load_curriculum_spec(resolve_curriculum_spec_name(cfg.preprocess))
        layout = get_token_layout("gpt2")
        latent_pad_token_id = layout.pad_token_id
        curriculum_sampler = LatentCurriculumSampler.build(
            cur_spec,
            dataset=cfg.dataset,
            pad_token_id=layout.pad_token_id,
            seed=cfg.seed,
            world_size=world_size,
            batch_size=cfg.batch_size,
            stage_batch_sizes={
                int(k): int(v)
                for k, v in (cfg.extra.get("stage_batch_size") or {}).items()
            } or None,
        )
        seg512 = get_preprocessed(cur_spec.seg512_preprocess, cfg.dataset)
        get_preprocessed(cur_spec.bucket_preprocess, cfg.dataset)
        eval_split = resolve_eval_split(seg512, cfg.eval_split)
        if is_distributed and rank == 0:
            dist.barrier()
        curriculum_eval_ctx = LatentCurriculumEvalContext.build(
            cur_spec,
            dataset=cfg.dataset,
            pad_token_id=layout.pad_token_id,
            batch_size=cfg.batch_size,
            eval_sample_count=cfg.eval_sample_count,
            eval_sample_seed=cfg.eval_sample_seed,
            eval_split=eval_split,
            rank=rank,
            world_size=world_size,
        )
        latent_probe_pool = curriculum_eval_ctx.seg512_probe_pool
        eval_ds_full = TokenChunkDataset(seg512.load_split(eval_split))
    else:
        try:
            # On a cache miss only rank 0 downloads/builds; the other ranks wait
            # and then attach to the finished cache. Concurrent builds would write
            # the same shard/manifest files and corrupt the cache.
            if is_distributed and rank != 0:
                dist.barrier()
            preprocessed = get_preprocessed(cfg.preprocess, cfg.dataset)
            if is_distributed and rank == 0:
                dist.barrier()
        except FileNotFoundError as exc:
            msg = (
                f"Preprocessed data unavailable: {exc}\n"
                f"Check that dataset={cfg.dataset}, preprocess={cfg.preprocess} "
                f"are configured correctly (first run will download the dataset "
                f"and build the preprocess cache automatically)"
            )
            if rank == 0:
                _train_log(msg, file=sys.stderr)
            raise SystemExit(msg) from exc

        from train.eval_split import require_train_and_holdout, resolve_eval_split

        splits = preprocessed.get_splits()
        require_train_and_holdout(splits)
        eval_split = resolve_eval_split(preprocessed, cfg.eval_split)

        train_ds = TokenChunkDataset(preprocessed.load_split("train"))
        eval_ds_full = TokenChunkDataset(preprocessed.load_split(eval_split))

    eval_loader: DataLoader | None = None
    eval_run_size = 0
    gpt2_model: nn.Module | None = None
    model_kind = kind_of(model_name)
    eval_ds, eval_run_size = build_eval_subset(
        eval_ds_full,
        cfg.eval_sample_count,
        cfg.eval_sample_seed,
    )
    if model_kind == "latent" and latent_probe_pool is None:
        from models.tokens import get_token_layout

        latent_pad_token_id = get_token_layout("gpt2").pad_token_id
        latent_probe_pool = eval_ds
    if len(eval_ds) == 0:
        if rank == 0:
            _train_log("WARNING: eval dataset is empty; eval will be skipped")
    else:
        eval_ds_local = shard_eval_dataset(
            eval_ds, rank=rank, world_size=world_size,
        )
        eval_loader = DataLoader(
            eval_ds_local,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_input_ids,
        )
    if model_kind == "lm":
        if is_distributed:
            if rank == 0:
                gpt2_model = load_gen_eval_baseline(cfg)
                dist.barrier()
            else:
                dist.barrier()
                gpt2_model = load_gen_eval_baseline(cfg)
        else:
            gpt2_model = load_gen_eval_baseline(cfg)
        if rank == 0:
            _train_log(
                f"Loaded gen-eval baseline {cfg.gen_eval_model} "
                f"on {cfg.gen_eval_model_device} (all {world_size} ranks)",
            )

    model_cfg_path = resolve_model_config_path(model_name, model_size)
    import yaml

    with open(model_cfg_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}
    model_overrides = (
        (cfg.extra.get("config_refs") or {}).get("overrides") or {}
    ).get("model")
    if model_overrides:
        model_cfg = {**model_cfg, **dict(model_overrides)}

    if model_name == "cola":
        # Help Stage-2 auto-resolve matching cola_vae-{size}-{variant}* checkpoints.
        model_cfg = dict(model_cfg)
        model_cfg["train_variant"] = cfg.variant
        # Resume packs VAE+DiT; skip Stage-1 disk resolve so missing VAE dirs
        # do not block loading an existing cola run checkpoint.
        from train.run_path import checkpoint_run_dir_from_cfg

        latest_cola = checkpoint_run_dir_from_cfg(cfg) / "checkpoint_latest.pt"
        if cfg.resume and latest_cola.is_file():
            model_cfg["load_vae_weights"] = False

    model = build_model(model_name, model_cfg).to(device)
    model_meta = {
        "name": model_name,
        "config_file": str(model_cfg_path),
        "config": model_cfg,
    }

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _train_log(f"Train model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
        if curriculum_sampler is not None:
            st = curriculum_sampler.curriculum_state()
            _train_log(
                f"curriculum: {st['stage']} peak_L={st['graph_l']} "
                f"target={st['target_effective_tokens']:,} effective tokens"
            )
        elif train_ds is not None:
            _train_log(
                f"train split: {len(train_ds):,} samples"
                + (
                    f", eval split: {len(eval_ds_full):,} samples"
                    if eval_ds_full is not None
                    else ""
                ),
            )
        elif eval_ds_full is not None:
            _train_log(f"eval split: {len(eval_ds_full):,} samples")
        if eval_ds_full is not None and eval_run_size < len(eval_ds_full):
            _train_log(
                f"eval subsample: {eval_run_size:,} / {len(eval_ds_full):,} "
                f"(seed={cfg.eval_sample_seed})",
            )
        if world_size > 1 and eval_loader is not None:
            _train_log(
                f"eval sharded across {world_size} ranks "
                f"(~{eval_run_size // world_size} samples/rank)",
            )
        if model_kind == "lm" and eval_loader is not None:
            _train_log(
                f"gen. ppl: {cfg.gen_eval_samples} samples / eval via "
                f"{cfg.gen_eval_model} "
                f"({cfg.gen_eval_model_dtype} on {cfg.gen_eval_model_device}"
                + (f", sharded×{world_size}" if world_size > 1 else "")
                + ")",
            )
        if model_kind == "latent" and cfg.vae_probe_samples > 0:
            _train_log(
                f"vae probe: {cfg.vae_probe_samples} samples / eval "
                f"(seed={cfg.eval_sample_seed}+step)",
            )

    train_loop(
        model,
        cfg,
        model_meta,
        train_ds,
        eval_loader,
        gpt2_model,
        rank=rank,
        world_size=world_size,
        device=device,
        is_distributed=is_distributed,
        curriculum_sampler=curriculum_sampler,
        curriculum_eval_ctx=curriculum_eval_ctx,
        latent_probe_pool=latent_probe_pool,
        latent_pad_token_id=latent_pad_token_id,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    # launch 包装器会先写入 runner pid；此处仅在未设置时用本进程 pid。
    os.environ.setdefault("BDELF_JOB_ID", str(os.getpid()))
    atexit.register(_cleanup_this_job_scratch)
    # 须在 validate_args → device_count 之前屏蔽可见卡（common 机看得到全部 GPU）
    gpu_ids = apply_cuda_visible_devices(args.gpus)
    if gpu_ids is not None:
        _train_log(
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
            f"(physical gpu_ids={gpu_ids})"
        )
    model_name, model_size, cfg = validate_args(args)

    if cfg.world_size > 1 and "RANK" not in os.environ:
        if not torch.cuda.is_available():
            raise SystemExit("Multi-GPU training requires CUDA.")
        n_gpu = torch.cuda.device_count()
        if n_gpu < cfg.world_size:
            raise SystemExit(
                f"Configured world_size={cfg.world_size}, "
                f"but this machine has only {n_gpu} GPU(s)."
            )
        import torch.multiprocessing as mp

        # Build the dataset/preprocess cache once in the parent so workers hit
        # a warm cache; a cold build inside a worker could exceed the NCCL
        # barrier timeout that the other ranks wait on.
        try:
            if cfg.preprocess == "latent-curriculum":
                from train.latent_curriculum import (
                    load_curriculum_spec,
                    resolve_curriculum_spec_name,
                )

                spec = load_curriculum_spec(
                    resolve_curriculum_spec_name(cfg.preprocess),
                )
                get_preprocessed(spec.seg512_preprocess, cfg.dataset)
                get_preprocessed(spec.bucket_preprocess, cfg.dataset)
            else:
                get_preprocessed(cfg.preprocess, cfg.dataset)
        except FileNotFoundError as exc:
            raise SystemExit(f"Preprocessed data unavailable: {exc}") from exc

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        # 子进程继承 BDELF_JOB_ID，stage/compile 共用同一 job 目录（勿用各 rank pid）
        _train_log(
            f"Auto-spawning {cfg.world_size} processes (spawn), "
            f"MASTER_PORT={os.environ.get('MASTER_PORT', '29500')}",
        )
        try:
            mp.spawn(
                _spawn_worker,
                args=(
                    model_name,
                    args.train_config,
                    cfg.world_size,
                    args.dataset,
                    args.preprocess,
                    args.generate,
                    (cfg.extra.get("config_refs") or {}).get("overrides") or None,
                ),
                nprocs=cfg.world_size,
                join=True,
            )
        finally:
            _cleanup_this_job_scratch()
        return

    run_training(model_name, model_size, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _train_log("Interrupt received; exiting.")
    except Exception as exc:
        _train_log(f"Error: {exc}", file=sys.stderr)
        raise
