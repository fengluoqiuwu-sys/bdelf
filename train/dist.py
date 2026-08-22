"""分布式进程组初始化与 world_size 解析。"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from train.train import FL_TrainConfig
from train.scratch import _isolate_compile_cache

def setup_distributed(cfg: FL_TrainConfig) -> tuple[int, int, torch.device, bool]:
    if cfg.world_size <= 1:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device found. Single-GPU training requires a GPU.")
        _isolate_compile_cache(0)
        return 0, 1, torch.device("cuda"), False

    if "RANK" not in os.environ:
        raise RuntimeError("Distributed worker missing RANK environment variable")

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    torch.cuda.set_device(local_rank)
    # Isolate compile caches before any torch.compile happens in train_loop.
    _isolate_compile_cache(local_rank)

    if not dist.is_initialized():
        # Rank0 ELF gen-eval (SDE/32 + GPT-2 score) can exceed the default
        # 10min NCCL watchdog while peers wait on a 1-element all_reduce in
        # _sync_after_rank0_work; use a longer PG timeout to avoid false hangs.
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(minutes=60),
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != cfg.world_size:
        raise RuntimeError(
            f"Configured world_size={cfg.world_size}, but {world_size} processes were launched."
        )

    return rank, world_size, torch.device(f"cuda:{local_rank}"), True


ALLOWED_WORLD_SIZES = frozenset({1, 2, 4, 8})


def _resolve_launch_world_size() -> int:
    """Auto-detect visible GPU count; must be in {1, 2, 4, 8} (CPU-only → 1)."""
    if not torch.cuda.is_available():
        return 1
    n = torch.cuda.device_count()
    if n not in ALLOWED_WORLD_SIZES:
        raise SystemExit(
            f"训练需要 1/2/4/8 张可见 GPU，当前 device_count={n}"
        )
    return n
