"""编译缓存隔离与 job scratch 清理（Inductor / Dynamo 辅助）。"""

from __future__ import annotations

import atexit
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn

def _preload_frozen_encoders(model: nn.Module) -> None:
    """Load frozen HF encoders in eager mode before ``torch.compile``.

    ELF keeps T5 in a non-registered holder and loads it on first encode; if
    that happens under Dynamo, HF ``from_pretrained`` / peft / posix.stat paths
    get traced and spam warnings (and may re-load weights mid-compile).
    """
    backbone = getattr(model, "backbone", None)
    ensure = getattr(backbone, "_ensure_encoder", None)
    if callable(ensure):
        ensure()


def _patch_inductor_bool_eq() -> None:
    """Work around a torch.compile/Inductor bug on boolean value ranges.

    ``SymPyValueRangeAnalysis.eq`` runs ``a.lower > b.upper`` to test for
    disjoint ranges, but when the operands are boolean (e.g. the ``mode_*``
    embeddings and timestep embedding feed a bool-typed indexing expr), sympy
    forbids ordered comparison on Booleans and raises "A Boolean argument can
    only be used in Eq and Ne". Sibling ops (lt/gt/mul/...) already special-case
    ``is_bool``; only ``eq`` was missed (fixed upstream, not in this torch).
    Patch ``eq`` to mirror that handling; ``ne`` delegates to ``eq`` and is
    fixed for free. See https://github.com/pytorch/pytorch/issues/188231.
    """
    try:
        import sympy
        from torch.utils._sympy.value_ranges import (
            SymPyValueRangeAnalysis,
            ValueRanges,
        )
    except Exception:
        return

    @staticmethod
    def _eq(a, b):  # type: ignore[no-untyped-def]
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton() and a.lower == b.lower:
            return ValueRanges.wrap(sympy.true)
        if a.is_bool or b.is_bool:
            # Booleans are unorderable; two unequal singletons are disjoint,
            # otherwise the result is an unknown bool.
            if a.is_singleton() and b.is_singleton():
                return ValueRanges.wrap(sympy.false)
            return ValueRanges(sympy.false, sympy.true)
        if a.lower > b.upper or b.lower > a.upper:  # ranges disjoint
            return ValueRanges.wrap(sympy.false)
        return ValueRanges(sympy.false, sympy.true)

    SymPyValueRangeAnalysis.eq = _eq


_patch_inductor_bool_eq()


def _is_safe_job_id(val: str) -> bool:
    if not val or len(val) > 64:
        return False
    return all(c.isalnum() or c in "._-" for c in val)


def _scratch_job_id() -> str:
    """多进程共用的 scratch 目录后缀。

    Slurm 用 ``SLURM_JOB_ID``；common / ``mp.spawn`` 各 rank 的 ``getpid()``
    不同——若用 pid，非 rank0 会等错路径并超时。优先显式 ``BDELF_JOB_ID``
    （launch 包装器或父进程 spawn 前写入），否则退回 ``getppid()``。
    """
    for key in ("SLURM_JOB_ID", "BDELF_JOB_ID"):
        val = os.environ.get(key)
        if val and _is_safe_job_id(val):
            return str(val)
    return str(os.getppid())


def _scratch_root() -> Path | None:
    """节点本地可写 scratch：``SLURM_TMPDIR`` → ``TMPDIR`` → ``/tmp``。"""
    for candidate in (
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("TMPDIR"),
        "/tmp",
    ):
        if not candidate:
            continue
        try:
            path = Path(candidate)
            if path.is_dir() and os.access(path, os.W_OK):
                return path
        except OSError:
            continue
    return None


def _scratch_job_dirs() -> list[Path]:
    """本 job 的 resume / compile 根目录（含 /tmp 旧路径，便于清残留）。"""
    job = _scratch_job_id()
    roots: list[Path] = []
    scratch = _scratch_root()
    if scratch is not None:
        roots.append(scratch)
    tmp = Path("/tmp")
    try:
        extra_tmp = scratch is None or scratch.resolve() != tmp.resolve()
    except OSError:
        extra_tmp = True
    if extra_tmp:
        roots.append(tmp)
    dirs: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for name in (
            f"bdelf-resume-{job}",
            f"bdelf-compile-{job}",
            f"bdelf-compile-pid{os.getpid()}",
        ):
            path = root / name
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            dirs.append(path)
    return dirs


def _rmtree_job_scratch(path: Path) -> None:
    name = path.name
    if not (name.startswith("bdelf-resume-") or name.startswith("bdelf-compile-")):
        return
    shutil.rmtree(path, ignore_errors=True)


def _cleanup_this_job_scratch() -> None:
    """只删本 job id / 本 pid 的 scratch，禁止通配 ``bdelf-*``。"""
    for path in _scratch_job_dirs():
        _rmtree_job_scratch(path)


def _cleanup_compile_rank(local_rank: int, compile_root: str) -> None:
    """子进程只清本 rank 子目录，避免先退出的 rank 删掉仍在跑的缓存。"""
    for sub in ("inductor", "triton"):
        shutil.rmtree(
            os.path.join(compile_root, sub, f"rank{local_rank}"),
            ignore_errors=True,
        )


def _local_compile_root() -> str | None:
    """选节点本地目录给 Triton/Inductor 缓存。

    Triton / Inductor 依赖 temp-file + rename；BeeGFS/NFS 上并发编译会丢
    ``*.cubin``。优先 Slurm 节点盘，再 TMPDIR，最后 /tmp。
    一次作业一个根（job id），rank 子目录隔离。
    """
    scratch = _scratch_root()
    if scratch is None:
        return None
    root = scratch / f"bdelf-compile-{_scratch_job_id()}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        return str(root)
    except OSError:
        return None


def _isolate_compile_cache(local_rank: int) -> None:
    """每个 local rank 使用节点本地盘上独立的 Triton/Inductor 缓存。

    始终优先节点本地 scratch，而不是 Slurm 设在共享盘（BeeGFS）上的路径。
    """
    local_root = _local_compile_root()
    for var, subdir in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        if local_root is not None:
            base = os.path.join(local_root, subdir)
        else:
            base = os.environ.get(var)
            if not base:
                continue
        per_rank = os.path.join(base, f"rank{local_rank}")
        os.makedirs(per_rank, exist_ok=True)
        os.environ[var] = per_rank
    if local_root is not None and "RANK" in os.environ:
        atexit.register(_cleanup_compile_rank, local_rank, local_root)
    else:
        atexit.register(_cleanup_this_job_scratch)
