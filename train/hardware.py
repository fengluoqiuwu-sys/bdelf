"""训练硬件锁定：首次写入 run 目录，续跑必须一致（不进 config-hash）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

HARDWARE_FILENAME = "hardware.json"


def _round_memory_gb(value: Any) -> int:
    """单卡显存 GiB：四舍五入取整即可，不要求与标称字节数精确一致。"""
    return int(round(float(value)))


@dataclass(frozen=True)
class TrainHardware:
    """本 run 绑定的 GPU 规格。"""

    gpu_name: str
    gpu_count: int
    memory_gb_per_gpu: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainHardware":
        return cls(
            gpu_name=str(raw["gpu_name"]),
            gpu_count=int(raw["gpu_count"]),
            memory_gb_per_gpu=_round_memory_gb(raw["memory_gb_per_gpu"]),
        )


class HardwareMismatchError(ValueError):
    """续训 / 重拉起时 GPU 规格与首次记录不一致。"""


def detect_train_hardware() -> TrainHardware:
    """探测当前进程可见 GPU：型号、数量、单卡显存 GB。"""
    if not torch.cuda.is_available():
        raise HardwareMismatchError(
            "训练需要 CUDA GPU，但当前 torch.cuda.is_available()=False"
        )
    n = torch.cuda.device_count()
    if n < 1:
        raise HardwareMismatchError("训练需要至少 1 张可见 GPU")

    names: list[str] = []
    mems: list[int] = []
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        names.append(str(props.name))
        # 四舍五入到 GiB（可见显存常略小于标称，避免 16G 卡变成 15）
        mems.append(_round_memory_gb(props.total_memory / (1024**3)))

    if len(set(names)) != 1 or len(set(mems)) != 1:
        detail = ", ".join(f"{n}@{m}GiB" for n, m in zip(names, mems))
        raise HardwareMismatchError(
            f"可见 GPU 型号/显存不一致，拒绝训练: {detail}"
        )
    return TrainHardware(
        gpu_name=names[0],
        gpu_count=n,
        memory_gb_per_gpu=mems[0],
    )


def hardware_path(run_dir: Path) -> Path:
    return Path(run_dir) / HARDWARE_FILENAME


def load_hardware(run_dir: Path) -> TrainHardware | None:
    path = hardware_path(run_dir)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise HardwareMismatchError(f"{path}: 内容必须是 JSON object")
    return TrainHardware.from_dict(raw)


def save_hardware(run_dir: Path, hw: TrainHardware) -> Path:
    path = hardware_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(hw.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)
    return path


def ensure_hardware_lock(run_dir: Path, current: TrainHardware | None = None) -> TrainHardware:
    """首次写入 ``hardware.json``；若已存在则必须与当前探测一致。

    不参与 config-hash；仅绑定同一 hash 目录的物理机规格。
    """
    current = current or detect_train_hardware()
    saved = load_hardware(run_dir)
    if saved is None:
        save_hardware(run_dir, current)
        return current
    # 显存只比四舍五入后的 GiB 整数，不要求底层字节精确相同。
    if (
        saved.gpu_name != current.gpu_name
        or saved.gpu_count != current.gpu_count
        or _round_memory_gb(saved.memory_gb_per_gpu)
        != _round_memory_gb(current.memory_gb_per_gpu)
    ):
        raise HardwareMismatchError(
            f"GPU 规格与首次训练记录不一致，拒绝续跑/重拉起。\n"
            f"  记录: name={saved.gpu_name!r}, count={saved.gpu_count}, "
            f"memory_gb={saved.memory_gb_per_gpu}\n"
            f"  当前: name={current.gpu_name!r}, count={current.gpu_count}, "
            f"memory_gb={current.memory_gb_per_gpu}\n"
            f"  文件: {hardware_path(run_dir)}"
        )
    return saved
