"""Per-model generate / sampling configs under ``config/generate/<model>/``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from config_util import load_mapping_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "generate"

# 训练在线 gen-eval / 独立 generate.py 常用名
GENERATE_PROFILE = "generate"
EVAL_PROFILE = "eval"


@dataclass
class FL_GenerateConfig:
    """模型生成配置。YAML 中除 ``name`` 外的键均为采样参数，进入 ``extra``。"""

    _YAML_REQUIRED = frozenset({"name"})

    name: str = "prototype"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_sampling_cfg(self) -> Dict[str, Any]:
        """返回传给 ``model.generate(..., sampling_cfg=...)`` 的字典。

        ACE 关闭时不写入 ``ace`` / ``ace_direction``，使缺省与旧 YAML 指纹一致。
        """
        out = {
            k: v
            for k, v in self.extra.items()
            if not str(k).startswith("_")
        }
        ace = out.get("ace", False)
        # 与 models.elf.ace.ace_is_enabled 对齐：未写 / false / 0 → 关
        ace_off = (
            ace is False
            or ace is None
            or (isinstance(ace, (int, float)) and float(ace) == 0.0)
            or (isinstance(ace, str) and not str(ace).strip())
        )
        if ace_off:
            out.pop("ace", None)
            out.pop("ace_direction", None)
        elif out.get("ace_direction") in (None, False, ""):
            # 自动缓存方向时不把 null 写入采样字典 / 指纹
            out.pop("ace_direction", None)

        # 采样 DMA 默认开：缺省/true 不写入，保持与旧 YAML 指纹一致
        if "dma" in out:
            dma = out.get("dma", True)
            dma_off = (
                dma is False
                or (isinstance(dma, (int, float)) and float(dma) == 0.0)
                or (
                    isinstance(dma, str)
                    and str(dma).strip().lower() in ("false", "0", "off", "no")
                )
            )
            if dma_off:
                out["dma"] = False
            else:
                out.pop("dma", None)

        # dma_ace_order 默认 after（先 ACE 再 DMA）；缺省/after 不写入
        if "dma_ace_order" in out:
            order = str(out.get("dma_ace_order") or "after").lower().strip()
            if order in ("", "after"):
                out.pop("dma_ace_order", None)
            else:
                out["dma_ace_order"] = order
        return out


def generate_config_path(model: str, name: str) -> Path:
    return CONFIG_DIR / model / f"{name}.yaml"


def list_generate(model: str | None = None) -> List[str]:
    """列出可用生成配置名（``model`` 给定时只列该模型；否则 ``model/name``）。"""
    if model is not None:
        model_dir = CONFIG_DIR / model
        if not model_dir.is_dir():
            return []
        return sorted(
            path.stem
            for path in model_dir.glob("*.yaml")
            if path.stem != "prototype"
        )

    names: List[str] = []
    if not CONFIG_DIR.is_dir():
        return names
    for model_dir in sorted(CONFIG_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.glob("*.yaml")):
            if path.stem == "prototype":
                continue
            names.append(f"{model_dir.name}/{path.stem}")
    return names


def get_generate(
    model: str,
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> FL_GenerateConfig:
    """加载 ``config/generate/<model>/<name>.yaml``。"""
    if name == "prototype":
        raise ValueError("Prototype generate config cannot be instantiated.")
    path = generate_config_path(model, name)
    if not path.is_file():
        available = ", ".join(list_generate(model)) or "<none>"
        raise FileNotFoundError(
            f"Generate config {path} does not exist. "
            f"Available for {model}: {available}"
        )
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    if overrides:
        raw = dict(raw)
        raw.update(overrides)
    return load_mapping_config(
        FL_GenerateConfig,
        raw,
        required=FL_GenerateConfig._YAML_REQUIRED,
        label=str(path),
    )
