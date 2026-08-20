"""Training config loading for language-model pretraining."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeVar

import yaml

from config_util import load_mapping_config
from preprocess import get_preprocess
from train.generate_config import get_generate
from train.run_path import (
    CHECKPOINT_ROOT,
    build_train_fingerprint,
    config_hash_from_fingerprint,
    run_dir_for,
    run_relpath,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "train"
MODEL_DIR = CONFIG_DIR / "model"

TrainVariant = Literal["fast", "full"]
TrainDtype = Literal["bf16", "fp16", "fp32"]

TSub = TypeVar("TSub")

_MODEL_CONFIG_RE = re.compile(r"^([0-9]+m)-(fast|full)$")
_ARCH_SIZE_RE = re.compile(r"^[0-9]+m$")
_ARCH_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "models"
# 按整段 chunk 计 token 预算（非因果减 1）。
_SEQ_FULL_CHUNK_MODELS = frozenset(
    {
        "ar1_5",
        "ar2",
        "bd3lm",
        "bdelf",
        "elf",
        "late_ce",
        "lexce",
        "odar",
        "posbeta",
        "trace",
        "cola_vae",
        "cola",
        "denoiser_chart",
        "jac_ellipsoid",
        "residw",
        "loopsc",
    }
)
# DataLoader workers per rank; world_size comes from visible GPU count at launch.
DEFAULT_NUM_WORKERS = 8


@dataclass
class FL_OptimizerConfig:
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "dtype",
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "grad_clip",
        }
    )

    name: str = "prototype"
    dtype: TrainDtype = "bf16"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    muon_learning_rate: float = 0.003
    # Official Muon default; separate from AdamW weight_decay.
    muon_weight_decay: float = 0.01
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FL_ScheduleConfig:
    """Unified schedule: token budget drives optimizer steps; intervals are absolute.

    YAML ``{eval,save,snapshot,log_plot}_step`` are in optimizer-step units.
    ``compose_train_config`` multiplies them (and ``max_steps`` / warmup) by
    the derived accum (``global_batch_size / (batch_size * world_size)``) so
    ``train_loop`` can count every micro-batch.

    The only accepted knobs are ``target_tokens`` + ``warmup_ratio`` +
    ``{eval,save,snapshot,log_plot}_step``. Legacy ``max_steps`` / ``*_every`` /
    ``*_ratio`` fields are no longer supported.
    """

    _YAML_REQUIRED = frozenset(
        {
            "name",
            "variant",
            "target_tokens",
            "warmup_ratio",
            "min_lr_ratio",
            "eval_step",
            "save_step",
            "snapshot_step",
            "log_plot_step",
            "resume",
            "seed",
        }
    )

    name: str = "prototype"
    variant: TrainVariant = "fast"
    target_tokens: Optional[int] = None
    warmup_ratio: Optional[float] = None
    min_lr_ratio: float = 0.1
    log_plot_step: int = 100
    eval_step: int = 500
    save_step: int = 2000
    snapshot_step: int = 10_000
    resume: bool = True
    seed: int = 42
    use_muon: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FL_EvalConfig:
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "eval_sample_seed",
        }
    )
    # 不进指纹；YAML 省略视为 false（勿写入 default.yaml）
    _HASH_EXCLUDE = frozenset({"skip"})

    name: str = "prototype"
    # Online eval subsample; None / omitted runs the full eval split
    eval_sample_count: Optional[int] = None
    eval_sample_seed: int = 42
    # Generative PPL: train model samples → scored by HF causal LM
    gen_eval_model: str = "gpt2-large"
    gen_eval_model_dtype: TrainDtype = "bf16"
    gen_eval_model_device: str = "cuda"
    # Total sequences to generate+score per gen-PPL eval
    gen_eval_samples: int = 32
    skip: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FL_BatchConfig:
    """Per-run micro-batch. ``global_batch_size`` is required.

    Compose derives micro-batch accum as
    ``global_batch_size / (batch_size * world_size)``.
    """

    _YAML_REQUIRED = frozenset({"name", "batch_size", "global_batch_size"})

    name: str = "prototype"
    batch_size: int = 4
    global_batch_size: int = 4
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FL_TrainConfig:
    """Composed training config resolved from per-model recipe + global schedule/eval."""

    name: str
    model: str
    model_config: str
    variant: TrainVariant
    dataset: str
    preprocess: str
    generate: str
    checkpoint_root: str
    batch_size: int
    grad_accum_steps: int  # derived: global_batch_size / (batch_size * world_size)
    global_batch_size: int
    world_size: int
    dtype: TrainDtype
    max_steps: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    warmup_steps: int
    min_lr_ratio: float
    log_plot_step: int
    eval_step: int
    save_step: int
    snapshot_step: int
    num_workers: int
    resume: bool
    seed: int
    eval_sample_count: Optional[int]
    eval_sample_seed: int
    gen_eval_model: str
    gen_eval_model_dtype: TrainDtype
    gen_eval_model_device: str
    gen_eval_samples: int
    skip_eval: bool
    generate_sampling: Dict[str, Any]  # from config/generate/<model>/<name>.yaml
    use_muon: bool = True
    muon_learning_rate: float = 0.003
    muon_weight_decay: float = 0.01
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def seq_tokens(self) -> int:
        chunk = int(self.extra.get("chunk_length", 1024))
        if self.model in _SEQ_FULL_CHUNK_MODELS:
            return chunk
        return max(1, chunk - 1)

    @property
    def tokens_per_optimizer_step(self) -> int:
        return (
            self.batch_size
            * self.grad_accum_steps
            * self.world_size
            * self.seq_tokens
        )

    @property
    def tokens_per_micro_step(self) -> int:
        """全局数据 token / 微批（各 rank 合计）。"""
        return self.batch_size * self.world_size * self.seq_tokens

    def tokens_seen_after_step(self, step: int) -> int:
        """完成 0-based 微批 ``step`` 后累计吃掉的数据 token。"""
        return (step + 1) * self.tokens_per_micro_step

    @property
    def target_tokens(self) -> int | None:
        raw = self.extra.get("target_tokens")
        return int(raw) if raw is not None else None


def _parse_train_ref(model: str, config_name: str | None = None) -> tuple[str, str]:
    if config_name is None:
        if "/" not in model:
            raise ValueError(
                f"Invalid train ref {model!r}, expected model/name (e.g. ar/100m-fast)"
            )
        model, config_name = model.split("/", 1)

    known = list_train_models()
    if model not in known:
        raise ValueError(
            f"Unknown model {model!r}. Expected one of: {', '.join(known) or '<none>'}"
        )
    if not _MODEL_CONFIG_RE.fullmatch(config_name):
        raise ValueError(
            f"Invalid config name {config_name!r}, expected {{size}}m-{{fast,full}} "
            "(e.g. 100m-full, 300m-full)"
        )
    available = list_train_configs(model)
    if config_name not in available:
        raise ValueError(
            f"Unknown train config {config_name!r} for {model}. Available: "
            f"{', '.join(available) or '<none>'}"
        )
    return model, config_name


def _parse_model_config_variant(config_name: str) -> tuple[str, TrainVariant]:
    match = _MODEL_CONFIG_RE.fullmatch(config_name)
    if match is None:
        raise ValueError(f"Invalid config name {config_name!r}")
    return match.group(1), match.group(2)  # type: ignore[return-value]


_OVERRIDE_SECTIONS = frozenset(
    {"optimizer", "batch", "schedule", "eval", "generate", "model", "extra"}
)


def parse_train_overrides(items: list[str] | None) -> dict[str, dict[str, Any]]:
    """Parse CLI ``section.key=value`` overrides into nested dicts.

    Values are parsed with ``yaml.safe_load`` (so ``1e-3``, ``true``, ``null`` work).
    Allowed sections: optimizer, batch, schedule, eval, generate, model, extra.
    ``model.*`` 覆盖 ``config/models/<model>/<size>.yaml`` 键（进指纹）。
    ``extra.init_ckpt`` 为跨 run 初始化权重路径（进指纹；不恢复优化器）。
    ``eval.skip`` 关闭在线 held-out / gen-eval（省略为 false；不进指纹）。
    """
    if not items:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(
                f"Invalid --set {raw!r}; expected section.key=value "
                f"(e.g. optimizer.learning_rate=1e-3)"
            )
        key, value_raw = raw.split("=", 1)
        key = key.strip()
        if "." not in key:
            raise ValueError(
                f"Invalid --set key {key!r}; expected section.key "
                f"(sections: {', '.join(sorted(_OVERRIDE_SECTIONS))})"
            )
        section, field_name = key.split(".", 1)
        section = section.strip()
        field_name = field_name.strip()
        if section not in _OVERRIDE_SECTIONS:
            raise ValueError(
                f"Unknown --set section {section!r}; "
                f"expected one of: {', '.join(sorted(_OVERRIDE_SECTIONS))}"
            )
        if not field_name or "." in field_name:
            raise ValueError(
                f"Invalid --set field {key!r}; use a single section.key "
                f"(got nested path)"
            )
        try:
            value = yaml.safe_load(value_raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid --set value for {key!r}: {value_raw!r}") from exc
        # PyYAML keeps bare forms like ``1e-3`` as strings; coerce numerics.
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        out.setdefault(section, {})[field_name] = value
    return out


def _apply_mapping_overrides(
    target: dict[str, Any],
    overrides: dict[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    if not overrides:
        return target
    merged = dict(target)
    for key, value in overrides.items():
        merged[key] = value
    return merged


def model_train_config_path(model: str, variant: TrainVariant) -> Path:
    return MODEL_DIR / model / f"{variant}.yaml"


def _load_model_recipe(model: str, variant: TrainVariant) -> dict[str, Any]:
    if model == "prototype":
        raise ValueError("Prototype train config cannot be instantiated.")
    path = model_train_config_path(model, variant)
    if not path.is_file():
        available = ", ".join(list_train_models()) or "<none>"
        raise FileNotFoundError(
            f"Train recipe {path} does not exist. Available models: {available}"
        )
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    return raw


def _load_optimizer_from_recipe(
    recipe: dict[str, Any],
    *,
    path: Path,
    model_config: str,
    overrides: dict[str, Any] | None = None,
) -> FL_OptimizerConfig:
    section = recipe.get("optimizer")
    if not isinstance(section, dict):
        raise ValueError(f"{path}: missing mapping field 'optimizer'")
    raw = _apply_mapping_overrides(
        dict(section), overrides, label=f"{path}#optimizer",
    )
    raw.setdefault("name", model_config)
    return load_mapping_config(
        FL_OptimizerConfig,
        raw,
        required=FL_OptimizerConfig._YAML_REQUIRED,
        label=f"{path}#optimizer",
    )


def _load_batch_from_recipe(
    recipe: dict[str, Any],
    *,
    path: Path,
    config_name: str,
    overrides: dict[str, Any] | None = None,
) -> FL_BatchConfig:
    section = recipe.get("batch")
    if not isinstance(section, dict):
        raise ValueError(f"{path}: missing mapping field 'batch'")
    raw = _apply_mapping_overrides(
        dict(section), overrides, label=f"{path}#batch",
    )
    if "grad_accum_steps" in raw:
        raise ValueError(
            f"{path}#batch: grad_accum_steps is removed; set global_batch_size "
            f"(accum = global_batch_size / (batch_size * world_size))"
        )
    raw.setdefault("name", config_name)
    return load_mapping_config(
        FL_BatchConfig,
        raw,
        required=FL_BatchConfig._YAML_REQUIRED,
        label=f"{path}#batch",
    )


def _load_schedule(
    variant: TrainVariant,
    overrides: dict[str, Any] | None = None,
) -> FL_ScheduleConfig:
    path = CONFIG_DIR / "schedule" / f"{variant}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Schedule config {path} does not exist.")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    raw = _apply_mapping_overrides(raw, overrides, label=str(path))
    return load_mapping_config(
        FL_ScheduleConfig,
        raw,
        required=FL_ScheduleConfig._YAML_REQUIRED,
        label=str(path),
    )


def _load_eval(overrides: dict[str, Any] | None = None) -> FL_EvalConfig:
    path = CONFIG_DIR / "eval" / "default.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Eval config {path} does not exist.")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    raw = _apply_mapping_overrides(raw, overrides, label=str(path))
    for removed in ("use_fast_infer", "eval_gen_steps"):
        if removed in raw:
            raise ValueError(
                f"{path}: {removed} moved to config/generate/<model>/; "
                f"pass --generate <name> and/or --set generate.*"
            )
    return load_mapping_config(
        FL_EvalConfig, raw, required=FL_EvalConfig._YAML_REQUIRED, label=str(path),
    )


def _merge_extra(*parts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return merged


def _validate_dtype(dtype: str, *, path: str, label: str) -> None:
    if dtype not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"{path}: unsupported {label} {dtype!r}")


@dataclass(frozen=True)
class _ResolvedSchedule:
    max_steps: int
    warmup_steps: int
    log_plot_step: int
    eval_step: int
    save_step: int
    snapshot_step: int


def _resolve_schedule(
    schedule: FL_ScheduleConfig,
    *,
    run_name: str,
    tokens_per_step: int,
) -> _ResolvedSchedule:
    """Derive absolute steps from the token budget; intervals pass through."""
    if schedule.target_tokens is None or schedule.target_tokens < 1:
        raise ValueError(f"{run_name}: schedule.target_tokens must be set (>= 1)")
    if schedule.warmup_ratio is None:
        raise ValueError(f"{run_name}: schedule.warmup_ratio must be set")
    if tokens_per_step < 1:
        raise ValueError(f"{run_name}: tokens_per_optimizer_step must be >= 1")

    max_steps = max(1, math.ceil(schedule.target_tokens / tokens_per_step))
    warmup_steps = max(1, round(max_steps * schedule.warmup_ratio))

    for field_name in ("eval_step", "save_step", "snapshot_step", "log_plot_step"):
        if getattr(schedule, field_name) < 1:
            raise ValueError(f"{run_name}: schedule.{field_name} must be >= 1")

    return _ResolvedSchedule(
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        log_plot_step=schedule.log_plot_step,
        eval_step=schedule.eval_step,
        save_step=schedule.save_step,
        snapshot_step=schedule.snapshot_step,
    )


def _resolve_grad_accum(
    batch: FL_BatchConfig,
    *,
    world_size: int,
    run_name: str,
) -> tuple[int, int]:
    """Return ``(grad_accum_steps, global_batch_size)`` from ``global_batch_size``."""
    if batch.batch_size < 1 or world_size < 1:
        raise ValueError(f"{run_name}: batch_size/world_size must be >= 1")
    global_batch = int(batch.global_batch_size)
    if global_batch < 1:
        raise ValueError(
            f"{run_name}: global_batch_size must be >= 1, got {global_batch}"
        )
    denom = batch.batch_size * world_size
    if global_batch % denom != 0:
        raise ValueError(
            f"{run_name}: global_batch_size={global_batch} must be divisible "
            f"by batch_size*world_size={batch.batch_size}*{world_size}={denom}"
        )
    accum = global_batch // denom
    if accum < 1:
        raise ValueError(f"{run_name}: derived grad_accum_steps must be >= 1")
    return accum, global_batch


def compose_train_config(
    model: str,
    config_name: str | None = None,
    *,
    dataset: str,
    preprocess: str,
    generate: str,
    world_size: int | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> FL_TrainConfig:
    """Merge per-model recipe with global schedule/eval + generate.

    ``config_name`` must be ``{size}m-{fast,full}`` (e.g. ``100m-full``,
    ``300m-full``) and loads ``config/train/model/<model>/{fast|full}.yaml``
    plus architecture ``config/models/<model>/{size}.yaml``. Shared refs:
      - schedule ← ``schedule/<variant>.yaml``
      - eval ← ``eval/default.yaml``
      - generate ← ``config/generate/<model>/<generate>.yaml``

    ``overrides`` may contain ``optimizer`` / ``batch`` / ``schedule`` /
    ``eval`` / ``generate`` / ``model`` / ``extra`` dicts applied before dataclass
    validation（CLI ``--set section.key=value``；``model.*`` 覆盖架构 YAML；
    ``extra.init_ckpt`` 跨 run 加载权重）。

    ``dataset`` / ``preprocess`` / ``generate`` are supplied at launch
    (not from the train recipe yaml).
    ``world_size`` is the visible GPU count at launch (1/2/4/8); defaults to 1.
    """
    model, config_name = _parse_train_ref(model, config_name)
    model_config, variant = _parse_model_config_variant(config_name)
    ov = overrides or {}

    recipe_path = model_train_config_path(model, variant)
    recipe = _load_model_recipe(model, variant)
    optimizer = _load_optimizer_from_recipe(
        recipe,
        path=recipe_path,
        model_config=model_config,
        overrides=ov.get("optimizer"),
    )
    batch = _load_batch_from_recipe(
        recipe,
        path=recipe_path,
        config_name=config_name,
        overrides=ov.get("batch"),
    )
    schedule = _load_schedule(variant, overrides=ov.get("schedule"))
    eval_cfg = _load_eval(overrides=ov.get("eval"))
    # cola_vae 默认跳过在线 eval（可用 --set eval.skip=false 打开）；不进哈希
    if model == "cola_vae" and "skip" not in (ov.get("eval") or {}):
        eval_cfg.skip = True
    generate_cfg = get_generate(model, generate, overrides=ov.get("generate"))

    run_label = f"{model}-{config_name}"
    if schedule.use_muon:
        run_label = f"{run_label}-muon"
    chunk_length = get_preprocess(preprocess).chunk_length

    resolved_world_size = 1 if world_size is None else world_size
    accum, global_batch = _resolve_grad_accum(
        batch, world_size=resolved_world_size, run_name=run_label,
    )

    if schedule.variant != variant:
        raise ValueError(
            f"{run_label}: schedule.variant={schedule.variant!r} != {variant!r}"
        )

    _validate_dtype(optimizer.dtype, path=run_label, label="dtype")
    _validate_dtype(
        eval_cfg.gen_eval_model_dtype, path=run_label, label="gen_eval_model_dtype",
    )

    if eval_cfg.eval_sample_count is not None and eval_cfg.eval_sample_count < 1:
        raise ValueError(
            f"{run_label}: eval_sample_count must be >= 1 when set, "
            f"got {eval_cfg.eval_sample_count}"
        )
    if not isinstance(eval_cfg.skip, bool):
        raise ValueError(
            f"{run_label}: eval.skip must be a bool, got {eval_cfg.skip!r}"
        )
    if eval_cfg.gen_eval_samples < 1:
        raise ValueError(
            f"{run_label}: gen_eval_samples must be >= 1, "
            f"got {eval_cfg.gen_eval_samples}"
        )
    if eval_cfg.gen_eval_model_device not in ("cuda", "cpu"):
        raise ValueError(
            f"{run_label}: gen_eval_model_device must be 'cuda' or 'cpu', "
            f"got {eval_cfg.gen_eval_model_device!r}"
        )

    # 数据 token / 优化器步：global_batch × 每序列计入预算的 token 数。
    # denoise/decode 混合只影响 loss，不改变数据消耗与日程推导。
    tokens_per_step = (
        batch.batch_size
        * accum
        * resolved_world_size
        * (
            chunk_length
            if model in _SEQ_FULL_CHUNK_MODELS
            else max(1, chunk_length - 1)
        )
    )
    resolved = _resolve_schedule(
        schedule,
        run_name=run_label,
        tokens_per_step=tokens_per_step,
    )
    # ``_resolve_schedule`` 返回优化器步；``train_loop`` 按微批递增 step，此处换算。
    max_optimizer_steps = resolved.max_steps
    target_tokens = max_optimizer_steps * tokens_per_step

    log_plot_step = resolved.log_plot_step
    eval_step = resolved.eval_step
    save_step = resolved.save_step
    snapshot_step = resolved.snapshot_step

    max_steps = max_optimizer_steps * accum
    warmup_steps = resolved.warmup_steps * accum
    log_plot_step = max(1, log_plot_step * accum)
    eval_step = max(1, eval_step * accum)
    save_step = max(1, save_step * accum)
    snapshot_step = max(1, snapshot_step * accum)

    fingerprint = build_train_fingerprint(
        model=model,
        model_config=model_config,
        variant=variant,
        dataset=dataset,
        preprocess=preprocess,
        generate=generate,
        optimizer=optimizer,
        batch=batch,
        schedule=schedule,
        eval_cfg=eval_cfg,
        generate_cfg=generate_cfg,
        overrides=ov,
    )
    config_hash = config_hash_from_fingerprint(fingerprint)
    run_rel = run_relpath(variant=variant, model=model, config_hash=config_hash)

    extra = _merge_extra(
        optimizer.extra,
        schedule.extra,
        eval_cfg.extra,
        batch.extra,
        dict(ov.get("extra") or {}),
        {
            "chunk_length": chunk_length,
            "tokens_per_optimizer_step": tokens_per_step,
            "target_tokens": target_tokens,
            "max_optimizer_steps": max_optimizer_steps,
            "config_hash": config_hash,
            "run_relpath": run_rel,
            "config_refs": {
                "recipe": f"model/{model}/{variant}.yaml",
                "schedule": variant,
                "eval": "default",
                "generate": generate,
                "batch_profile": variant,
                "dataset": dataset,
                "preprocess": preprocess,
                "overrides": ov,
            },
            "use_muon": schedule.use_muon,
            "compile": bool(schedule.extra.get("compile", False)),
        },
    )

    return FL_TrainConfig(
        name=config_hash,
        model=model,
        model_config=model_config,
        variant=variant,
        dataset=dataset,
        preprocess=preprocess,
        generate=generate,
        checkpoint_root=CHECKPOINT_ROOT,
        batch_size=batch.batch_size,
        grad_accum_steps=accum,
        global_batch_size=global_batch,
        world_size=resolved_world_size,
        dtype=optimizer.dtype,
        max_steps=max_steps,
        learning_rate=optimizer.learning_rate,
        weight_decay=optimizer.weight_decay,
        beta1=optimizer.beta1,
        beta2=optimizer.beta2,
        grad_clip=optimizer.grad_clip,
        warmup_steps=warmup_steps,
        min_lr_ratio=schedule.min_lr_ratio,
        log_plot_step=log_plot_step,
        eval_step=eval_step,
        save_step=save_step,
        snapshot_step=snapshot_step,
        num_workers=DEFAULT_NUM_WORKERS,
        resume=schedule.resume,
        seed=schedule.seed,
        eval_sample_count=eval_cfg.eval_sample_count,
        eval_sample_seed=eval_cfg.eval_sample_seed,
        gen_eval_model=eval_cfg.gen_eval_model,
        gen_eval_model_dtype=eval_cfg.gen_eval_model_dtype,
        gen_eval_model_device=eval_cfg.gen_eval_model_device,
        gen_eval_samples=eval_cfg.gen_eval_samples,
        skip_eval=bool(eval_cfg.skip),
        generate_sampling=generate_cfg.to_sampling_cfg(),
        use_muon=schedule.use_muon,
        muon_learning_rate=optimizer.muon_learning_rate,
        muon_weight_decay=optimizer.muon_weight_decay,
        muon_momentum=optimizer.muon_momentum,
        muon_ns_steps=optimizer.muon_ns_steps,
        extra=extra,
    )


def list_train_models() -> List[str]:
    if not MODEL_DIR.is_dir():
        return []
    names: List[str] = []
    for path in sorted(MODEL_DIR.iterdir()):
        if not path.is_dir() or path.name == "prototype":
            continue
        if (path / "fast.yaml").is_file() or (path / "full.yaml").is_file():
            names.append(path.name)
    return names


def _arch_sizes(model: str) -> List[str]:
    """``config/models/<model>/{size}.yaml`` 中形如 ``100m`` / ``300m`` 的规格。"""
    model_dir = _ARCH_CONFIG_DIR / model
    if not model_dir.is_dir():
        return []
    return sorted(
        path.stem
        for path in model_dir.glob("*.yaml")
        if path.stem != "prototype" and _ARCH_SIZE_RE.fullmatch(path.stem)
    )


def list_train_configs(model: str | None = None) -> List[str]:
    """Return available ``{size}m-{fast,full}`` profiles for ``model`` (or all)."""
    models = [model] if model is not None else list_train_models()
    names: List[str] = []
    for m in models:
        if m == "prototype":
            continue
        model_dir = MODEL_DIR / m
        if not model_dir.is_dir():
            continue
        sizes = _arch_sizes(m)
        if not sizes:
            continue
        for size in sizes:
            for variant in ("fast", "full"):
                if not (model_dir / f"{variant}.yaml").is_file():
                    continue
                name = f"{size}-{variant}"
                if model is not None:
                    names.append(name)
                else:
                    names.append(f"{m}/{name}")
    return names


def get_train_config(
    model: str,
    config_name: str | None = None,
    *,
    dataset: str,
    preprocess: str,
    generate: str,
    world_size: int | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> FL_TrainConfig:
    return compose_train_config(
        model,
        config_name,
        dataset=dataset,
        preprocess=preprocess,
        generate=generate,
        world_size=world_size,
        overrides=overrides,
    )


def resolve_train_config_path(config_arg: str) -> Path:
    """Return the per-model train recipe yaml for the selected variant."""
    as_path = Path(config_arg)
    if as_path.suffix in (".yaml", ".yml") and as_path.is_file():
        return as_path
    model, config_name = _parse_train_ref(config_arg, None)
    _model_config, variant = _parse_model_config_variant(config_name)
    path = model_train_config_path(model, variant)
    if not path.is_file():
        available = ", ".join(list_train_models()) or "<none>"
        raise FileNotFoundError(
            f"Train recipe {path} does not exist. Available models: {available}"
        )
    return path
