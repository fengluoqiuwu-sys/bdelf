"""Training config loading for language-model pretraining."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

from config_util import load_yaml_config
from preprocess import get_preprocess

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "train"
BATCH_DIR = CONFIG_DIR / "batch"
CHECKPOINT_ROOT = "cache/checkpoints"

TrainVariant = Literal["fast", "full"]
TrainDtype = Literal["bf16", "fp16", "fp32"]

TSub = TypeVar("TSub")

_TRAIN_MODELS = ("ar", "bd3lm", "bdelf")
_MODEL_CONFIG_RE = re.compile(r"^(100m|300m|900m)-(fast|full)$")
_HARDWARE_BY_VARIANT = {"fast": "fast-16gb", "full": "full-8x4090"}


@dataclass
class FL_HardwareConfig:
    _YAML_REQUIRED = frozenset(
        {"name", "world_size", "num_workers", "gpu_memory_gb", "memory_headroom_gb"}
    )

    name: str = "prototype"
    world_size: int = 1
    num_workers: int = 2
    gpu_memory_gb: float = 16.0
    memory_headroom_gb: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


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
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FL_ScheduleConfig:
    _YAML_REQUIRED = frozenset(
        {
            "name",
            "variant",
            "max_steps",
            "warmup_steps",
            "min_lr_ratio",
            "log_plot_every",
            "eval_every",
            "save_every",
            "snapshot_every",
            "resume",
            "seed",
        }
    )

    name: str = "prototype"
    variant: TrainVariant = "fast"
    max_steps: int = 5000
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    log_plot_every: int = 100
    eval_every: int = 500
    save_every: int = 2000
    snapshot_every: int = 10_000
    resume: bool = True
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FL_EvalConfig:
    _YAML_REQUIRED = frozenset(
        {"name", "eval_model", "eval_model_dtype", "eval_model_device"}
    )

    name: str = "prototype"
    eval_model: str = "gpt2-large"
    eval_model_dtype: TrainDtype = "bf16"
    eval_model_device: str = "cuda"
    # BDELF generate eval: defaults to legacy (full AdaLN), consistent with training
    use_fast_infer: bool = False
    # Online eval subsample; None / omitted runs the full eval split
    eval_sample_count: Optional[int] = None
    eval_sample_seed: int = 42
    # Prefix ratio on eval chunks for generative PPL (0 = unconditional)
    eval_prefix_ratio: float = 0.5
    # BD3LM sampling steps during online generative eval
    eval_gen_steps: int = 128
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FL_BatchConfig:
    _YAML_REQUIRED = frozenset(
        {"name", "batch_size", "grad_accum_steps", "num_params_m"}
    )

    name: str = "prototype"
    batch_size: int = 4
    grad_accum_steps: int = 1
    num_params_m: float = 100.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FL_TrainConfig:
    """Composed training config resolved by convention from sub-configs."""

    name: str
    model: str
    model_config: str
    variant: TrainVariant
    dataset: str
    preprocess: str
    checkpoint_root: str
    batch_size: int
    grad_accum_steps: int
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
    log_plot_every: int
    eval_every: int
    save_every: int
    snapshot_every: int
    num_workers: int
    resume: bool
    seed: int
    eval_model: str
    eval_model_dtype: TrainDtype
    eval_model_device: str
    eval_use_fast_infer: bool
    eval_sample_count: Optional[int]
    eval_sample_seed: int
    eval_prefix_ratio: float
    eval_gen_steps: int
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def seq_tokens(self) -> int:
        chunk = int(self.extra.get("chunk_length", 1024))
        if self.model in ("bd3lm", "bdelf"):
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
    def target_tokens(self) -> int | None:
        raw = self.extra.get("target_tokens")
        return int(raw) if raw is not None else None


_SUBCONFIG: Dict[str, tuple[Type[Any], frozenset[str]]] = {
    "hardware": (FL_HardwareConfig, FL_HardwareConfig._YAML_REQUIRED),
    "optimizer": (FL_OptimizerConfig, FL_OptimizerConfig._YAML_REQUIRED),
    "schedule": (FL_ScheduleConfig, FL_ScheduleConfig._YAML_REQUIRED),
    "eval": (FL_EvalConfig, FL_EvalConfig._YAML_REQUIRED),
    "batch": (FL_BatchConfig, FL_BatchConfig._YAML_REQUIRED),
}

_MODEL_SCOPED_KINDS = frozenset({"batch"})


def _parse_train_ref(model: str, config_name: str | None = None) -> tuple[str, str]:
    if config_name is None:
        if "/" not in model:
            raise ValueError(
                f"Invalid train ref {model!r}, expected model/name (e.g. ar/100m-fast)"
            )
        model, config_name = model.split("/", 1)

    if model not in _TRAIN_MODELS:
        raise ValueError(
            f"Unknown model {model!r}. Expected one of: {', '.join(_TRAIN_MODELS)}"
        )
    if not _MODEL_CONFIG_RE.fullmatch(config_name):
        raise ValueError(
            f"Invalid config name {config_name!r}, expected {{100m,300m,900m}}-{{fast,full}}"
        )
    return model, config_name


def _parse_model_config_variant(config_name: str) -> tuple[str, TrainVariant]:
    match = _MODEL_CONFIG_RE.fullmatch(config_name)
    if match is None:
        raise ValueError(f"Invalid config name {config_name!r}")
    return match.group(1), match.group(2)  # type: ignore[return-value]


def _resolve_eval_ref(model_config: str, variant: TrainVariant) -> str:
    if model_config == "900m" and variant == "fast":
        return "gpt2-large-cpu"
    return "gpt2-large-cuda"


def _subconfig_path(kind: str, name: str, *, model: str | None = None) -> Path:
    if kind in _MODEL_SCOPED_KINDS:
        if model is None:
            raise ValueError(f"{kind} sub-config requires model")
        return CONFIG_DIR / kind / model / f"{name}.yaml"
    return CONFIG_DIR / kind / f"{name}.yaml"


def _load_subconfig(kind: str, name: str, *, model: str | None = None) -> Any:
    if name == "prototype":
        raise ValueError(f"Prototype {kind} config cannot be instantiated.")
    cls, required = _SUBCONFIG[kind]
    path = _subconfig_path(kind, name, model=model)
    if not path.is_file():
        available = ", ".join(list_subconfigs(kind, model=model)) or "<none>"
        raise FileNotFoundError(
            f"Config {path} does not exist. Available: {available}"
        )
    return load_yaml_config(cls, path, required=required)


def _merge_extra(*parts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return merged


def _validate_dtype(dtype: str, *, path: str, label: str) -> None:
    if dtype not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"{path}: unsupported {label} {dtype!r}")


def _resolve_max_steps(schedule: FL_ScheduleConfig) -> int:
    if schedule.max_steps < 1:
        raise ValueError(f"schedule {schedule.name}: max_steps must be >= 1")
    return schedule.max_steps


def compose_train_config(
    model: str,
    config_name: str | None = None,
    *,
    dataset: str,
    preprocess: str,
) -> FL_TrainConfig:
    """Merge sub-configs by naming convention.

    ``config_name`` must be ``{100m,300m,900m}-{fast,full}``. Sub-config refs:
      - hardware ← variant
      - optimizer ← model size
      - schedule ← variant
      - eval ← ``gpt2-large-cuda`` (``900m-fast`` uses CPU eval)
      - batch ← ``batch/<model>/<config_name>.yaml``

    ``dataset`` / ``preprocess`` are supplied at launch (not from yaml).
    """
    model, config_name = _parse_train_ref(model, config_name)
    model_config, variant = _parse_model_config_variant(config_name)
    run_name = f"{model}-{config_name}"

    hardware_name = _HARDWARE_BY_VARIANT[variant]
    eval_name = _resolve_eval_ref(model_config, variant)

    hardware = _load_subconfig("hardware", hardware_name)
    optimizer = _load_subconfig("optimizer", model_config)
    schedule = _load_subconfig("schedule", variant)
    eval_cfg = _load_subconfig("eval", eval_name)
    batch = _load_subconfig("batch", config_name, model=model)
    chunk_length = get_preprocess(preprocess).chunk_length

    if schedule.variant != variant:
        raise ValueError(
            f"{run_name}: schedule.variant={schedule.variant!r} != {variant!r}"
        )

    _validate_dtype(optimizer.dtype, path=run_name, label="dtype")
    _validate_dtype(eval_cfg.eval_model_dtype, path=run_name, label="eval_model_dtype")

    if eval_cfg.eval_sample_count is not None and eval_cfg.eval_sample_count < 1:
        raise ValueError(
            f"{run_name}: eval_sample_count must be >= 1 when set, "
            f"got {eval_cfg.eval_sample_count}"
        )
    if not (0.0 <= eval_cfg.eval_prefix_ratio < 1.0):
        raise ValueError(
            f"{run_name}: eval_prefix_ratio must be in [0, 1), "
            f"got {eval_cfg.eval_prefix_ratio}"
        )
    if eval_cfg.eval_gen_steps < 1:
        raise ValueError(
            f"{run_name}: eval_gen_steps must be >= 1, got {eval_cfg.eval_gen_steps}"
        )

    if batch.batch_size < 1 or batch.grad_accum_steps < 1 or hardware.world_size < 1:
        raise ValueError(f"{run_name}: batch/world_size must be >= 1")

    max_steps = _resolve_max_steps(schedule)

    tokens_per_step = (
        batch.batch_size
        * batch.grad_accum_steps
        * hardware.world_size
        * (
            chunk_length
            if model in ("bd3lm", "bdelf")
            else max(1, chunk_length - 1)
        )
    )
    target_tokens = max_steps * tokens_per_step

    extra = _merge_extra(
        hardware.extra,
        optimizer.extra,
        schedule.extra,
        eval_cfg.extra,
        batch.extra,
        {
            "chunk_length": chunk_length,
            "tokens_per_optimizer_step": tokens_per_step,
            "target_tokens": target_tokens,
            "config_refs": {
                "hardware": hardware_name,
                "optimizer": model_config,
                "schedule": variant,
                "eval": eval_name,
                "batch": f"{model}/{config_name}",
                "dataset": dataset,
                "preprocess": preprocess,
            },
        },
    )

    return FL_TrainConfig(
        name=run_name,
        model=model,
        model_config=model_config,
        variant=variant,
        dataset=dataset,
        preprocess=preprocess,
        checkpoint_root=CHECKPOINT_ROOT,
        batch_size=batch.batch_size,
        grad_accum_steps=batch.grad_accum_steps,
        world_size=hardware.world_size,
        dtype=optimizer.dtype,
        max_steps=max_steps,
        learning_rate=optimizer.learning_rate,
        weight_decay=optimizer.weight_decay,
        beta1=optimizer.beta1,
        beta2=optimizer.beta2,
        grad_clip=optimizer.grad_clip,
        warmup_steps=schedule.warmup_steps,
        min_lr_ratio=schedule.min_lr_ratio,
        log_plot_every=schedule.log_plot_every,
        eval_every=schedule.eval_every,
        save_every=schedule.save_every,
        snapshot_every=schedule.snapshot_every,
        num_workers=hardware.num_workers,
        resume=schedule.resume,
        seed=schedule.seed,
        eval_model=eval_cfg.eval_model,
        eval_model_dtype=eval_cfg.eval_model_dtype,
        eval_model_device=eval_cfg.eval_model_device,
        eval_use_fast_infer=eval_cfg.use_fast_infer,
        eval_sample_count=eval_cfg.eval_sample_count,
        eval_sample_seed=eval_cfg.eval_sample_seed,
        eval_prefix_ratio=eval_cfg.eval_prefix_ratio,
        eval_gen_steps=eval_cfg.eval_gen_steps,
        extra=extra,
    )


def list_subconfigs(kind: str, *, model: str | None = None) -> List[str]:
    subdir = CONFIG_DIR / kind
    if not subdir.is_dir():
        return []

    if kind in _MODEL_SCOPED_KINDS:
        if model is not None:
            model_dir = subdir / model
            if not model_dir.is_dir():
                return []
            return sorted(
                path.stem
                for path in model_dir.glob("*.yaml")
                if path.stem != "prototype"
            )
        names: List[str] = []
        for model_dir in sorted(subdir.iterdir()):
            if not model_dir.is_dir():
                continue
            for path in sorted(model_dir.glob("*.yaml")):
                if path.stem != "prototype":
                    names.append(f"{model_dir.name}/{path.stem}")
        return names

    return sorted(
        path.stem
        for path in subdir.glob("*.yaml")
        if path.stem != "prototype"
    )


def list_train_models() -> List[str]:
    if not BATCH_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in BATCH_DIR.iterdir()
        if path.is_dir() and path.name != "prototype"
    )


def list_train_configs(model: str | None = None) -> List[str]:
    return list_subconfigs("batch", model=model)


def get_train_config(
    model: str,
    config_name: str | None = None,
    *,
    dataset: str,
    preprocess: str,
) -> FL_TrainConfig:
    return compose_train_config(
        model, config_name, dataset=dataset, preprocess=preprocess,
    )


def resolve_train_config_path(config_arg: str) -> Path:
    """Return the batch yaml used as the train-config anchor."""
    as_path = Path(config_arg)
    if as_path.suffix in (".yaml", ".yml") and as_path.is_file():
        return as_path
    model, config_name = _parse_train_ref(config_arg, None)
    path = BATCH_DIR / model / f"{config_name}.yaml"
    if not path.is_file():
        available = ", ".join(list_train_configs()) or "<none>"
        raise FileNotFoundError(
            f"Train config {path} does not exist. Available: {available}"
        )
    return path

