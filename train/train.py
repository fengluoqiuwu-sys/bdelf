"""Training config loading for language-model pretraining."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

import yaml

from config_util import load_yaml_config
from models import resolve_model_config_path
from preprocess import get_preprocess

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "train"
BATCH_DIR = CONFIG_DIR / "batch"
CHECKPOINT_ROOT = "cache/checkpoints"

TrainVariant = Literal["fast", "full", "ultra"]
TrainDtype = Literal["bf16", "fp16", "fp32"]

TSub = TypeVar("TSub")

_TRAIN_MODELS = ("ar", "bd3lm", "bdelf")
_MODEL_CONFIG_RE = re.compile(r"^(100m|300m|900m)-(fast|full|ultra)$")
_HARDWARE_BY_VARIANT = {
    "fast": "fast-16gb",
    "full": "full-4x4090",
    "ultra": "full-8x4090",
}


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
    muon_learning_rate: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FL_ScheduleConfig:
    """Unified schedule: token budget drives max_steps; intervals are absolute steps.

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
    _YAML_REQUIRED = frozenset({"name", "eval_sample_seed"})

    name: str = "prototype"
    # Online eval subsample; None / omitted runs the full eval split
    eval_sample_count: Optional[int] = None
    eval_sample_seed: int = 42
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
    log_plot_step: int
    eval_step: int
    save_step: int
    snapshot_step: int
    num_workers: int
    resume: bool
    seed: int
    eval_sample_count: Optional[int]
    eval_sample_seed: int
    use_muon: bool = True
    muon_learning_rate: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
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
    def effective_tokens_per_optimizer_step(self) -> int:
        """Decode-equivalent tokens per step (BDELF: scaled by ``decoder_prob``)."""
        raw = self.extra.get("effective_tokens_per_optimizer_step")
        if raw is not None:
            return int(raw)
        return self.tokens_per_optimizer_step

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

_MODEL_SCOPED_KINDS = frozenset({"batch", "optimizer"})


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
            f"Invalid config name {config_name!r}, expected {{100m,300m,900m}}-{{fast,full,ultra}}"
        )
    return model, config_name


def _model_decoder_prob(model: str, model_config: str) -> float:
    if model != "bdelf":
        return 1.0
    path = resolve_model_config_path(model, model_config)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return float(cfg.get("decoder_prob", 0.2))


def _parse_model_config_variant(config_name: str) -> tuple[str, TrainVariant]:
    match = _MODEL_CONFIG_RE.fullmatch(config_name)
    if match is None:
        raise ValueError(f"Invalid config name {config_name!r}")
    return match.group(1), match.group(2)  # type: ignore[return-value]


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


def compose_train_config(
    model: str,
    config_name: str | None = None,
    *,
    dataset: str,
    preprocess: str,
) -> FL_TrainConfig:
    """Merge sub-configs by naming convention.

    ``config_name`` must be ``{100m,300m,900m}-{fast,full,ultra}``. Sub-config refs:
      - hardware ← variant
      - optimizer ← ``optimizer/<model>/<model size>.yaml``
      - schedule ← variant (``full``/``ultra`` derive ``max_steps`` from ``target_tokens``)
      - eval ← ``default``
      - batch ← ``batch/<model>/<config_name>.yaml``

    ``dataset`` / ``preprocess`` are supplied at launch (not from yaml).
    """
    model, config_name = _parse_train_ref(model, config_name)
    model_config, variant = _parse_model_config_variant(config_name)

    hardware_name = _HARDWARE_BY_VARIANT[variant]

    hardware = _load_subconfig("hardware", hardware_name)
    optimizer = _load_subconfig("optimizer", model_config, model=model)
    schedule = _load_subconfig("schedule", variant)
    run_name = f"{model}-{config_name}"
    if schedule.use_muon:
        run_name = f"{run_name}-muon"
    eval_cfg = _load_subconfig("eval", "default")
    batch = _load_subconfig("batch", config_name, model=model)
    chunk_length = get_preprocess(preprocess).chunk_length

    if schedule.variant != variant:
        raise ValueError(
            f"{run_name}: schedule.variant={schedule.variant!r} != {variant!r}"
        )

    _validate_dtype(optimizer.dtype, path=run_name, label="dtype")

    if eval_cfg.eval_sample_count is not None and eval_cfg.eval_sample_count < 1:
        raise ValueError(
            f"{run_name}: eval_sample_count must be >= 1 when set, "
            f"got {eval_cfg.eval_sample_count}"
        )

    if batch.batch_size < 1 or batch.grad_accum_steps < 1 or hardware.world_size < 1:
        raise ValueError(f"{run_name}: batch/world_size must be >= 1")

    raw_tokens_per_step = (
        batch.batch_size
        * batch.grad_accum_steps
        * hardware.world_size
        * (
            chunk_length
            if model in ("bd3lm", "bdelf")
            else max(1, chunk_length - 1)
        )
    )
    decoder_prob = _model_decoder_prob(model, model_config)
    if model == "bdelf":
        effective_tokens_per_step = max(
            1, round(raw_tokens_per_step * decoder_prob),
        )
    else:
        effective_tokens_per_step = raw_tokens_per_step
    resolved = _resolve_schedule(
        schedule,
        run_name=run_name,
        tokens_per_step=effective_tokens_per_step,
    )
    max_steps = resolved.max_steps
    target_tokens = max_steps * effective_tokens_per_step

    # BDELF max_steps is ~1/decoder_prob× longer (decode-equivalent token
    # budget). Scale absolute eval/save/plot intervals by the same factor so
    # wall-clock cadence stays comparable to AR/BD3LM.
    log_plot_step = resolved.log_plot_step
    eval_step = resolved.eval_step
    save_step = resolved.save_step
    snapshot_step = resolved.snapshot_step
    if model == "bdelf":
        if decoder_prob <= 0.0 or decoder_prob > 1.0:
            raise ValueError(
                f"{run_name}: decoder_prob must be in (0, 1], got {decoder_prob}"
            )
        branch_scale = max(1, round(1.0 / decoder_prob))
        log_plot_step = max(1, log_plot_step * branch_scale)
        eval_step = max(1, eval_step * branch_scale)
        save_step = max(1, save_step * branch_scale)
        snapshot_step = max(1, snapshot_step * branch_scale)

    extra = _merge_extra(
        hardware.extra,
        optimizer.extra,
        schedule.extra,
        eval_cfg.extra,
        batch.extra,
        {
            "chunk_length": chunk_length,
            "tokens_per_optimizer_step": raw_tokens_per_step,
            "effective_tokens_per_optimizer_step": effective_tokens_per_step,
            "decoder_prob": decoder_prob,
            "target_tokens": target_tokens,
            "schedule_branch_scale": (
                max(1, round(1.0 / decoder_prob)) if model == "bdelf" else 1
            ),
            "config_refs": {
                "hardware": hardware_name,
                "optimizer": f"{model}/{model_config}",
                "schedule": variant,
                "eval": "default",
                "batch": f"{model}/{config_name}",
                "dataset": dataset,
                "preprocess": preprocess,
            },
            "use_muon": schedule.use_muon,
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
        warmup_steps=resolved.warmup_steps,
        min_lr_ratio=schedule.min_lr_ratio,
        log_plot_step=log_plot_step,
        eval_step=eval_step,
        save_step=save_step,
        snapshot_step=snapshot_step,
        num_workers=hardware.num_workers,
        resume=schedule.resume,
        seed=schedule.seed,
        eval_sample_count=eval_cfg.eval_sample_count,
        eval_sample_seed=eval_cfg.eval_sample_seed,
        use_muon=schedule.use_muon,
        muon_learning_rate=optimizer.muon_learning_rate,
        muon_momentum=optimizer.muon_momentum,
        muon_ns_steps=optimizer.muon_ns_steps,
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

