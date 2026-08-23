"""latent Stage-1 长度课程：owt-seg512 → owt-bucket 四阶段采样。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from preprocess.owt_split import bucket_pad_length
from preprocess.preprocess import FL_PreprocessedDataset, _PreprocessedSplitDataset, get_preprocessed

CURRICULUM_DIR = Path(__file__).resolve().parents[1] / "config" / "train" / "curriculum"


@dataclass(frozen=True)
class CurriculumStageSpec:
    name: str
    effective_budget: int
    graph_l: int
    global_seq_batch: int
    dataset: str  # seg512 | bucket
    mix: Dict[int, float]


@dataclass(frozen=True)
class LatentCurriculumSpec:
    name: str
    effective_target_tokens: int
    seg512_preprocess: str
    bucket_preprocess: str
    observation_window_tokens: int
    stages: List[CurriculumStageSpec]


def load_curriculum_spec(name: str) -> LatentCurriculumSpec:
    path = CURRICULUM_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Curriculum config not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: root must be a mapping")
    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError(f"{path}: stages must be a non-empty list")
    stages: List[CurriculumStageSpec] = []
    for entry in stages_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: each stage must be a mapping")
        mix_raw = entry.get("mix") or {}
        if not isinstance(mix_raw, dict) or not mix_raw:
            raise ValueError(f"{path}: stage {entry.get('name')!r} mix must be non-empty")
        mix = {int(k): float(v) for k, v in mix_raw.items()}
        total = sum(mix.values())
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError(
                f"{path}: stage {entry.get('name')!r} mix must sum to 1, got {total}"
            )
        stages.append(
            CurriculumStageSpec(
                name=str(entry["name"]),
                effective_budget=int(entry["effective_budget"]),
                graph_l=int(entry["graph_l"]),
                global_seq_batch=int(entry["global_seq_batch"]),
                dataset=str(entry["dataset"]),
                mix=mix,
            )
        )
    budget_sum = sum(s.effective_budget for s in stages)
    target = int(raw["effective_target_tokens"])
    if budget_sum != target:
        raise ValueError(
            f"{path}: stage budgets sum to {budget_sum}, "
            f"effective_target_tokens={target}"
        )
    return LatentCurriculumSpec(
        name=str(raw.get("name", name)),
        effective_target_tokens=target,
        seg512_preprocess=str(raw["seg512_preprocess"]),
        bucket_preprocess=str(raw["bucket_preprocess"]),
        observation_window_tokens=int(raw.get("observation_window_tokens", 0)),
        stages=stages,
    )


def curriculum_fingerprint_piece(spec: LatentCurriculumSpec) -> Dict[str, Any]:
    path = CURRICULUM_DIR / f"{spec.name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _bucket_indices(split: _PreprocessedSplitDataset) -> Dict[int, np.ndarray]:
    if not split.meta.has_lengths or split._lengths is None:
        raise ValueError(
            f"Split '{split.split}' has no .len metadata; bucket curriculum requires lengths"
        )
    pools: Dict[int, List[int]] = {256: [], 512: [], 1024: [], 2048: []}
    lengths = split._lengths
    for idx in range(len(split)):
        eff = int(lengths[idx])
        bucket = bucket_pad_length(eff)
        if bucket is not None and bucket in pools:
            pools[bucket].append(idx)
    return {
        b: np.asarray(idxs, dtype=np.int64)
        for b, idxs in pools.items()
        if idxs
    }


def batch_graph_l(stage_graph_l: int, bucket: int) -> int:
    """本 opt 步的训练图长。

    同一步已是单桶。不把短桶垫到阶段最大 L（注意力按 L² 计），
    也不按桶原生四档切换：full 开 ``torch.compile``，档数×微批组合会
    让 Inductor 多图来回切，CUDA caching allocator 按 size-class 囤块，
    峰值近似「各档激活之和」而非 max。折中为**每阶段最多两档**，
    微批仍按阶段最大 L 固定（不随短档加大 B）：

    * S1/S2（cap≤512）：钉死 512（256 桶 pad 到 512）
    * S3（cap=1024）：256/512 → 512，1024 → 1024
    * S4（cap=2048）：≤1024 → 1024，2048 → 2048
    """
    if bucket < 1:
        raise ValueError(f"bucket must be positive, got {bucket}")
    if bucket > stage_graph_l:
        raise ValueError(
            f"bucket={bucket} exceeds stage graph_l={stage_graph_l}"
        )
    if stage_graph_l <= 512:
        return stage_graph_l
    if stage_graph_l == 1024:
        return 1024 if bucket >= 1024 else 512
    if stage_graph_l == 2048:
        return 2048 if bucket >= 2048 else 1024
    return stage_graph_l


def resolve_stage_batch_sizes(
    stages: List[CurriculumStageSpec],
    *,
    default_batch_size: int,
    raw: Any | None,
) -> Dict[int, int]:
    """按图长 ``graph_l`` 解析每 GPU 微批；缺省用 ``default_batch_size``。"""
    parsed: Dict[int, int] = {}
    if raw is not None:
        if not isinstance(raw, dict) or not raw:
            raise ValueError(
                "batch.stage_batch_size must be a non-empty mapping (graph_l → batch)"
            )
        for key, val in raw.items():
            try:
                length = int(key)
                size = int(val)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"batch.stage_batch_size 键/值须为整数，got {key!r}: {val!r}"
                ) from exc
            if length < 1 or size < 1:
                raise ValueError(
                    f"batch.stage_batch_size[{length}]={size} 须为正整数"
                )
            parsed[length] = size
    out: Dict[int, int] = {}
    for stage in stages:
        out[stage.graph_l] = parsed.get(stage.graph_l, default_batch_size)
    return out


def stage_grad_accum(
    stage: CurriculumStageSpec,
    *,
    batch_size: int,
    world_size: int,
) -> int:
    denom = batch_size * world_size
    if stage.global_seq_batch % denom != 0:
        raise ValueError(
            f"Stage {stage.name}: global_seq_batch={stage.global_seq_batch} "
            f"not divisible by batch_size*world_size={denom}"
        )
    return stage.global_seq_batch // denom


@dataclass
class _StageRuntime:
    spec: CurriculumStageSpec
    split: _PreprocessedSplitDataset
    pools: Dict[int, np.ndarray]


@dataclass
class LatentCurriculumSampler:
    """按全局有效 token 推进阶段；同一步仅采同一桶。"""

    spec: LatentCurriculumSpec
    pad_token_id: int
    seed: int
    world_size: int
    batch_size: int
    stage_batch_sizes: Dict[int, int] = field(default_factory=dict)
    effective_tokens_global: int = 0
    _stage_idx: int = 0
    _last_bucket: int = 0
    _last_batch_l: int = 0
    _stages: List[_StageRuntime] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        spec: LatentCurriculumSpec,
        *,
        dataset: str,
        pad_token_id: int,
        seed: int,
        world_size: int,
        batch_size: int,
        stage_batch_sizes: Dict[int, int] | None = None,
    ) -> LatentCurriculumSampler:
        seg512 = get_preprocessed(spec.seg512_preprocess, dataset)
        bucket = get_preprocessed(spec.bucket_preprocess, dataset)
        stages: List[_StageRuntime] = []
        for stage in spec.stages:
            if stage.dataset == "seg512":
                split = seg512.load_split("train")
            elif stage.dataset == "bucket":
                split = bucket.load_split("train")
            else:
                raise ValueError(f"Unknown stage dataset {stage.dataset!r}")
            if stage.dataset == "bucket":
                pools = _bucket_indices(split)
            else:
                pools = {512: np.arange(len(split), dtype=np.int64)}
            for bucket_key in stage.mix:
                if bucket_key not in pools or pools[bucket_key].size == 0:
                    raise ValueError(
                        f"Stage {stage.name}: bucket {bucket_key} has no samples in "
                        f"{stage.dataset} train split"
                    )
            stages.append(_StageRuntime(spec=stage, split=split, pools=pools))
        sizes = dict(stage_batch_sizes or {})
        if not sizes:
            sizes = {stage.graph_l: batch_size for stage in spec.stages}
        return cls(
            spec=spec,
            pad_token_id=pad_token_id,
            seed=seed,
            world_size=world_size,
            batch_size=batch_size,
            stage_batch_sizes=sizes,
            _stages=stages,
        )

    @property
    def current_stage(self) -> CurriculumStageSpec:
        return self._stages[self._stage_idx].spec

    @property
    def graph_l(self) -> int:
        return self.current_stage.graph_l

    @property
    def current_batch_size(self) -> int:
        return int(self.stage_batch_sizes.get(self.graph_l, self.batch_size))

    @property
    def grad_accum_steps(self) -> int:
        return stage_grad_accum(
            self.current_stage,
            batch_size=self.current_batch_size,
            world_size=self.world_size,
        )

    def _stage_end_tokens(self, stage_idx: int) -> int:
        return sum(s.effective_budget for s in self.spec.stages[: stage_idx + 1])

    def sync_stage(self) -> None:
        while (
            self._stage_idx < len(self._stages) - 1
            and self.effective_tokens_global >= self._stage_end_tokens(self._stage_idx)
        ):
            self._stage_idx += 1

    def add_effective_tokens(self, count: int) -> None:
        self.effective_tokens_global += int(count)
        self.sync_stage()

    def is_complete(self) -> bool:
        return self.effective_tokens_global >= self.spec.effective_target_tokens

    def _rng(self, step: int) -> np.random.Generator:
        return np.random.default_rng(
            self.seed + step * 9973 + self._stage_idx * 7919,
        )

    def _sample_bucket(self, rng: np.random.Generator, stage: CurriculumStageSpec) -> int:
        keys = sorted(stage.mix.keys())
        weights = np.asarray([stage.mix[k] for k in keys], dtype=np.float64)
        return int(rng.choice(keys, p=weights / weights.sum()))

    def _sample_indices(
        self,
        pool: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if pool.size < count:
            raise ValueError(
                f"Bucket pool size {pool.size} < global batch {count}; "
                "cannot form a full batch from one bucket"
            )
        return rng.choice(pool, size=count, replace=False)

    def _pad_row(self, row: torch.Tensor, eff_len: int, graph_l: int) -> torch.Tensor:
        row = row[:eff_len]
        if row.numel() > graph_l:
            row = row[:graph_l]
        if row.numel() < graph_l:
            pad = torch.full(
                (graph_l - row.numel(),),
                self.pad_token_id,
                dtype=row.dtype,
            )
            row = torch.cat([row, pad])
        return row

    def fetch_batch(
        self,
        step: int,
        rank: int,
    ) -> tuple[torch.Tensor, int]:
        """返回 (batch, 本 rank 有效 token 数)。"""
        self.sync_stage()
        rt = self._stages[self._stage_idx]
        stage = rt.spec
        micro_bs = self.current_batch_size
        global_batch = micro_bs * self.world_size
        stage_global = stage.global_seq_batch
        if global_batch > stage_global:
            raise ValueError(
                f"Curriculum stage {stage.name}: batch_size*world_size={global_batch} "
                f"> global_seq_batch={stage_global}"
            )
        if stage_global % global_batch != 0:
            raise ValueError(
                f"Curriculum stage {stage.name}: global_seq_batch={stage_global} "
                f"not divisible by batch_size*world_size={global_batch}"
            )
        micros_per_opt = stage_global // global_batch
        opt_step = step // micros_per_opt
        micro_in_opt = step % micros_per_opt

        rng = self._rng(opt_step)
        bucket = self._sample_bucket(rng, stage)
        batch_l = batch_graph_l(stage.graph_l, bucket)
        self._last_bucket = bucket
        self._last_batch_l = batch_l
        indices = self._sample_indices(rt.pools[bucket], stage_global, rng)
        micro_start = micro_in_opt * global_batch
        rank_indices = indices[micro_start + rank * micro_bs : micro_start + (rank + 1) * micro_bs]

        rows: List[torch.Tensor] = []
        eff_sum = 0
        split = rt.split
        for idx in rank_indices:
            item = split[int(idx)]
            row = item["input_ids"]
            eff = int(item["length"]) if "length" in item else int(row.numel())
            eff_sum += eff
            rows.append(self._pad_row(row, eff, batch_l))
        out = torch.stack(rows, dim=0)
        if torch.cuda.is_available():
            out = out.pin_memory()
        return out, eff_sum

    def curriculum_state(self) -> Dict[str, Any]:
        stage = self.current_stage
        return {
            "stage": stage.name,
            "stage_idx": self._stage_idx,
            "graph_l": stage.graph_l,
            "batch_l": self._last_batch_l,
            "bucket": self._last_bucket,
            "batch_size": self.current_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "effective_tokens_global": self.effective_tokens_global,
            "target_effective_tokens": self.spec.effective_target_tokens,
        }


def load_curriculum_datasets(
    spec: LatentCurriculumSpec,
    dataset: str,
) -> tuple[FL_PreprocessedDataset, FL_PreprocessedDataset]:
    seg512 = get_preprocessed(spec.seg512_preprocess, dataset)
    bucket = get_preprocessed(spec.bucket_preprocess, dataset)
    return seg512, bucket


def resolve_curriculum_spec_name(preprocess_name: str) -> str:
    from preprocess import get_preprocess

    pp = get_preprocess(preprocess_name)
    name = pp.extra.get("curriculum")
    if not name:
        raise ValueError(
            f"preprocess {preprocess_name!r} has no extra.curriculum pointer"
        )
    return str(name)
