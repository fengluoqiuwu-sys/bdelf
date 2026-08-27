"""分阶段训练：Stage2 绑定同参 Stage1 的 latest EMA。

Stage2 preprocess 若声明 ``predecessor_preprocess``，启动时：
1. 用同模型 / 同 ``--config`` / 同 dataset / 同 generate、以及除 batch 与
   ``schedule.target_tokens`` 外的 ``--set``，解析 Stage1 哈希目录；
2. 要求该目录已写出 ``complete.json``（训练正常跑完），否则直接退出；
3. 把 Stage1 ``checkpoint_latest.pt`` 的 EMA 写入 ``extra.init_ckpt``，
   本 Stage2 另开哈希目录（可换卡，不沿用 Stage1 ``hardware.json``）。

``batch.*`` 不参与 Stage1 定位（扩展阶段微批可因显存改小）。
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from preprocess import get_preprocess
from train.run_path import checkpoint_run_dir_from_cfg
from train.train import FL_TrainConfig, get_train_config

COMPLETE_FILENAME = "complete.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def predecessor_preprocess_name(preprocess: str) -> str:
    """空字符串表示无前置。"""
    raw = get_preprocess(preprocess).extra.get("predecessor_preprocess")
    return str(raw or "").strip()


def predecessor_target_tokens(preprocess: str) -> int:
    return int(get_preprocess(preprocess).extra.get("predecessor_target_tokens") or 0)


def predecessor_overrides(
    overrides: Mapping[str, Mapping[str, Any]] | None,
    *,
    target_tokens: int,
) -> dict[str, dict[str, Any]]:
    """从 Stage2 的 ``--set`` 还原 Stage1 定位用覆盖。

    丢掉 ``batch``（扩展可改微批）和 ``extra.init_ckpt``；把
    ``schedule.target_tokens`` 换成前置预算。
    """
    out: dict[str, dict[str, Any]] = copy.deepcopy(dict(overrides or {}))
    out.pop("batch", None)
    extra = dict(out.get("extra") or {})
    extra.pop("init_ckpt", None)
    extra.pop("init_from_ema", None)
    extra.pop("stage1_tokens_seen", None)
    if extra:
        out["extra"] = extra
    else:
        out.pop("extra", None)
    sched = dict(out.get("schedule") or {})
    sched["target_tokens"] = int(target_tokens)
    out["schedule"] = sched
    return out


def write_complete_marker(
    run_dir: Path,
    *,
    step: int,
    cfg: FL_TrainConfig,
    curriculum_state: dict[str, Any] | None = None,
) -> Path:
    """训练正常结束时写入；中断保存不写。"""
    effective: int | None = None
    if curriculum_state:
        raw = curriculum_state.get("effective_tokens_global")
        if raw is not None:
            effective = int(raw)
    if effective is None:
        if step > 0:
            effective = int(cfg.tokens_seen_after_step(step - 1))
        else:
            effective = 0
    payload = {
        "complete": True,
        "step": int(step),
        "max_steps": int(cfg.max_steps),
        "effective_tokens": int(effective),
        "run_relpath": cfg.extra.get("run_relpath"),
        "finished_at": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
    }
    path = Path(run_dir) / COMPLETE_FILENAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_complete_marker(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / COMPLETE_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("complete"):
        return None
    return data


def _latest_ckpt(run_dir: Path) -> Path:
    return Path(run_dir) / "checkpoint_latest.pt"


def bind_stage_predecessor(
    cfg: FL_TrainConfig,
    *,
    model: str,
    train_config: str,
    dataset: str,
    generate: str,
    world_size: int,
    overrides: Mapping[str, Mapping[str, Any]] | None,
    user_init_ckpt: str | None = None,
) -> str | None:
    """若当前 preprocess 声明前置，校验同参 Stage1 已完成并写入 extra。

    续训也必须写入 ``stage1_tokens_seen`` / ``init_ckpt`` / ``init_from_ema``：
    ``_thawed`` 与 ``q_ref`` 不进 checkpoint，进程重建后要靠这三项立刻解冻
    并从 Stage1 EMA 重冻 ``q_ref``。不覆盖本 run 的 live 权重（由
    ``checkpoint_latest`` 恢复）。失败则 ``SystemExit``。
    """
    pred = predecessor_preprocess_name(cfg.preprocess)
    if not pred:
        return None

    self_dir = checkpoint_run_dir_from_cfg(cfg)
    resuming = bool(cfg.resume and _latest_ckpt(self_dir).is_file())

    target = predecessor_target_tokens(cfg.preprocess)
    if target < 1:
        raise SystemExit(
            f"preprocess {cfg.preprocess!r} 声明了 predecessor_preprocess={pred!r}，"
            "但 predecessor_target_tokens 未设正整数"
        )

    pred_ov = predecessor_overrides(overrides, target_tokens=target)
    try:
        pred_cfg = get_train_config(
            model,
            train_config,
            dataset=dataset,
            preprocess=pred,
            generate=generate,
            world_size=world_size,
            overrides=pred_ov,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"解析同参 Stage1 失败: {exc}") from exc

    pred_dir = checkpoint_run_dir_from_cfg(pred_cfg)
    latest = _latest_ckpt(pred_dir)
    marker = read_complete_marker(pred_dir)
    if marker is None or not latest.is_file():
        rel = pred_dir.as_posix()
        raise SystemExit(
            f"Stage2 拒绝启动：同参 Stage1 尚未完成。\n"
            f"  期望目录: {rel}\n"
            f"  期望文件: {COMPLETE_FILENAME} 与 {latest.name}\n"
            f"  Stage1 哈希: {pred_cfg.name}\n"
            f"  请先跑完 scripts/train/{model}-100m-full-s1.sh"
            f"（相同 --model/--config/--dataset/--generate 与 model.*/optimizer.* 的 --set）"
        )

    repo = _repo_root()
    try:
        rel_ckpt = latest.resolve().relative_to(repo).as_posix()
    except ValueError:
        rel_ckpt = str(latest.resolve())

    if user_init_ckpt:
        raw = str(user_init_ckpt).strip().replace("\\", "/")
        if Path(raw).as_posix() != rel_ckpt and raw != latest.name:
            raise SystemExit(
                f"--init-ckpt={raw!r} 与自动解析的同参 Stage1 latest 不一致: {rel_ckpt}"
            )

    cfg.extra["init_ckpt"] = rel_ckpt
    cfg.extra["init_from_ema"] = True
    cfg.extra["stage1_tokens_seen"] = int(
        marker.get("effective_tokens")
        or pred_cfg.extra.get("curriculum_effective_tokens")
        or target
    )
    cfg.extra["stage1_run_relpath"] = pred_cfg.extra.get("run_relpath")
    if resuming:
        return (
            f"Stage2 续训 {self_dir.as_posix()}，已重绑 Stage1 extra "
            f"(不覆盖 live；hash={pred_cfg.name}, "
            f"tokens_seen={cfg.extra['stage1_tokens_seen']})"
        )
    return (
        f"Stage2 绑定 Stage1 EMA: {rel_ckpt} "
        f"(hash={pred_cfg.name}, tokens_seen={cfg.extra['stage1_tokens_seen']})"
    )
