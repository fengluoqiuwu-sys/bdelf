#!/usr/bin/env bash
# M5 冻 d 扫 w_attr（默认 1）。覆盖：launch 后 -- --set model.attr_weight=3|15。
# 与 M1 同一 ELF 初始化 / 5k；snapshot 每 1000 步省盘。w=7 已是 M1 冻臂。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "找不到 Python（请创建 .venv 或激活环境）" >&2
  exit 1
fi

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

INIT_CKPT="${INIT_CKPT:-cache/checkpoints/full/elf/bd945c7dae40a939/checkpoint_latest.pt}"
ATTR_WEIGHT="${ATTR_WEIGHT:-1}"

exec "$PY" train.py \
  --model trace \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  --init-ckpt "$INIT_CKPT" \
  --set eval.gen_eval_samples=32 \
  --set schedule.warmup_ratio=0 \
  --set schedule.min_lr_ratio=1.0 \
  --set schedule.target_tokens=2621440000 \
  --set schedule.save_step=1000 \
  --set schedule.snapshot_step=1000 \
  --set schedule.eval_step=1000 \
  --set model.attr_freeze_d=true \
  --set model.attr_warmup_steps=0 \
  --set model.attr_estimate_n=512 \
  --set model.attr_weight="$ATTR_WEIGHT" \
  "$@"
