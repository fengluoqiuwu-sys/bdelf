#!/usr/bin/env bash
# JacEllipsoid M3：已有 ckpt + 换 M1 表短 ft（默认 300m-full）。
# 环境变量：
#   INIT_CKPT  必填，相对仓库根的 checkpoint_latest.pt（ELF 或 jac_ellipsoid）
#   QDA_TABLE  默认 result/m1/tables.pt
#   CONFIG     默认 300m-full（可 100m-full / 900m-full）
#   FREEZE_DIT 设为 1 则冻 DiT 只训读出
# 经 sbatch-train / launch-train 提交。不把 Cola 16 维 latent 接 512 维椭球。
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

INIT_CKPT="${INIT_CKPT:-}"
if [[ -z "$INIT_CKPT" ]]; then
  echo "须设置 INIT_CKPT=cache/checkpoints/full/<model>/<hash>/checkpoint_latest.pt" >&2
  exit 1
fi
if [[ ! -f "$ROOT/$INIT_CKPT" && ! -f "$INIT_CKPT" ]]; then
  echo "INIT_CKPT 不存在: $INIT_CKPT" >&2
  exit 1
fi

QDA_TABLE="${QDA_TABLE:-temp/ideas/jac-ellipsoid/result/m1/tables.pt}"
if [[ ! -f "$ROOT/$QDA_TABLE" ]]; then
  echo "缺少 QDA 表: $QDA_TABLE（先跑 M1 build，或改 QDA_TABLE）" >&2
  exit 1
fi

CONFIG="${CONFIG:-300m-full}"
SET_ARGS=(
  --init-ckpt "$INIT_CKPT"
  --set "model.qda_table_path=$QDA_TABLE"
  --set schedule.warmup_ratio=0
  --set schedule.min_lr_ratio=1.0
  --set schedule.target_tokens=2621440000
  --set eval.gen_eval_samples=32
)
if [[ "${FREEZE_DIT:-0}" == "1" ]]; then
  SET_ARGS+=(--set model.freeze_dit=true --set model.qda_decode_only=true --set schedule.use_muon=false)
fi

exec "$PY" train.py \
  --model jac_ellipsoid \
  --config "$CONFIG" \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  "${SET_ARGS[@]}" \
  "$@"
