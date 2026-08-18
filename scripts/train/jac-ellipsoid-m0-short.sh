#!/usr/bin/env bash
# JacEllipsoid M0 第三步：冻 DiT，只训 decode CE（短 token）。
# 冻 DiT 后无 Muon 可训 2D 隐层，须 AdamW（--set schedule.use_muon=false）。
# 对照（进哈希）：
#   --set model.qda_mode=softmax
#   --set model.qda_mode=isotropic
#   --set model.qda_mode=learned
#   --set model.qda_diag=true
# 经 sbatch-train / launch-train 提交；本脚本不直接占远端 GPU。
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

DEFAULT_TABLE="temp/ideas/jac-ellipsoid/result/m0/tables.pt"
QDA_TABLE="${QDA_TABLE:-}"
if [[ -z "$QDA_TABLE" && -f "$ROOT/$DEFAULT_TABLE" ]]; then
  QDA_TABLE="$DEFAULT_TABLE"
fi

SET_ARGS=(
  --set model.qda_decode_only=true
  --set model.freeze_dit=true
  --set schedule.use_muon=false
  --set schedule.warmup_ratio=0
  --set schedule.min_lr_ratio=1.0
  --set schedule.target_tokens=2621440000
  --set eval.gen_eval_samples=32
)
if [[ -n "$QDA_TABLE" ]]; then
  SET_ARGS+=(--set "model.qda_table_path=$QDA_TABLE")
fi

exec "$PY" train.py \
  --model jac_ellipsoid \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  "${SET_ARGS[@]}" \
  "$@"
