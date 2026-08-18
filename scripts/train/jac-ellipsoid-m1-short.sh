#!/usr/bin/env bash
# JacEllipsoid M1 短训：加载模板句 native ∇m 表，冻 DiT，只训 decode CE。
# 须先跑 source/m1_probe.py build。对照同 M0（--set model.qda_mode=…）。
# 表路径：QDA_TABLE 或默认 temp/ideas/jac-ellipsoid/result/m1/tables.pt
# （temp/ 不同步，远端须先拷表）。经 sbatch-train / launch-train 提交。
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

QDA_TABLE="${QDA_TABLE:-temp/ideas/jac-ellipsoid/result/m1/tables.pt}"
if [[ ! -f "$ROOT/$QDA_TABLE" ]]; then
  echo "缺少 M1 表: $QDA_TABLE（先跑 bash scripts/eval/jac-ellipsoid-m1-probe.sh build）" >&2
  exit 1
fi

exec "$PY" train.py \
  --model jac_ellipsoid \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  --set "model.qda_table_path=$QDA_TABLE" \
  --set model.qda_decode_only=true \
  --set model.freeze_dit=true \
  --set schedule.use_muon=false \
  --set schedule.warmup_ratio=0 \
  --set schedule.min_lr_ratio=1.0 \
  --set schedule.target_tokens=2621440000 \
  --set eval.gen_eval_samples=32 \
  "$@"
