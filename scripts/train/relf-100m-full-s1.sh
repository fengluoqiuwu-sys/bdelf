#!/usr/bin/env bash
# RELF 100m Stage1 主训：45B@512（owt-seg512）；峰值后恒定 lr（不套 latent WSD）。
# Stage2 见 scripts/train/relf-100m-full-s2.sh（须本脚本跑完，另开哈希目录，可换卡）。
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

exec "$PY" train.py \
  --model relf \
  --config 100m-full \
  --dataset owt \
  --preprocess belf-relf-s1 \
  --generate eval \
  --set schedule.target_tokens=45000000000 \
  --set schedule.warmup_ratio=0.005 \
  --set schedule.min_lr_ratio=1 \
  "$@"
