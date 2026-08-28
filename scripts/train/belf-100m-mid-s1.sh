#!/usr/bin/env bash
# BELF 100m Stage1 中档：10B@128（owt-seg512）；峰值后恒定 lr（不套 latent WSD）。
# 无 Stage2 mid。正式 45B 见 scripts/train/belf-100m-full-s1.sh。
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
  --model belf \
  --config 100m-mid \
  --dataset owt \
  --preprocess belf-relf-s1-mid \
  --generate eval \
  --set schedule.target_tokens=10000000000 \
  --set schedule.warmup_ratio=0.005 \
  --set schedule.min_lr_ratio=1 \
  "$@"
