#!/usr/bin/env bash
# RELF 100m Stage2 扩展：5B 混桶（owt-bucket，graph_l=2048）。
# 启动时校验同参 Stage1 已完成，并从 Stage1 checkpoint_latest 的 EMA 初始化（不恢复优化器）。
# 本 run 另开哈希目录，hardware.json 独立，可用与 Stage1 不同的 GPU。
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
  --preprocess belf-relf-s2 \
  --generate eval \
  --set schedule.target_tokens=5000000000 \
  --set schedule.warmup_ratio=0.005 \
  --set schedule.min_lr_ratio=1 \
  "$@"
