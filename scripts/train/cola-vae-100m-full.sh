#!/usr/bin/env bash
# Cola Stage-1 VAE 100m full；默认跳过在线 eval（eval.skip，不进哈希）。
# 经 slurm/sbatch-train.sh 提交或本地直接跑。
# 两阶段连跑用 scripts/train/cola-seq-100m-full.sh。
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
  --model cola_vae \
  --config 100m-full \
  --dataset owt \
  --preprocess default \
  --generate eval \
  "$@"
