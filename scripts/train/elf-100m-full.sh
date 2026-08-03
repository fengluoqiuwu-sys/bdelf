#!/usr/bin/env bash
# ELF 100m full（含训练期 gen-eval）；经 slurm/sbatch-train.sh 提交或本地直接跑。
# 需要已缓存的 t5-small（登录节点可预热 ensure_t5_encoder_cached）。
# Rank0 gen-eval 时 peer 会卡在短 all_reduce，拉长 NCCL 超时避免误杀。
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
  --model elf \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval
