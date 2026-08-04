#!/usr/bin/env bash
# LateCE 100m full（均匀 t + 晚段 CE；无 decode 分支）。经 slurm/sbatch-train.sh 提交。
# 数据与 ELF 相同：--preprocess elf。batch_size=16（alloc 查表：4090 / global_bs=512 / ws=2）。
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
  --model late_ce \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval
