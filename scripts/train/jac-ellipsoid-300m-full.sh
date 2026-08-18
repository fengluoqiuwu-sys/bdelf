#!/usr/bin/env bash
# JacEllipsoid 300m full（ELF 复制体；M3 规模抽查的从零配方）。
# 对照：--set model.qda_mode=softmax|isotropic|learned
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

exec "$PY" train.py \
  --model jac_ellipsoid \
  --config 300m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  "$@"
