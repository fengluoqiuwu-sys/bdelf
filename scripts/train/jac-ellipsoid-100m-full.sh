#!/usr/bin/env bash
# JacEllipsoid 100m full（ELF 复制：decode 改为 QDA 观测，中段 MSE 不动）。
# 默认 jacobian G_k（冻 T5 表闭式精度）。对照：
#   点 softmax     --set model.qda_mode=softmax
#   Plaid 式 G∝I   --set model.qda_mode=isotropic
#   同秩可学 Σ     --set model.qda_mode=learned
# 经 slurm/sbatch-train.sh 或 launch-train.sh 提交；本脚本不直接占远端 GPU。
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
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  "$@"
