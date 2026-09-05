#!/usr/bin/env bash
# latent_ovan 100m full（csh BlockVAE s1；本机 gpt2 + owt-seg512）。
# 共享 schedule 不改文件：关 Muon；token 预算 / warmup / min_lr / seed 用本仓库 full。
# 经 slurm/sbatch-train.sh 提交或本地直接跑。
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
  --model latent_ovan \
  --config 100m-full \
  --dataset owt \
  --preprocess owt-seg512 \
  --generate eval \
  --set schedule.use_muon=false \
  "$@"
