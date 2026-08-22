#!/usr/bin/env bash
# =============================================================================
# 训练启动脚本模板（复制为 scripts/train/<name>.sh 后修改）
# =============================================================================
# - 工作目录须为仓库根（本地直接跑，或经包装器拉起）
# - 只写「怎么训」：train.py 参数与可选的模型专属环境变量
# - 不要写 #SBATCH；资源与集群环境由 slurm/prototype.slurm + sbatch-train.sh 负责
#
# 本地（fast 冒烟）::
#   bash scripts/train/<name>.sh
# 远端 full（Slurm）::
#   bash slurm/sbatch-train.sh <name>
#   bash slurm/sbatch-train.sh <name> --name my-job --exclude=cls1-srv2
# common 远端（须经 launch-train；勿直接跑本脚本）::
#   bash scripts/launch-train.sh <name> --server <服务名> --gpus 0,1
# =============================================================================
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

# 可选：模型专属环境变量（例）
# export COLA_VAE_CHECKPOINT=cache/checkpoints/artifacts/cola_vae/<tag>/checkpoint_latest.pt
# 或: export COLA_VAE_TAG=<tag>  （从 full/cola_vae/<hash> 拷贝到 artifacts）
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
# export NCCL_TIMEOUT=3600

exec "$PY" train.py \
  --model <model> \
  --config 100m-full \
  --dataset owt \
  --preprocess default \
  --generate eval \
  "$@"
