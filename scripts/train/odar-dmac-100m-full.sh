#!/usr/bin/env bash
# ODAR 100m full + DMA-commit（T2：主损失 commitment；λ 进指纹 → 新 hash 从头训）。
# 经 slurm/sbatch-train.sh 提交；默认 4 GPU（prototype.slurm）。
# 其余配方对齐 odar-100m-full / ELF-100m full。
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
  --model odar \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  --set eval.gen_eval_samples=32 \
  --set schedule.warmup_ratio=0.1 \
  --set schedule.min_lr_ratio=1.0 \
  --set schedule.target_tokens=45200000000 \
  --set model.dma_commit_lambda=0.1 \
  --set model.dma_commit_t0=0.0 \
  "$@"
