#!/usr/bin/env bash
# ELF-B 100m full + SC-CFG（训练期 gen-eval 用 32 samples，控制评测开销）。
# 经 slurm/sbatch-train.sh 提交；默认 2 GPU（prototype.slurm）。
# fingerprint：resolve_checkpoint.py … --set eval.gen_eval_samples=32
#   → full/elf/57ef50375e85d826（无 checkpoint_latest 时从 step 0 开训）
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
  --generate eval \
  --set eval.gen_eval_samples=32
