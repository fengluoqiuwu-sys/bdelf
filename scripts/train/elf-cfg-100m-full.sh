#!/usr/bin/env bash
# ELF-B 100m full + SC-CFG，对齐论文 Tab.7 训练日程后从头开跑：
#   - warmup_ratio=0.1（≈ warmup_epochs 0.5 / 5 epochs）
#   - min_lr_ratio=1.0 → warmup 后 constant LR=0.002（等价官方 lr_schedule=constant）
#   - target_tokens=45.2B（5 × ~9.04B OWT）
# 在线 gen-eval 仍用 32 samples 控开销（正式复现评测另用 1000）。
# 经 slurm/sbatch-train.sh 提交；默认 4 GPU（prototype.slurm）。
# fingerprint：resolve_checkpoint.py … 下列 --set
#   → full/elf/4ab96e311b796009（无 checkpoint 时从 step 0 开训）
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
  --set eval.gen_eval_samples=32 \
  --set schedule.warmup_ratio=0.1 \
  --set schedule.min_lr_ratio=1.0 \
  --set schedule.target_tokens=45200000000 \
  "$@"
