#!/usr/bin/env bash
# Posβ 100m full + SC-CFG：ELF-B 骨架、位置相关插值（默认 κ=1）。
#   - warmup_ratio=0.1（≈ warmup_epochs 0.5 / 5 epochs）
#   - min_lr_ratio=1.0 → warmup 后 constant LR=0.002
#   - target_tokens=45.2B（5 × ~9.04B OWT）
# 对照 isotropic：`--set model.pos_beta_kappa=0`（会改 config-hash）。
# 经 slurm/sbatch-train.sh 提交；默认 4 GPU（prototype.slurm）。
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
  --model posbeta \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  --set eval.gen_eval_samples=32 \
  --set schedule.warmup_ratio=0.1 \
  --set schedule.min_lr_ratio=1.0 \
  --set schedule.target_tokens=45200000000 \
  "$@"
