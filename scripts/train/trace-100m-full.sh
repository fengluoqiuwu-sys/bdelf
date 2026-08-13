#!/usr/bin/env bash
# TrACE 100m full：ELF-cfg 日程 + 训练期 L_attr，长训慢跟踪 d。
#   - warmup_ratio=0.1、min_lr_ratio=1.0、target_tokens=45.2B（同 elf-cfg）
#   - 推理 ACE=off（config/generate/trace/eval.yaml）
#   - attr_warmup_steps=1000 后开始估 d，之后每 1k opt-step × N=128 EMA 更新（β=0.9）
# 经 slurm/sbatch-train.sh 提交；默认 4 GPU（prototype.slurm）。
# 短 FT 冻 d：加 --set model.attr_freeze_d=true --set model.attr_warmup_steps=0
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
  --model trace \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  --set eval.gen_eval_samples=32 \
  --set schedule.warmup_ratio=0.1 \
  --set schedule.min_lr_ratio=1.0 \
  --set schedule.target_tokens=45200000000 \
  "$@"
