#!/usr/bin/env bash
# LoopSC 100m full（ELF 复制：训练展开 2 步闭环 SC，主损失仍对 x0）。
# 日程与 elf-cfg-100m-full 对齐：warmup_ratio=0.1、min_lr_ratio=1.0（constant LR）、
# target_tokens=45.2B、在线 gen-eval 32 samples；微批用 full.yaml 的 16。
# 预处理仍用 elf（共享 T5 嵌入管线）。经 sbatch-train / launch-train；勿直接占远端 GPU。
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
  --model loopsc \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  --set eval.gen_eval_samples=32 \
  --set schedule.warmup_ratio=0.1 \
  --set schedule.min_lr_ratio=1.0 \
  --set schedule.target_tokens=45200000000 \
  "$@"
