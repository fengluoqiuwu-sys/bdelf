#!/usr/bin/env bash
# ODAR 100m full（DMA-H；配方对齐 ELF-100m full / elf-cfg 日程）。
# 经 slurm/sbatch-train.sh 提交或本地直接跑；默认 4 GPU（prototype.slurm）。
# 训练日程与 elf-cfg-100m-full 对齐（论文 Tab.7）：
#   - warmup_ratio=0.1
#   - min_lr_ratio=1.0 → warmup 后 constant LR=0.002
#   - target_tokens=45.2B
#   - gen_eval_samples=32（控在线评测开销）
# batch_size=16（alloc：4090 / global_bs=512 / ws=4；A6000 可 --set 32）
# 采样默认 FAR-SC：config/generate/odar → self_cond_cfg_scale=0.5
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
  "$@"
