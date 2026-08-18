#!/usr/bin/env bash
# Cola Stage-2 100m full（含训练期 gen-eval）；经 slurm/sbatch-train.sh 提交或本地直接跑。
# 需已有 Stage-1 VAE checkpoint：COLA_VAE_CHECKPOINT 或 --set 指向 full/cola_vae/<hash>。
# 两阶段连跑用 scripts/train/cola-seq-100m-full.sh。
# 训练日程与 elf-cfg-100m-full 对齐（论文 Tab.7）；Stage-1 VAE 仍用默认 50B 日程，勿把下列 --set 传给 cola_vae。
#   - warmup_ratio=0.1
#   - min_lr_ratio=1.0 → warmup 后 constant LR=0.002
#   - target_tokens=45.2B（5 × ~9.04B OWT）
#   - gen_eval_samples=32（控在线评测开销）
# 预处理仍为 default（GPT-2），不用 elf（T5）。
# fingerprint：resolve_checkpoint.py … 下列 --set → full/cola/d349aca8ed249f46
# Rank0 gen-eval 时 peer 会卡在短 all_reduce，拉长 NCCL 超时避免误杀。
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
  --model cola \
  --config 100m-full \
  --dataset owt \
  --preprocess default \
  --generate eval \
  --set eval.gen_eval_samples=32 \
  --set schedule.warmup_ratio=0.1 \
  --set schedule.min_lr_ratio=1.0 \
  --set schedule.target_tokens=45200000000 \
  "$@"
