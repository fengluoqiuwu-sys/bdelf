#!/usr/bin/env bash
# latent_t5 100m 长度课程：双向默认 readout=e 与原版 T5（readout=none）串跑。
# 同一 Slurm 作业内两段（经 sbatch-train 提交；本脚本不直接占远端 GPU）。
# 两段各 10B，默认 --time=2-00:00:00 不够，提交须 long，例如：
#   bash slurm/sbatch-train.sh latent-t5-100m-curriculum-bi --name t5-cur-bi --qos=long --time=8-00:00:00
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

echo "=== latent_t5 curriculum bi readout=e $(date -Is) ==="
bash "$ROOT/scripts/train/latent-t5-100m-curriculum.sh" \
  "$@"
echo "=== done bi readout=e $(date -Is) ==="

echo "=== latent_t5 curriculum 原 T5 readout=none $(date -Is) ==="
bash "$ROOT/scripts/train/latent-t5-100m-curriculum.sh" \
  --set model.readout=none \
  "$@"
echo "=== done 原 T5 readout=none $(date -Is) ==="

echo "=== latent-t5 curriculum bi+none 两段全部完成 $(date -Is) ==="
