#!/usr/bin/env bash
# latent_t5 100m 长度课程：单向两组串跑（readout=e 再 readout=b）。
# 同一 Slurm 作业内两段（经 sbatch-train 提交；本脚本不直接占远端 GPU）。
# 两段各 10B，默认 --time=2-00:00:00 不够，提交须 long，例如：
#   bash slurm/sbatch-train.sh latent-t5-100m-curriculum-uni --name t5-cur-uni --qos=long --time=8-00:00:00
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

echo "=== latent_t5 curriculum uni readout=e $(date -Is) ==="
bash "$ROOT/scripts/train/latent-t5-100m-curriculum.sh" \
  --set model.bidirectional=false \
  "$@"
echo "=== done uni readout=e $(date -Is) ==="

echo "=== latent_t5 curriculum uni readout=b $(date -Is) ==="
bash "$ROOT/scripts/train/latent-t5-100m-curriculum.sh" \
  --set model.bidirectional=false \
  --set model.readout=b \
  "$@"
echo "=== done uni readout=b $(date -Is) ==="

echo "=== latent-t5 curriculum uni 两段全部完成 $(date -Is) ==="
