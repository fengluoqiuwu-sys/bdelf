#!/usr/bin/env bash
# latent_vae 100m 长度课程扫 bottleneck：block_size=1，latent_dim=16/32/64/128 顺序跑。
# 同一 Slurm 作业内四段（经 sbatch-train 提交；本脚本不直接占远端 GPU）。
# 四段各 10B，默认 --time=2-00:00:00 不够，提交须 long，例如：
#   bash slurm/sbatch-train.sh latent-vae-100m-curriculum-d1 --name vae-cur-d1 --qos=long --time=14-00:00:00
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

for b in 16 32 64 128; do
  echo "=== latent_vae curriculum latent_dim=${b} block_size=1 $(date -Is) ==="
  bash "$ROOT/scripts/train/latent-vae-100m-curriculum.sh" \
    --set model.latent_dim="$b" \
    --set model.block_size=1 \
    "$@"
  echo "=== done latent_dim=${b} block_size=1 $(date -Is) ==="
done
echo "=== latent-vae curriculum d=1 四段全部完成 $(date -Is) ==="
