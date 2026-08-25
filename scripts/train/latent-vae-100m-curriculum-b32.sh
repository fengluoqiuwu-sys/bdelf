#!/usr/bin/env bash
# latent_vae 100m 长度课程：B=32 块因果两组串跑（D=16 再 D=32）。
# 同一 Slurm 作业内两段（经 sbatch-train 提交；本脚本不直接占远端 GPU）。
# 两段各 10B，默认 --time=2-00:00:00 不够，提交须 long，例如：
#   bash slurm/sbatch-train.sh latent-vae-100m-curriculum-b32 --name vae-cur-b32 --qos=long --time=8-00:00:00
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

echo "=== latent_vae curriculum latent_dim=32 block_size=16 $(date -Is) ==="
bash "$ROOT/scripts/train/latent-vae-100m-curriculum.sh" \
  --set model.latent_dim=32 \
  --set model.block_size=16 \
  "$@"
echo "=== done latent_dim=32 block_size=16 $(date -Is) ==="

echo "=== latent_vae curriculum latent_dim=32 block_size=32 $(date -Is) ==="
bash "$ROOT/scripts/train/latent-vae-100m-curriculum.sh" \
  --set model.latent_dim=32 \
  --set model.block_size=32 \
  "$@"
echo "=== done latent_dim=32 block_size=32 $(date -Is) ==="

echo "=== latent-vae curriculum B=32 D=16/32 两段全部完成 $(date -Is) ==="
