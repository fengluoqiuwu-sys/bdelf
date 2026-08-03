#!/bin/bash
#SBATCH --job-name=cola-vae-100m-full
#SBATCH --partition=cls1
#SBATCH --nodelist=cls1-srv[2-4]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/cls1-beegfs/home/csh/source/bdelf/slurm/logs/%x-%j.out
#SBATCH --error=/data/cls1-beegfs/home/csh/source/bdelf/slurm/logs/%x-%j.err

set -euo pipefail

PROJECT=/data/cls1-beegfs/home/csh/source/bdelf
cd "$PROJECT"
source .venv/bin/activate

LOCAL_COMPILE="${SLURM_TMPDIR:-/tmp}/bdelf-compile-${SLURM_JOB_ID}"
mkdir -p "$LOCAL_COMPILE"/{triton,inductor}
export TRITON_CACHE_DIR="$LOCAL_COMPILE/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_COMPILE/inductor"
export MPLCONFIGDIR="$PROJECT/cache/compile/mpl"
export XDG_CACHE_HOME="$PROJECT/cache/compile/xdg"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$PROJECT/cache/huggingface"

export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== job start: $(date -Is) | host: $(hostname) | job: ${SLURM_JOB_ID} ==="
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

GPU_LOG="$PROJECT/slurm/logs/gpu-${SLURM_JOB_ID}.log"
(
  echo "timestamp, index, memory.used [MiB], memory.total [MiB], utilization.gpu [%], power.draw [W], temperature.gpu"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader
    sleep 30
  done
) > "$GPU_LOG" 2>&1 &
GPU_MON_PID=$!
trap 'kill "$GPU_MON_PID" 2>/dev/null || true' EXIT

python train.py \
  --model cola_vae \
  --config 100m-full \
  --dataset owt \
  --preprocess default \
  --run-name cola_vae-100m-full-muon

echo "=== job end: $(date -Is) ==="
