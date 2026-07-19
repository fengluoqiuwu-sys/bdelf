#!/bin/bash
#SBATCH --job-name=ar2-100m-full
#SBATCH --partition=cls1
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

# Triton/Inductor 缓存在 BeeGFS 上会因 rename 竞态丢 .cubin；必须用节点本地盘。
# train.py 也会再按 local_rank 隔离并优先改写到本地。mpl/xdg 可留在项目目录。
LOCAL_COMPILE="${SLURM_TMPDIR:-/tmp}/bdelf-compile-${SLURM_JOB_ID}"
mkdir -p "$LOCAL_COMPILE"/{triton,inductor}
export TRITON_CACHE_DIR="$LOCAL_COMPILE/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_COMPILE/inductor"
export MPLCONFIGDIR="$PROJECT/cache/compile/mpl"
export XDG_CACHE_HOME="$PROJECT/cache/compile/xdg"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# 计算节点无外网，使用 BeeGFS 上已缓存的 HuggingFace 数据。
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$PROJECT/cache/huggingface"

# 避免同节点多作业 MASTER_PORT 冲突。
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== job start: $(date -Is) | host: $(hostname) | job: ${SLURM_JOB_ID} ==="
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# 后台 GPU 监控：每 30 秒记录一次显存/利用率到独立日志，不影响训练主进程。
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
# 训练脚本退出（正常结束/报错/被取消）时，确保关闭监控进程。
trap 'kill "$GPU_MON_PID" 2>/dev/null || true' EXIT

python train.py \
  --model ar2 \
  --config 100m-full \
  --dataset owt \
  --preprocess default

echo "=== job end: $(date -Is) ==="
