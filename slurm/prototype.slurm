#!/bin/bash
# =============================================================================
# 训练作业唯一 Slurm 模板（勿再为每个模型复制本文件）
# =============================================================================
# 通过包装器提交，并注入要跑的训练脚本：
#   bash slurm/sbatch-train.sh <name> [--name JOB_NAME] [额外 sbatch 参数...]
#
# 包装器会设置：
#   --job-name=<脚本 basename>
#   --export=ALL,BDELF_TRAIN_SCRIPT=<绝对路径>
# =============================================================================
#SBATCH --job-name=bdelf-train
#SBATCH --partition=cls1
#SBATCH --nodelist=cls1-srv[2-4]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/cls1-beegfs/home/csh/source/bdelf/slurm/logs/%x-%j.out
#SBATCH --error=/data/cls1-beegfs/home/csh/source/bdelf/slurm/logs/%x-%j.err

set -euo pipefail

PROJECT=/data/cls1-beegfs/home/csh/source/bdelf
cd "$PROJECT"
source .venv/bin/activate

if [[ -z "${BDELF_TRAIN_SCRIPT:-}" ]]; then
  echo "BDELF_TRAIN_SCRIPT 未设置；请用: bash slurm/sbatch-train.sh <name>" >&2
  exit 1
fi
if [[ ! -f "$BDELF_TRAIN_SCRIPT" ]]; then
  echo "训练脚本不存在: $BDELF_TRAIN_SCRIPT" >&2
  exit 1
fi

# Triton/Inductor 缓存在 BeeGFS 上会因 rename 竞态丢 .cubin；必须用节点本地盘。
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
echo "TRAIN_SCRIPT=$BDELF_TRAIN_SCRIPT"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# 后台 GPU 监控：每 30 秒记录一次显存/利用率到独立日志。
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

bash "$BDELF_TRAIN_SCRIPT"

echo "=== job end: $(date -Is) ==="
