#!/bin/bash
# =============================================================================
# 训练作业唯一 Slurm 模板（勿再为每个模型复制本文件）
# =============================================================================
# 通过包装器提交：
#   bash slurm/sbatch-train.sh <name> [--name JOB_NAME] [额外 sbatch 参数...]
#
# 包装器写入 logs/ovan-server/pending/train-script-<job-name>.pending
# （第 1 行：训练脚本绝对路径；第 2 行：日志目录 logs/ovan-server/<时间戳>/），
# 本脚本按 SLURM_JOB_NAME 读取。不使用 sbatch --export（避免集群
# user env retrieval hold）。
#
# 三个日志文件（均在上述时间戳目录下）：
#   <job-name>-<job-id>.out / .err（sbatch --output/--error）
#   gpu-<job-id>.log（本脚本后台 nvidia-smi）
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
# 默认落点（包装器会用 --output/--error 覆盖到 logs/<server>/<ts>/）
#SBATCH --output=/data/cls1-beegfs/home/csh/source/bdelf/logs/ovan-server/_sbatch-default/%x-%j.out
#SBATCH --error=/data/cls1-beegfs/home/csh/source/bdelf/logs/ovan-server/_sbatch-default/%x-%j.err

set -euo pipefail

PROJECT=/data/cls1-beegfs/home/csh/source/bdelf
SERVER_NAME="${BDELF_SERVER_NAME:-ovan-server}"
cd "$PROJECT"
source .venv/bin/activate

PENDING="$PROJECT/logs/${SERVER_NAME}/pending/train-script-${SLURM_JOB_NAME}.pending"
if [[ ! -f "$PENDING" ]]; then
  # 兼容旧路径（迁移前已提交的作业）
  PENDING_LEGACY="$PROJECT/slurm/logs/train-script-${SLURM_JOB_NAME}.pending"
  if [[ -f "$PENDING_LEGACY" ]]; then
    PENDING="$PENDING_LEGACY"
  else
    echo "缺少 $PENDING；请用: bash slurm/sbatch-train.sh <name>" >&2
    exit 1
  fi
fi
BDELF_TRAIN_SCRIPT="$(sed -n '1p' "$PENDING")"
JOB_LOG_DIR="$(sed -n '2p' "$PENDING")"
if [[ -z "$BDELF_TRAIN_SCRIPT" || ! -f "$BDELF_TRAIN_SCRIPT" ]]; then
  echo "训练脚本无效: ${BDELF_TRAIN_SCRIPT:-<empty>}（来自 $PENDING）" >&2
  exit 1
fi
if [[ -z "$JOB_LOG_DIR" ]]; then
  # 旧 pending 仅一行：回退到 slurm/logs
  JOB_LOG_DIR="$PROJECT/slurm/logs"
fi
mkdir -p "$JOB_LOG_DIR"

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
echo "JOB_LOG_DIR=$JOB_LOG_DIR"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# 后台 GPU 监控：每 30 秒记录一次显存/利用率到独立日志。
GPU_LOG="$JOB_LOG_DIR/gpu-${SLURM_JOB_ID}.log"
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
