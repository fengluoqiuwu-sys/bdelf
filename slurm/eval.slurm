#!/bin/bash
# =============================================================================
# TriFluency 离线 eval 作业 Slurm 模板（默认资源与 prototype.slurm 训练一致）
# =============================================================================
# 通过包装器提交：
#   bash slurm/sbatch-eval.sh <name> [--name JOB_NAME] [额外 sbatch 参数...] [--] [传给脚本的参数...]
#
# 包装器写入：
#   logs/ovan-server/pending/eval-script-<job-name>.pending
#     第 1 行：scripts/eval/*.sh 绝对路径
#     第 2 行：日志目录 logs/ovan-server/<时间戳>/
#   logs/ovan-server/pending/eval-args-<job-name>.pending
#     每行一个 argv（可空；消费后删除）
#
# 三个日志文件（均在上述时间戳目录下）：
#   <job-name>-<job-id>.out / .err（sbatch --output/--error）
#   gpu-<job-id>.log（本脚本后台 nvidia-smi）
# =============================================================================
#SBATCH --job-name=bdelf-eval
#SBATCH --partition=cls1
#SBATCH --nodelist=cls1-srv[2-4]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/cls1-beegfs/home/csh/source/bdelf/logs/ovan-server/_sbatch-default/%x-%j.out
#SBATCH --error=/data/cls1-beegfs/home/csh/source/bdelf/logs/ovan-server/_sbatch-default/%x-%j.err

set -euo pipefail

PROJECT=/data/cls1-beegfs/home/csh/source/bdelf
SERVER_NAME="${BDELF_SERVER_NAME:-ovan-server}"
cd "$PROJECT"
source .venv/bin/activate

PENDING="$PROJECT/logs/${SERVER_NAME}/pending/eval-script-${SLURM_JOB_NAME}.pending"
ARGS_FILE="$PROJECT/logs/${SERVER_NAME}/pending/eval-args-${SLURM_JOB_NAME}.pending"
if [[ ! -f "$PENDING" ]]; then
  echo "缺少 $PENDING；请用: bash slurm/sbatch-eval.sh <name>" >&2
  exit 1
fi
BDELF_EVAL_SCRIPT="$(sed -n '1p' "$PENDING")"
JOB_LOG_DIR="$(sed -n '2p' "$PENDING")"
if [[ -z "$BDELF_EVAL_SCRIPT" || ! -f "$BDELF_EVAL_SCRIPT" ]]; then
  echo "eval 脚本无效: ${BDELF_EVAL_SCRIPT:-<empty>}（来自 $PENDING）" >&2
  exit 1
fi
if [[ -z "$JOB_LOG_DIR" ]]; then
  JOB_LOG_DIR="$PROJECT/logs/${SERVER_NAME}"
fi
mkdir -p "$JOB_LOG_DIR"

EVAL_ARGV=()
if [[ -f "$ARGS_FILE" ]]; then
  mapfile -t EVAL_ARGV < "$ARGS_FILE"
  rm -f "$ARGS_FILE"
  # 去掉可能的末尾空行带来的空元素
  if [[ ${#EVAL_ARGV[@]} -gt 0 && -z "${EVAL_ARGV[-1]}" ]]; then
    unset 'EVAL_ARGV[-1]'
  fi
fi

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

echo "=== job start: $(date -Is) | host: $(hostname) | job: ${SLURM_JOB_ID} ==="
echo "EVAL_SCRIPT=$BDELF_EVAL_SCRIPT"
echo "JOB_LOG_DIR=$JOB_LOG_DIR"
echo "EVAL_ARGV=${EVAL_ARGV[*]:-<none>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

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
trap 'kill "$GPU_MON_PID" 2>/dev/null || true; rm -rf -- "${SLURM_TMPDIR:-/tmp}/bdelf-compile-${SLURM_JOB_ID}" "${SLURM_TMPDIR:-/tmp}/bdelf-resume-${SLURM_JOB_ID}"' EXIT

if [[ ${#EVAL_ARGV[@]} -gt 0 ]]; then
  bash "$BDELF_EVAL_SCRIPT" "${EVAL_ARGV[@]}"
else
  bash "$BDELF_EVAL_SCRIPT"
fi

echo "=== job end: $(date -Is) ==="
