#!/bin/bash
# =============================================================================
# 通用预处理作业（勿在文件名里写死 dataset/preprocess）
# 提交：bash slurm/sbatch-preprocess.sh --dataset owt --preprocess default
# 参数文件：slurm/logs/preprocess-args.pending（由包装器写入；本脚本消费后删除）
# =============================================================================
#SBATCH --job-name=preprocess
#SBATCH --partition=cls1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/cls1-beegfs/home/csh/source/bdelf/slurm/logs/%x-%j.out
#SBATCH --error=/data/cls1-beegfs/home/csh/source/bdelf/slurm/logs/%x-%j.err

set -euo pipefail

PROJECT=/data/cls1-beegfs/home/csh/source/bdelf
cd "$PROJECT"
source .venv/bin/activate

ARGS_FILE="$PROJECT/slurm/logs/preprocess-args.pending"
if [[ ! -f "$ARGS_FILE" ]]; then
  echo "缺少 $ARGS_FILE；请用: bash slurm/sbatch-preprocess.sh --dataset … --preprocess …" >&2
  exit 1
fi
mapfile -t PRE_ARGV < "$ARGS_FILE"
rm -f "$ARGS_FILE"

export TRITON_CACHE_DIR="$PROJECT/cache/compile/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/cache/compile/inductor"
export MPLCONFIGDIR="$PROJECT/cache/compile/mpl"
export XDG_CACHE_HOME="$PROJECT/cache/compile/xdg"

# 计算节点无外网；原始 parquet 与 tokenizer 需已缓存在 BeeGFS。
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$PROJECT/cache/huggingface"

echo "=== preprocess start: $(date -Is) | host: $(hostname) | job: ${SLURM_JOB_ID} ==="
echo "cpus=$(nproc) mem=$(grep MemTotal /proc/meminfo)"
echo "ARGS: ${PRE_ARGV[*]}"

# 必须用磁盘上的 .py：预处理内部 multiprocessing spawn 无法从 <stdin> 启动子进程。
.venv/bin/python scripts/preprocess.py "${PRE_ARGV[@]}"

echo "=== disk usage ==="
du -sh cache/preprocessed_datasets/* 2>/dev/null || true
echo "=== preprocess end: $(date -Is) ==="
