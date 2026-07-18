#!/bin/bash
#SBATCH --job-name=owt-preprocess-elf
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

export TRITON_CACHE_DIR="$PROJECT/cache/compile/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/cache/compile/inductor"
export MPLCONFIGDIR="$PROJECT/cache/compile/mpl"
export XDG_CACHE_HOME="$PROJECT/cache/compile/xdg"

# 计算节点无外网；原始 parquet 与 t5-small tokenizer 需已缓存在 BeeGFS。
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$PROJECT/cache/huggingface"

echo "=== preprocess start: $(date -Is) | host: $(hostname) | job: ${SLURM_JOB_ID} ==="
echo "cpus=$(nproc) mem=$(grep MemTotal /proc/meminfo)"

# 必须用磁盘上的 .py 文件：预处理内部 multiprocessing spawn 无法从 <stdin> 启动子进程。
# 复用通用入口；ELF 使用 T5-small 分词（config/preprocess/elf.yaml）。
python slurm/preprocess-owt-default.py --dataset owt --preprocess elf

echo "=== disk usage ==="
du -sh cache/preprocessed_datasets/owt_elf_* 2>/dev/null || true
echo "=== preprocess end: $(date -Is) ==="
