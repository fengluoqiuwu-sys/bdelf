#!/bin/bash
# =============================================================================
# 多模型 VRAM 探针 suite（1 卡；不写死 nodelist）
# 提交：bash slurm/sbatch-vram-probe.sh --suite [--nodelist=…]
# 本模板不依赖 sbatch --export（集群上 ALL/NONE 均曾触发 env retrieval hold）
# =============================================================================
#SBATCH --job-name=vram-probe-suite
#SBATCH --partition=cls1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/data/cls1-beegfs/home/csh/source/bdelf/logs/ovan-server/_sbatch-default/%x-%j.out
#SBATCH --error=/data/cls1-beegfs/home/csh/source/bdelf/logs/ovan-server/_sbatch-default/%x-%j.err

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

echo "=== vram-probe-suite start: $(date -Is) | host: $(hostname) | job: ${SLURM_JOB_ID} ==="
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

bash scripts/vram_probe_suite.sh

echo "=== vram-probe-suite end: $(date -Is) ==="
