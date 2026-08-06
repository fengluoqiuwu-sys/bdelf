#!/bin/bash
# =============================================================================
# 单卡单模型 VRAM 探针（勿写死 nodelist）
# 提交：bash slurm/sbatch-vram-probe.sh --nodelist=… -- --model …
# 参数文件：logs/ovan-server/pending/vram-probe-args.pending（包装器写入；本脚本消费后删除）
# 不使用 sbatch --export（集群上曾触发 user env retrieval hold）
# =============================================================================
#SBATCH --job-name=vram-probe
#SBATCH --partition=cls1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=/data/cls1-beegfs/home/csh/source/bdelf/logs/ovan-server/_sbatch-default/%x-%j.out
#SBATCH --error=/data/cls1-beegfs/home/csh/source/bdelf/logs/ovan-server/_sbatch-default/%x-%j.err

set -euo pipefail

PROJECT=/data/cls1-beegfs/home/csh/source/bdelf
SERVER_NAME="${BDELF_SERVER_NAME:-ovan-server}"
cd "$PROJECT"
source .venv/bin/activate

ARGS_FILE="$PROJECT/logs/${SERVER_NAME}/pending/vram-probe-args.pending"
if [[ ! -f "$ARGS_FILE" ]]; then
  ARGS_LEGACY="$PROJECT/slurm/logs/vram-probe-args.pending"
  if [[ -f "$ARGS_LEGACY" ]]; then
    ARGS_FILE="$ARGS_LEGACY"
  else
    echo "缺少 $ARGS_FILE；请用: bash slurm/sbatch-vram-probe.sh -- <vram_probe.py 参数>" >&2
    exit 1
  fi
fi
mapfile -t PROBE_ARGV < "$ARGS_FILE"
rm -f "$ARGS_FILE"

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

echo "=== vram-probe start: $(date -Is) | host: $(hostname) | job: ${SLURM_JOB_ID} ==="
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "ARGS: ${PROBE_ARGV[*]}"

.venv/bin/python scripts/vram_probe.py "${PROBE_ARGV[@]}"

echo "=== vram-probe end: $(date -Is) ==="
