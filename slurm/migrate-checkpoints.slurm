#!/bin/bash
#SBATCH --job-name=migrate-ckpts
#SBATCH --partition=cls1
#SBATCH --nodelist=cls1-srv4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/cls1-beegfs/home/csh/source/bdelf/slurm/logs/%x-%j.out
#SBATCH --error=/data/cls1-beegfs/home/csh/source/bdelf/slurm/logs/%x-%j.err

# 一次性：旧扁平 checkpoint → {full|fast}/{model}/{hash}/
# CPU + BeeGFS IO；不申请 GPU（与 preprocess 同类）；固定 srv4

set -euo pipefail

PROJECT=/data/cls1-beegfs/home/csh/source/bdelf
cd "$PROJECT"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=""

echo "=== migrate start: $(date -Is) | host: $(hostname) | job: ${SLURM_JOB_ID} ==="
echo "checkpoints before:"
ls -la cache/checkpoints || true
find cache/checkpoints -maxdepth 3 -type d | sort || true

python migrate_checkpoints.py --root "$PROJECT/cache/checkpoints"

echo "checkpoints after:"
find cache/checkpoints -maxdepth 4 \( -name config.json -o -name hardware.json -o -name launch_hint.json \) | sort
ls -la cache/checkpoints || true
echo "=== migrate end: $(date -Is) ==="
