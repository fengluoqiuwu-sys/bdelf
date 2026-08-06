#!/usr/bin/env bash
# 提交预处理作业（模板 slurm/preprocess.slurm；入口 scripts/preprocess.py）。
#
# 用法（仓库根；在 ovan-server 上执行）::
#   bash slurm/sbatch-preprocess.sh --dataset owt --preprocess default
#   bash slurm/sbatch-preprocess.sh --dataset owt --preprocess elf --exclude=cls1-srv2
#
# ``--`` 之前也可夹杂其它 sbatch 选项；dataset/preprocess 均须显式给出。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF' >&2
用法: bash slurm/sbatch-preprocess.sh --dataset NAME --preprocess NAME [sbatch 选项...]

  --dataset NAME      数据集（config/datasets/）；必填
  --preprocess NAME   预处理配置（config/preprocess/）；必填
  -n, --name JOB_NAME Slurm --job-name（默认 <dataset>-<preprocess>-preprocess）
  其余以 - 开头的参数原样传给 sbatch

示例:
  bash slurm/sbatch-preprocess.sh --dataset owt --preprocess default
  bash slurm/sbatch-preprocess.sh --dataset owt --preprocess elf --nodelist=cls1-srv2
EOF
  exit 1
}

JOB_NAME=""
DATASET=""
PREPROCESS=""
SBATCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      ;;
    -n|--name)
      [[ $# -ge 2 ]] || { echo "--name 需要参数" >&2; exit 1; }
      JOB_NAME=$2
      shift 2
      ;;
    --name=*)
      JOB_NAME=${1#--name=}
      shift
      ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "--dataset 需要参数" >&2; exit 1; }
      DATASET=$2
      shift 2
      ;;
    --dataset=*)
      DATASET=${1#--dataset=}
      shift
      ;;
    --preprocess)
      [[ $# -ge 2 ]] || { echo "--preprocess 需要参数" >&2; exit 1; }
      PREPROCESS=$2
      shift 2
      ;;
    --preprocess=*)
      PREPROCESS=${1#--preprocess=}
      shift
      ;;
    --)
      shift
      SBATCH_ARGS+=("$@")
      break
      ;;
    -*)
      SBATCH_ARGS+=("$1")
      shift
      ;;
    *)
      echo "未知位置参数: $1（请用 --dataset / --preprocess）" >&2
      usage
      ;;
  esac
done

[[ -n "$DATASET" ]] || { echo "缺少 --dataset" >&2; usage; }
[[ -n "$PREPROCESS" ]] || { echo "缺少 --preprocess" >&2; usage; }
[[ -n "$JOB_NAME" ]] || JOB_NAME="${DATASET}-${PREPROCESS}-preprocess"

mkdir -p "$ROOT/slurm/logs"

ARGS_FILE="$ROOT/slurm/logs/preprocess-args.pending"
printf '%s\n' --dataset "$DATASET" --preprocess "$PREPROCESS" > "$ARGS_FILE"

exec sbatch \
  --job-name="$JOB_NAME" \
  "${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}" \
  "$ROOT/slurm/preprocess.slurm"
