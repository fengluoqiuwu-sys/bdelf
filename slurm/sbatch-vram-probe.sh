#!/usr/bin/env bash
# 提交单卡 VRAM 探针作业（模板 slurm/vram-probe.slurm）。
#
# 用法（仓库根；在 ovan-server 上执行）::
#   bash slurm/sbatch-vram-probe.sh --nodelist=cls1-srv3 -- \
#     --model elf --config 100m-full --dataset owt --preprocess elf --generate eval \
#     --batches 8,16,24,32 --world-size 4
#
# ``--`` 之前：sbatch 参数（--nodelist / --gpus-per-node / --name 等）。
# ``--`` 之后：传给 scripts/vram_probe.py 的参数。
# 未写 ``--`` 时，第一个探针旗标起视为探针 CLI。
#
# 日志：logs/<服务名>/<时间戳>/{job-name}-{job-id}.{out,err}
# 探针参数写入 logs/<服务名>/pending/vram-probe-args.pending。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=../scripts/job_log_dir.sh
source "$ROOT/scripts/job_log_dir.sh"
SCRIPT_DIR="$ROOT/scripts"
# shellcheck source=../scripts/servers_lib.sh
source "$SCRIPT_DIR/servers_lib.sh"

SERVER_NAME="${BDELF_SERVER_NAME:-ovan-server}"
PROJECT=/data/cls1-beegfs/home/csh/source/bdelf

usage() {
  cat <<'EOF' >&2
用法: bash slurm/sbatch-vram-probe.sh [sbatch 选项...] [--] <vram_probe.py 参数...>
   或: bash slurm/sbatch-vram-probe.sh [sbatch 选项...] --suite

  -n, --name JOB_NAME   Slurm --job-name（默认 vram-probe / vram-probe-suite）
  --suite               跑 scripts/vram_probe_suite.sh（多模型依次测，跳过 cola 系）
  --nodelist=...        指定节点（推荐；模板不写死）
  --gpus-per-node=N     默认模板为 1；可覆盖
  其余以 - 开头的参数原样传给 sbatch

  -- 之后（或首个探针旗标起）传给 scripts/vram_probe.py，例如：
     --model elf --config 100m-full --dataset owt --preprocess elf --generate eval
     --batches 8,16,24,32 --world-size 4

示例:
  bash slurm/sbatch-vram-probe.sh --nodelist=cls1-srv2 --suite
  bash slurm/sbatch-vram-probe.sh --nodelist=cls1-srv3 -- \
    --model elf --config 100m-full --dataset owt --preprocess elf --generate eval
EOF
  exit 1
}

JOB_NAME=""
SUITE=0
SBATCH_ARGS=()
PROBE_ARGS=()
SEEN_SEP=0

is_probe_flag() {
  case "$1" in
    --model|--config|--dataset|--preprocess|--generate|--batches|--world-size|--set)
      return 0
      ;;
    --model=*|--config=*|--dataset=*|--preprocess=*|--generate=*|--batches=*|--world-size=*|--set=*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

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
    --suite)
      SUITE=1
      shift
      ;;
    --)
      SEEN_SEP=1
      shift
      PROBE_ARGS+=("$@")
      break
      ;;
    *)
      if [[ $SEEN_SEP -eq 1 ]]; then
        PROBE_ARGS+=("$1")
        shift
      elif is_probe_flag "$1"; then
        PROBE_ARGS+=("$@")
        break
      else
        SBATCH_ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

if [[ $SUITE -eq 1 ]]; then
  if [[ ${#PROBE_ARGS[@]} -gt 0 ]]; then
    echo "--suite 与 vram_probe.py 参数互斥" >&2
    exit 1
  fi
  [[ -n "$JOB_NAME" ]] || JOB_NAME="vram-probe-suite"
else
  [[ -n "$JOB_NAME" ]] || JOB_NAME="vram-probe"
  if [[ ${#PROBE_ARGS[@]} -eq 0 ]]; then
    echo "缺少 vram_probe.py 参数（或改用 --suite）" >&2
    usage
  fi
fi

load_server "$SERVER_NAME" || exit 1
job_log_alloc "$SERVER_NAME" "$PROJECT"
PENDING_DIR="$(job_log_pending_dir "$SERVER_NAME" "$PROJECT")"

if [[ $SUITE -eq 1 ]]; then
  SBATCH_OUT="$(sbatch \
    --job-name="$JOB_NAME" \
    --output="${JOB_LOG_DIR}/%x-%j.out" \
    --error="${JOB_LOG_DIR}/%x-%j.err" \
    "${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}" \
    "$ROOT/slurm/vram-probe-suite.slurm")"
  printf '%s\n' "$SBATCH_OUT"
  echo "log_dir=$JOB_LOG_DIR"
  exit 0
fi

ARGS_FILE="$PENDING_DIR/vram-probe-args.pending"
printf '%s\n' "${PROBE_ARGS[@]}" > "$ARGS_FILE"

SBATCH_OUT="$(sbatch \
  --job-name="$JOB_NAME" \
  --output="${JOB_LOG_DIR}/%x-%j.out" \
  --error="${JOB_LOG_DIR}/%x-%j.err" \
  "${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}" \
  "$ROOT/slurm/vram-probe.slurm")"
printf '%s\n' "$SBATCH_OUT"
echo "log_dir=$JOB_LOG_DIR"
