#!/usr/bin/env bash
# 用唯一模板 slurm/prototype.slurm 拉起 scripts/train/*.sh
#
# 用法（仓库根；在 ovan-server 上执行）::
#   bash slurm/sbatch-train.sh ar-100m-full
#   bash slurm/sbatch-train.sh elf-100m-full --name elf-cfg-100m-full
#   bash slurm/sbatch-train.sh elf-100m-full --exclude=cls1-srv2
#
# 日志：logs/ovan-server/<时间戳>/{job-name}-{job-id}.{out,err} 与 gpu-{job-id}.log
# 训练脚本默认在 scripts/train/ 下查找（可省略路径与 .sh）。
# --name / -n 指定 Slurm job-name；其余参数原样传给 sbatch。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$ROOT/scripts/train"
cd "$ROOT"
# shellcheck source=../scripts/job_log_dir.sh
source "$ROOT/scripts/job_log_dir.sh"

SERVER_NAME="${BDELF_SERVER_NAME:-ovan-server}"
# 计算节点上 /home/csh 常不可见；与 prototype.slurm 统一为 BeeGFS 绝对路径。
PROJECT=/data/cls1-beegfs/home/csh/source/bdelf

usage() {
  cat <<'EOF' >&2
用法: bash slurm/sbatch-train.sh [-n|--name JOB_NAME] <train> [额外 sbatch 参数...]

  <train>  默认在 scripts/train/ 下解析：
           ar-100m-full  |  ar-100m-full.sh  |  scripts/train/ar-100m-full.sh
  -n, --name   Slurm --job-name（默认取脚本文件名去掉 .sh）

日志写入 logs/ovan-server/<时间戳>/（可用 BDELF_SERVER_NAME 覆盖服务名）。

示例:
  bash slurm/sbatch-train.sh ar-100m-full
  bash slurm/sbatch-train.sh elf-100m-full --name elf-cfg-100m-full
  bash slurm/sbatch-train.sh elf-100m-full --exclude=cls1-srv2
EOF
  exit 1
}

JOB_NAME=""
TRAIN_ARG=""
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
    --)
      shift
      SBATCH_ARGS+=("$@")
      break
      ;;
    -*)
      # 其余以 - 开头的交给 sbatch（须已出现过 <train>）
      if [[ -z "$TRAIN_ARG" ]]; then
        echo "未知选项（训练脚本名须在 sbatch 参数之前）: $1" >&2
        usage
      fi
      SBATCH_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ -z "$TRAIN_ARG" ]]; then
        TRAIN_ARG=$1
        shift
      else
        SBATCH_ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

[[ -n "$TRAIN_ARG" ]] || usage

resolve_train_script() {
  local arg=$1
  local cand

  if [[ "$arg" = /* ]]; then
    cand=$arg
  elif [[ "$arg" == scripts/train/* || "$arg" == ./scripts/train/* ]]; then
    cand="$ROOT/${arg#./}"
  elif [[ "$arg" == */* ]]; then
    # 显式相对路径
    cand="$ROOT/$arg"
  else
    # 短名：scripts/train/<name>[.sh]
    if [[ "$arg" == *.sh ]]; then
      cand="$TRAIN_DIR/$arg"
    else
      cand="$TRAIN_DIR/${arg}.sh"
    fi
  fi

  if [[ ! -f "$cand" ]]; then
    echo "训练脚本不存在: $cand" >&2
    echo "可用脚本:" >&2
    ls -1 "$TRAIN_DIR"/*.sh 2>/dev/null | xargs -n1 basename | sed 's/\.sh$//' >&2 || true
    exit 1
  fi

  cand="$(cd "$(dirname "$cand")" && pwd)/$(basename "$cand")"
  case "$cand" in
    "$TRAIN_DIR"/*.sh) ;;
    *)
      echo "训练脚本须位于 $TRAIN_DIR/*.sh，收到: $cand" >&2
      exit 1
      ;;
  esac
  printf '%s\n' "$cand"
}

TRAIN_SCRIPT="$(resolve_train_script "$TRAIN_ARG")"
if [[ -z "$JOB_NAME" ]]; then
  JOB_NAME="$(basename "$TRAIN_SCRIPT" .sh)"
fi

TRAIN_SCRIPT="$PROJECT/scripts/train/$(basename "$TRAIN_SCRIPT")"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "BeeGFS 上找不到训练脚本: $TRAIN_SCRIPT" >&2
  exit 1
fi

job_log_alloc "$SERVER_NAME" "$PROJECT"
PENDING_DIR="$(job_log_pending_dir "$SERVER_NAME" "$PROJECT")"
PENDING="$PENDING_DIR/train-script-${JOB_NAME}.pending"
# 不用 sbatch --export：集群上 ALL/NONE 均曾触发
# 「user env retrieval failed requeued held」。改写 pending 文件，由
# prototype.slurm 按 SLURM_JOB_NAME 读取（与 vram-probe 同策略）。
{
  printf '%s\n' "$TRAIN_SCRIPT"
  printf '%s\n' "$JOB_LOG_DIR"
} > "$PENDING"

SBATCH_OUT="$(sbatch \
  --job-name="$JOB_NAME" \
  --output="${JOB_LOG_DIR}/%x-%j.out" \
  --error="${JOB_LOG_DIR}/%x-%j.err" \
  "${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}" \
  "$ROOT/slurm/prototype.slurm")"
printf '%s\n' "$SBATCH_OUT"

JOB_ID="$(printf '%s\n' "$SBATCH_OUT" | awk '/Submitted batch job/ {print $4; exit}')"
if [[ -n "$JOB_ID" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
  else
    PY=python3
  fi
  export BDELF_META_SERVER="$SERVER_NAME"
  export BDELF_META_JOB_ID="$JOB_ID"
  export BDELF_META_JOB_NAME="$JOB_NAME"
  export BDELF_META_SCRIPT="$TRAIN_SCRIPT"
  export BDELF_META_LOG_DIR="$JOB_LOG_DIR"
  export BDELF_META_STARTED="$(date -Is)"
  "$PY" <<'PY'
import json, os
from pathlib import Path
meta = {
    "server": os.environ["BDELF_META_SERVER"],
    "scheduler": "slurm",
    "job_id": os.environ["BDELF_META_JOB_ID"],
    "job_name": os.environ["BDELF_META_JOB_NAME"],
    "script": os.environ["BDELF_META_SCRIPT"],
    "log_dir": os.environ["BDELF_META_LOG_DIR"],
    "started_at": os.environ["BDELF_META_STARTED"],
}
Path(os.environ["BDELF_META_LOG_DIR"], "meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
  echo "log_dir=$JOB_LOG_DIR"
fi
