#!/usr/bin/env bash
# 用 slurm/eval.slurm 拉起 scripts/eval/*.sh（默认 4 GPU，与训练 prototype 一致）
#
# 用法（仓库根；在 ovan-server 上执行）::
#   bash slurm/sbatch-eval.sh odar-sc-ace -- --run full/odar/<hash>
#   bash slurm/sbatch-eval.sh odar-sc-ace --name odar-eval-360k --exclude=cls1-srv2 -- \
#     --run full/odar/<hash> --checkpoint cache/checkpoints/full/odar/<hash>/checkpoint_step_0360000.pt
#
# 模板默认 4 GPU / 16 CPU / 128G mem（与 sbatch-train / prototype.slurm 对齐）。
# 若要更少卡：追加 --gpus-per-node=1 --mem=64G。
#
# 日志：logs/ovan-server/<时间戳>/{job-name}-{job-id}.{out,err} 与 gpu-{job-id}.log
# -- 之后参数写入 pending，由 eval.slurm 转给 scripts/eval/<name>.sh。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$ROOT/scripts/eval"
cd "$ROOT"
# shellcheck source=../scripts/job_log_dir.sh
source "$ROOT/scripts/job_log_dir.sh"

SERVER_NAME="${BDELF_SERVER_NAME:-ovan-server}"
PROJECT=/data/cls1-beegfs/home/csh/source/bdelf

usage() {
  cat <<'EOF' >&2
用法: bash slurm/sbatch-eval.sh [-n|--name JOB_NAME] <eval> [额外 sbatch 参数...] [--] [传给脚本的参数...]

  <eval>  默认在 scripts/eval/ 下解析：
          odar-sc-ace  |  odar-sc-ace.sh  |  scripts/eval/odar-sc-ace.sh
  -n, --name   Slurm --job-name（默认取脚本文件名去掉 .sh）
  --           之后参数原样传给 scripts/eval/<eval>.sh（如 --run full/odar/<hash>）

日志写入 logs/ovan-server/<时间戳>/（可用 BDELF_SERVER_NAME 覆盖服务名）。

示例:
  bash slurm/sbatch-eval.sh odar-sc-ace -- --run full/odar/5428154e7c817e11
  bash slurm/sbatch-eval.sh odar-sc-ace --gpus-per-node=1 --mem=64G -- \
    --run full/odar/5428154e7c817e11 --micro-bs 32
EOF
  exit 1
}

JOB_NAME=""
EVAL_ARG=""
SBATCH_ARGS=()
EVAL_EXTRA=()
SEEN_SEP=0

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
      SEEN_SEP=1
      shift
      EVAL_EXTRA+=("$@")
      break
      ;;
    -*)
      if [[ -z "$EVAL_ARG" ]]; then
        echo "未知选项（eval 脚本名须在 sbatch 参数之前）: $1" >&2
        usage
      fi
      if [[ $SEEN_SEP -eq 1 ]]; then
        EVAL_EXTRA+=("$1")
        shift
      else
        SBATCH_ARGS+=("$1")
        shift
      fi
      ;;
    *)
      if [[ -z "$EVAL_ARG" ]]; then
        EVAL_ARG=$1
        shift
      elif [[ $SEEN_SEP -eq 1 ]]; then
        EVAL_EXTRA+=("$1")
        shift
      else
        # 无 -- 时，未识别的位置参数也当作传给 eval 脚本（便于 --run ...）
        EVAL_EXTRA+=("$1")
        shift
      fi
      ;;
  esac
done

[[ -n "$EVAL_ARG" ]] || usage

resolve_eval_script() {
  local arg=$1
  local cand

  if [[ "$arg" = /* ]]; then
    cand=$arg
  elif [[ "$arg" == scripts/eval/* || "$arg" == ./scripts/eval/* ]]; then
    cand="$ROOT/${arg#./}"
  elif [[ "$arg" == */* ]]; then
    cand="$ROOT/$arg"
  else
    if [[ "$arg" == *.sh ]]; then
      cand="$EVAL_DIR/$arg"
    else
      cand="$EVAL_DIR/${arg}.sh"
    fi
  fi

  if [[ ! -f "$cand" ]]; then
    echo "eval 脚本不存在: $cand" >&2
    echo "可用脚本:" >&2
    ls -1 "$EVAL_DIR"/*.sh 2>/dev/null | xargs -n1 basename | sed 's/\.sh$//' >&2 || true
    exit 1
  fi

  cand="$(cd "$(dirname "$cand")" && pwd)/$(basename "$cand")"
  case "$cand" in
    "$EVAL_DIR"/*.sh) ;;
    *)
      echo "eval 脚本须位于 $EVAL_DIR/*.sh，收到: $cand" >&2
      exit 1
      ;;
  esac
  printf '%s\n' "$cand"
}

EVAL_SCRIPT="$(resolve_eval_script "$EVAL_ARG")"
if [[ -z "$JOB_NAME" ]]; then
  JOB_NAME="$(basename "$EVAL_SCRIPT" .sh)"
fi

EVAL_SCRIPT="$PROJECT/scripts/eval/$(basename "$EVAL_SCRIPT")"
if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "BeeGFS 上找不到 eval 脚本: $EVAL_SCRIPT" >&2
  exit 1
fi

job_log_alloc "$SERVER_NAME" "$PROJECT"
PENDING_DIR="$(job_log_pending_dir "$SERVER_NAME" "$PROJECT")"
PENDING="$PENDING_DIR/eval-script-${JOB_NAME}.pending"
ARGS_PENDING="$PENDING_DIR/eval-args-${JOB_NAME}.pending"
{
  printf '%s\n' "$EVAL_SCRIPT"
  printf '%s\n' "$JOB_LOG_DIR"
} > "$PENDING"
: > "$ARGS_PENDING"
if [[ ${#EVAL_EXTRA[@]} -gt 0 ]]; then
  printf '%s\n' "${EVAL_EXTRA[@]}" > "$ARGS_PENDING"
fi

SBATCH_OUT="$(sbatch \
  --job-name="$JOB_NAME" \
  --output="${JOB_LOG_DIR}/%x-%j.out" \
  --error="${JOB_LOG_DIR}/%x-%j.err" \
  "${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}" \
  "$ROOT/slurm/eval.slurm")"
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
  export BDELF_META_SCRIPT="$EVAL_SCRIPT"
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
    "kind": "eval",
    "log_dir": os.environ["BDELF_META_LOG_DIR"],
    "started_at": os.environ["BDELF_META_STARTED"],
}
Path(os.environ["BDELF_META_LOG_DIR"], "meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
  echo "log_dir=$JOB_LOG_DIR"
fi
