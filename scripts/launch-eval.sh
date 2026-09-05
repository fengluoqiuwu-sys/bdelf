#!/usr/bin/env bash
# common 远端 eval 拉起（类比 scripts/launch-train.sh；勿直接 bash scripts/eval/*.sh 占 GPU）
#
# 用法（在 common 远端仓库根执行）::
#   bash scripts/launch-eval.sh <eval> --server <服务名> --gpus 0,1,2,3 \
#     [--name JOB_NAME] [--holder WHO] [--] [传给脚本的参数...]
#
# 默认卡数与训练一致：请传 4 张（例 --gpus 0,1,2,3）；张数须为 1/2/4/8。
# eval.py 当前主要吃单卡显存；多卡仅作额度/互斥与训练同构。
#
# 日志（与 Slurm 同构）::
#   logs/<服务名>/<时间戳>/<job-name>-<pid>.out|.err
#   logs/<服务名>/<时间戳>/gpu-<pid>.log
# 并写 temp/agent/active|launched/pid<PID>.json（scheduler=common）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$ROOT/scripts/eval"
cd "$ROOT"
# shellcheck source=job_log_dir.sh
source "$ROOT/scripts/job_log_dir.sh"
SCRIPT_DIR="$ROOT/scripts"
# shellcheck source=servers_lib.sh
source "$SCRIPT_DIR/servers_lib.sh"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY=python3
fi

usage() {
  cat <<'EOF' >&2
用法: bash scripts/launch-eval.sh <eval> --server NAME --gpus IDS [选项...] [--] [参数...]

  <eval>          scripts/eval/ 下脚本短名或路径（同 sbatch-eval.sh）
  --server NAME   csv 服务名（冒号左侧；无冒号则整列）。日志目录 logs/<服务名>/<时间戳>/
  --gpus IDS      物理卡号，逗号分隔（例 0,1,2,3）；必填；默认与训练同为 4 卡
  -n, --name NAME 作业名（默认取脚本文件名）
  --holder WHO    agent 登记 holder（默认 manual）
  --              之后参数原样传给 eval 脚本（如 --run full/odar/<hash>）

示例:
  bash scripts/launch-eval.sh odar-sc-ace --server train-server-1 --gpus 0,1,2,3 -- \
    --run full/odar/5428154e7c817e11
EOF
  exit 1
}

JOB_NAME=""
EVAL_ARG=""
SERVER_NAME=""
GPUS=""
HOLDER="manual"
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
    --server)
      [[ $# -ge 2 ]] || { echo "--server 需要参数" >&2; exit 1; }
      SERVER_NAME=$2
      shift 2
      ;;
    --server=*)
      SERVER_NAME=${1#--server=}
      shift
      ;;
    --gpus)
      [[ $# -ge 2 ]] || { echo "--gpus 需要参数" >&2; exit 1; }
      GPUS=$2
      shift 2
      ;;
    --gpus=*)
      GPUS=${1#--gpus=}
      shift
      ;;
    --holder)
      [[ $# -ge 2 ]] || { echo "--holder 需要参数" >&2; exit 1; }
      HOLDER=$2
      shift 2
      ;;
    --holder=*)
      HOLDER=${1#--holder=}
      shift
      ;;
    --)
      SEEN_SEP=1
      shift
      EVAL_EXTRA+=("$@")
      break
      ;;
    -*)
      if [[ $SEEN_SEP -eq 1 || -n "$EVAL_ARG" ]]; then
        EVAL_EXTRA+=("$1")
        shift
      else
        echo "未知选项: $1" >&2
        usage
      fi
      ;;
    *)
      if [[ -z "$EVAL_ARG" ]]; then
        EVAL_ARG=$1
        shift
      else
        EVAL_EXTRA+=("$1")
        shift
      fi
      ;;
  esac
done

[[ -n "$EVAL_ARG" ]] || usage
[[ -n "$SERVER_NAME" ]] || { echo "缺少 --server" >&2; usage; }
[[ -n "$GPUS" ]] || { echo "缺少 --gpus（建议与训练同为 4 卡，如 0,1,2,3）" >&2; usage; }
load_server "$SERVER_NAME" || exit 1

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

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
  echo "--gpus 不能为空" >&2
  exit 1
fi
declare -A SEEN_GPU=()
GPU_IDS=()
for g in "${GPU_ARR[@]}"; do
  g="${g// /}"
  [[ "$g" =~ ^[0-9]+$ ]] || { echo "--gpus 格式无效: $GPUS" >&2; exit 1; }
  [[ -z "${SEEN_GPU[$g]+x}" ]] || { echo "--gpus 有重复卡号: $GPUS" >&2; exit 1; }
  SEEN_GPU[$g]=1
  GPU_IDS+=("$g")
done
NGPUS=${#GPU_IDS[@]}
case "$NGPUS" in
  1|2|4|8) ;;
  *)
    echo "--gpus 张数须为 1/2/4/8，收到 $NGPUS" >&2
    exit 1
    ;;
esac

job_log_alloc "$SERVER_NAME" "$ROOT"
STARTED_AT="$(date -Is)"

ARGS_FILE="$JOB_LOG_DIR/eval.args"
: > "$ARGS_FILE"
if [[ ${#EVAL_EXTRA[@]} -gt 0 ]]; then
  printf '%s\0' "${EVAL_EXTRA[@]}" >> "$ARGS_FILE"
fi

RUNNER="$JOB_LOG_DIR/runner.sh"
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$ROOT")
JOB_ID=\$\$
JOB_NAME=$(printf '%q' "$JOB_NAME")
LOG_DIR=$(printf '%q' "$JOB_LOG_DIR")
EVAL_SCRIPT=$(printf '%q' "$EVAL_SCRIPT")
ARGS_FILE=$(printf '%q' "$ARGS_FILE")
GPUS=$(printf '%q' "$GPUS")
OUT="\$LOG_DIR/\${JOB_NAME}-\${JOB_ID}.out"
ERR="\$LOG_DIR/\${JOB_NAME}-\${JOB_ID}.err"
GPU_LOG="\$LOG_DIR/gpu-\${JOB_ID}.log"
# shellcheck source=job_log_dir.sh
source $(printf '%q' "$ROOT/scripts/job_log_dir.sh")
bdelf_export_scratch_tmpdir
export BDELF_JOB_ID="\$JOB_ID"
export CUDA_VISIBLE_DEVICES="\$GPUS"
exec >>"\$OUT" 2>>"\$ERR"
echo "=== job start: \$(date -Is) | host: \$(hostname) | pid: \$JOB_ID ==="
echo "EVAL_SCRIPT=\$EVAL_SCRIPT"
echo "JOB_LOG_DIR=\$LOG_DIR"
echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
echo "TMPDIR=\${TMPDIR:-/tmp} BDELF_JOB_ID=\$BDELF_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
(
  echo "timestamp, index, memory.used [MiB], memory.total [MiB], utilization.gpu [%], power.draw [W], temperature.gpu"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \\
      --format=csv,noheader || true
    sleep 30
  done
) > "\$GPU_LOG" 2>&1 &
GPU_MON_PID=\$!
trap 'kill "\$GPU_MON_PID" 2>/dev/null || true; bdelf_rm_job_scratch "\$JOB_ID"' EXIT
mapfile -d '' -t EVAL_ARGV < "\$ARGS_FILE"
if [[ \${#EVAL_ARGV[@]} -gt 0 && -z "\${EVAL_ARGV[-1]}" ]]; then
  unset 'EVAL_ARGV[-1]'
fi
if [[ \${#EVAL_ARGV[@]} -gt 0 ]]; then
  bash "\$EVAL_SCRIPT" "\${EVAL_ARGV[@]}"
else
  bash "\$EVAL_SCRIPT"
fi
echo "=== job end: \$(date -Is) ==="
EOF
chmod +x "$RUNNER"

nohup "$RUNNER" >/dev/null 2>&1 &
PID=$!
sleep 0.3
if ! kill -0 "$PID" 2>/dev/null; then
  echo "拉起失败：进程 $PID 已退出；见 $JOB_LOG_DIR" >&2
  if compgen -G "$JOB_LOG_DIR/${JOB_NAME}-*.out" > /dev/null; then
    echo "---- out (tail) ----" >&2
    tail -n 40 "$JOB_LOG_DIR"/${JOB_NAME}-*.out >&2 || true
  fi
  exit 1
fi

JOB_KEY="pid${PID}"
REL_LOG_DIR="logs/${SERVER_NAME}/${JOB_LOG_TS}"
SCRIPT_REL="scripts/eval/$(basename "$EVAL_SCRIPT")"
CMDLINE="bash scripts/launch-eval.sh $(basename "$EVAL_SCRIPT" .sh) --server ${SERVER_NAME} --gpus ${GPUS} --name ${JOB_NAME} --holder ${HOLDER}"

mkdir -p "$ROOT/temp/agent/active" "$ROOT/temp/agent/launched"

export BDELF_META_SERVER="$SERVER_NAME"
export BDELF_META_JOB_KEY="$JOB_KEY"
export BDELF_META_PID="$PID"
export BDELF_META_JOB_NAME="$JOB_NAME"
export BDELF_META_SCRIPT="$SCRIPT_REL"
export BDELF_META_CMDLINE="$CMDLINE"
export BDELF_META_NGPUS="$NGPUS"
export BDELF_META_GPU_IDS="$GPUS"
export BDELF_META_LOG_DIR="$REL_LOG_DIR"
export BDELF_META_ABS_LOG_DIR="$JOB_LOG_DIR"
export BDELF_META_STARTED="$STARTED_AT"
export BDELF_META_HOLDER="$HOLDER"
export BDELF_META_ROOT="$ROOT"

"$PY" <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BDELF_META_ROOT"])
abs_log = Path(os.environ["BDELF_META_ABS_LOG_DIR"])
gpu_ids = [int(x) for x in os.environ["BDELF_META_GPU_IDS"].split(",") if x.strip() != ""]
meta = {
    "server": os.environ["BDELF_META_SERVER"],
    "scheduler": "common",
    "job_id": os.environ["BDELF_META_JOB_KEY"],
    "pid": int(os.environ["BDELF_META_PID"]),
    "job_name": os.environ["BDELF_META_JOB_NAME"],
    "script": os.environ["BDELF_META_SCRIPT"],
    "cmdline": os.environ["BDELF_META_CMDLINE"],
    "kind": "eval",
    "gpus": int(os.environ["BDELF_META_NGPUS"]),
    "gpu_ids": gpu_ids,
    "log_dir": os.environ["BDELF_META_LOG_DIR"],
    "started_at": os.environ["BDELF_META_STARTED"],
    "state": "RUNNING",
    "holder": os.environ["BDELF_META_HOLDER"],
}
text = json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
(abs_log / "meta.json").write_text(text, encoding="utf-8")
key = os.environ["BDELF_META_JOB_KEY"]
for sub in ("active", "launched"):
    path = root / "temp" / "agent" / sub / f"{key}.json"
    path.write_text(text, encoding="utf-8")
PY

echo "Submitted common eval job $JOB_KEY (pid=$PID)"
echo "log_dir=$JOB_LOG_DIR"
echo "out=$JOB_LOG_DIR/${JOB_NAME}-${PID}.out"
echo "err=$JOB_LOG_DIR/${JOB_NAME}-${PID}.err"
echo "gpu=$JOB_LOG_DIR/gpu-${PID}.log"
