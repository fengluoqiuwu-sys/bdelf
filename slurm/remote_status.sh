#!/usr/bin/env bash
# 本机包装：ssh 到 Slurm 登录机跑 ~/bin/sar gpus，把 stdout/stderr 与退出码原样传回。
# 不占卡、不依赖 daemon。作业提交仍走 sar task add，不要用本脚本当提交闸。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SCRIPT_DIR="$ROOT/scripts"
# shellcheck source=../scripts/servers_lib.sh
source "${SCRIPT_DIR}/servers_lib.sh"

REMOTE_HOST="${REMOTE_HOST:-ovan-server}"
JSON=0
CLUSTER=""

usage() {
  cat <<'EOF'
用法: bash slurm/remote_status.sh [--json] [--cluster ID]

本机调用，经 ssh 在登录节点执行 ~/bin/sar gpus，输出原样传回。
等价于: ssh ovan-server '~/bin/sar gpus …'

环境变量：REMOTE_HOST（默认 ovan-server，须在 scripts/servers.csv 且调度类型=slurm）。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON=1; shift ;;
    --cluster)
      if [[ $# -lt 2 ]]; then
        echo "缺少 --cluster 参数" >&2
        exit 2
      fi
      CLUSTER="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
done

load_server "$REMOTE_HOST" || exit 1
if [[ "${SERVER_SCHEDULER}" != "slurm" ]]; then
  echo "remote_status.sh 仅用于调度类型=slurm 的主机（当前 ${REMOTE_HOST} 为 ${SERVER_SCHEDULER:-?}）" >&2
  exit 1
fi
if [[ -n "${CLUSTER}" && ! "${CLUSTER}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "非法 --cluster: ${CLUSTER}" >&2
  exit 2
fi

# 远端路径用 \$HOME，避免本机展开 ~/bin/sar。
remote_cmd='"$HOME"/bin/sar gpus'
if [[ -n "${CLUSTER}" ]]; then
  remote_cmd+=" --cluster ${CLUSTER}"
fi
if [[ "${JSON}" == "1" ]]; then
  remote_cmd+=" --json"
fi

# shellcheck disable=SC2029
ssh "$REMOTE_HOST" "${remote_cmd}"
