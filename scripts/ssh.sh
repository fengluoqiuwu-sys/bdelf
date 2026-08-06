#!/usr/bin/env bash
# 按 scripts/servers.csv 服务名 SSH（免每次查 IP/端口/密码/工作目录）。
# 用法:
#   bash scripts/ssh.sh <服务名>              # 交互登录，进入工作目录
#   bash scripts/ssh.sh <服务名> -- <远端命令...>
#   bash scripts/ssh.sh <服务名> <远端命令...>  # 在工作目录执行后退出
#   bash scripts/ssh.sh --list
#   bash scripts/ssh.sh <服务名> --print       # 只打印连接信息
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=servers_lib.sh
source "${SCRIPT_DIR}/servers_lib.sh"

usage() {
  cat <<EOF
用法: $(basename "$0") <服务名> [--print | -- <远端命令...> | <远端命令...>]
      $(basename "$0") --list

  <服务名>  scripts/servers.csv 的「名字」列
  --list    列出可用服务名
  --print   打印 host/port/目录/调度类型后退出（不连接）
  无额外参数：交互式 SSH，登录后 cd 到工作目录
  有额外参数：在工作目录非交互执行命令

可用服务名:
$(list_server_names | sed 's/^/  /' 2>/dev/null || echo "  (无 servers.csv)")
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
  --list|-l)
    list_server_names
    exit 0
    ;;
esac

NAME="$1"
shift
load_server "$NAME" || exit 1

PRINT=0
REMOTE_CMD=()
if [[ $# -gt 0 ]]; then
  case "$1" in
    --print)
      PRINT=1
      shift
      if [[ $# -gt 0 ]]; then
        echo "--print 后不应再跟参数" >&2
        exit 1
      fi
      ;;
    --)
      shift
      REMOTE_CMD=("$@")
      ;;
    *)
      REMOTE_CMD=("$@")
      ;;
  esac
fi

if [[ "${PRINT}" -eq 1 ]]; then
  cat <<EOF
名字=${SERVER_NAME}
host=${REMOTE_HOST}
user=${REMOTE_USER:-(默认)}
port=${REMOTE_PORT:-22}
工作目录=${REMOTE_DIR}
调度类型=${SERVER_SCHEDULER}
认证=$([ -n "${REMOTE_PASSWORD}" ] && echo password || echo key)
EOF
  exit 0
fi

# 远端 cd：工作目录可能含 ~（单引号会阻止 ~ 展开，故 ~/… 改写成 $HOME/…）
remote_cd_prefix() {
  local d="${REMOTE_DIR}"
  if [[ "${d}" == "~" ]]; then
    printf 'cd "$HOME"'
  elif [[ "${d}" == "~/"* ]]; then
    local rest="${d:2}"
    rest="${rest//\'/\'\\\'\'}"
    printf "cd \"\$HOME/%s\"" "${rest}"
  else
    # 单引号包裹路径中的单引号：' -> '\''
    local quoted="${d//\'/\'\\\'\'}"
    printf "cd '%s'" "${quoted}"
  fi
}

if [[ ${#REMOTE_CMD[@]} -eq 0 ]]; then
  echo "==> SSH ${SERVER_NAME} (${REMOTE_SSH_TARGET}:${REMOTE_PORT:-22}) → ${REMOTE_DIR}" >&2
  # -t：分配伪终端，便于交互
  exec "${SSH_BASE[@]}" -t "${REMOTE_SSH_TARGET}" "$(remote_cd_prefix) && exec \"\${SHELL:-bash}\" -l"
fi

# 非交互：把命令拼进远端 shell（与 ssh host 'cmd' 一致）
remote_script="$(remote_cd_prefix) && "
# 用 printf %q 逐参数转义后拼接
for a in "${REMOTE_CMD[@]}"; do
  remote_script+=" $(printf '%q' "$a")"
done

echo "==> SSH ${SERVER_NAME}: ${REMOTE_CMD[*]}" >&2
exec "${SSH_BASE[@]}" "${REMOTE_SSH_TARGET}" "${remote_script}"
