#!/usr/bin/env bash
# 拉起 / 关闭 monitor 网页。远端走 SSH 隧道；push 时 instance 标为 remote。
# 用法: bash scripts/web.sh <local|服务名> {up|down}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=servers_lib.sh
source "${SCRIPT_DIR}/servers_lib.sh"

cd "${ROOT}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY=python3
fi

STATE_DIR="${ROOT}/temp/web"
HOST="127.0.0.1"

usage() {
  cat <<EOF
用法: $(basename "$0") <local|服务名> {up|down}

  local       本机直接跑 monitor.py（--instance local）
  <服务名>    scripts/servers.csv「名字」列：远端跑 monitor（--instance remote），
              本机随机高端口做 SSH 隧道。远端若还没有 monitor.py 会
              sync push --code-only（instance.json 写成 remote）。

  up          拉起服务；已在跑则打印现有地址
  down        关掉本脚本拉起的进程（远端后端 + 本机隧道），避免继续占端口

可用服务名:
  local
$(list_server_names | sed 's/^/  /' 2>/dev/null || echo "  (无 servers.csv)")
EOF
}

die() { echo "$*" >&2; exit 1; }

state_file() {
  printf '%s/%s.json\n' "${STATE_DIR}" "$1"
}

pid_alive() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

kill_pid() {
  local pid="${1:-}"
  pid_alive "${pid}" || return 0
  kill "${pid}" 2>/dev/null || true
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    pid_alive "${pid}" || return 0
    sleep 0.2
  done
  kill -9 "${pid}" 2>/dev/null || true
}

pick_local_port() {
  local prefer="${1:-}"
  "${PY}" - "${prefer}" <<'PY'
import sys
from monitor.port import pick_port, port_available
prefer = sys.argv[1].strip()
if prefer.isdigit():
    p = int(prefer)
    if p >= 16385 and port_available("127.0.0.1", p):
        print(p)
        raise SystemExit(0)
print(pick_port("127.0.0.1"))
PY
}

port_listening() {
  local port="$1"
  "${PY}" - "${HOST}" "${port}" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(0.4)
try:
    s.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
}

wait_listening() {
  local port="$1" tries="${2:-40}"
  local i
  for ((i = 0; i < tries; i++)); do
    if port_listening "${port}"; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

parse_port() {
  local url="$1"
  "${PY}" - "${url}" <<'PY'
import sys
from urllib.parse import urlparse
u = urlparse(sys.argv[1].strip())
if u.port:
    print(u.port)
    raise SystemExit(0)
raise SystemExit(1)
PY
}

write_state() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"
  # 脚本用 -c，stdin 留给调用方传入的 JSON（不可 python - <<PY，否则 stdin 被源码占掉）
  "${PY}" -c 'import json, sys
path = sys.argv[1]
data = json.loads(sys.stdin.read())
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
' "${path}"
}

read_state_field() {
  local path="$1" key="$2"
  [[ -f "${path}" ]] || return 0
  "${PY}" - "${path}" "${key}" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
try:
    data = json.loads(open(path, encoding="utf-8").read())
except (OSError, json.JSONDecodeError):
    raise SystemExit(0)
val = data.get(key)
if val is None or val == "":
    raise SystemExit(0)
print(val)
PY
}

wait_url_file() {
  local file="$1" tries="${2:-80}"
  local i line
  for ((i = 0; i < tries; i++)); do
    if [[ -f "${file}" ]]; then
      line="$(head -n 1 "${file}" || true)"
      if [[ "${line}" == http://* ]]; then
        printf '%s\n' "${line}"
        return 0
      fi
    fi
    sleep 0.25
  done
  return 1
}

start_local_monitor() {
  local out="$1" err="$2" role="$3"
  mkdir -p "$(dirname "${out}")"
  : >"${out}"
  : >"${err}"
  PYTHONUNBUFFERED=1 nohup "${PY}" "${ROOT}/monitor.py" \
    --host "${HOST}" --instance "${role}" \
    < /dev/null >"${out}" 2>"${err}" &
  MONITOR_PID=$!
  disown "${MONITOR_PID}" 2>/dev/null || true
}

cmd_up_local() {
  local sf out err pid url port
  sf="$(state_file local)"
  out="${STATE_DIR}/local.out"
  err="${STATE_DIR}/local.err"
  pid="$(read_state_field "${sf}" monitor_pid || true)"
  port="$(read_state_field "${sf}" local_port || true)"
  if pid_alive "${pid}" && [[ -n "${port}" ]] && port_listening "${port}"; then
    echo "已在运行: http://${HOST}:${port}"
    return 0
  fi
  mkdir -p "${STATE_DIR}"
  start_local_monitor "${out}" "${err}" local
  pid="${MONITOR_PID}"
  url="$(wait_url_file "${out}")" || {
    kill_pid "${pid}"
    echo "本机 monitor 启动失败：" >&2
    tail -n 40 "${err}" >&2 || true
    exit 1
  }
  port="$(parse_port "${url}")"
  wait_listening "${port}" 20 || true
  write_state "${sf}" <<EOF
{"target": "local", "role": "local", "local_port": ${port}, "monitor_pid": ${pid}, "url": "http://${HOST}:${port}"}
EOF
  echo "http://${HOST}:${port}"
}

cmd_down_local() {
  local sf pid
  sf="$(state_file local)"
  pid="$(read_state_field "${sf}" monitor_pid || true)"
  if [[ -n "${pid}" ]]; then
    kill_pid "${pid}"
    echo "已关闭本机 monitor pid=${pid}"
  else
    echo "没有本脚本登记的本机 monitor"
  fi
  rm -f "${sf}"
}

ensure_remote_monitor() {
  # 与 sync.sh 相同：REMOTE_DIR 常为 ~/source/bdelf，不能 printf %q，否则远端 ~ 不展开。
  if remote_ssh "test -f ${REMOTE_DIR}/monitor.py && test -d ${REMOTE_DIR}/monitor"; then
    echo "==> 远端已有 monitor 服务"
  else
    echo "==> 远端没有 monitor.py，push 代码（instance.json → remote）..."
    bash "${SCRIPT_DIR}/sync.sh" "${SERVER_NAME}" push --code-only
  fi
  if ! remote_ssh "test -f ${REMOTE_DIR}/.venv/bin/activate"; then
    die "远端 ${SERVER_NAME} 没有 .venv（缺少 ${REMOTE_DIR}/.venv/bin/activate），无法启动网页"
  fi
  remote_ssh "mkdir -p ${REMOTE_DIR}/cache/monitor && printf '%s\n' '{\"role\": \"remote\"}' > ${REMOTE_DIR}/cache/monitor/instance.json"
}

remote_pid_alive() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  remote_ssh "kill -0 ${pid} 2>/dev/null"
}

kill_remote_pid() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 0
  remote_ssh "if kill -0 ${pid} 2>/dev/null; then kill ${pid} 2>/dev/null || true; sleep 0.4; if kill -0 ${pid} 2>/dev/null; then kill -9 ${pid} 2>/dev/null || true; fi; fi; rm -f ${REMOTE_DIR}/temp/web/monitor.pid"
}

start_remote_monitor() {
  local pid
  # `&` 优先级低于 `&&`：必须先 cd 再用 { python & ... }，否则 pid 文件会写到远端家目录。
  pid="$(remote_ssh "cd ${REMOTE_DIR} && . .venv/bin/activate >/dev/null && mkdir -p temp/web && { PYTHONUNBUFFERED=1 setsid python monitor.py --host 127.0.0.1 --instance remote < /dev/null > temp/web/monitor.out 2> temp/web/monitor.err & echo \$! > temp/web/monitor.pid; cat temp/web/monitor.pid; }")"
  pid="$(printf '%s\n' "${pid}" | awk '/^[0-9]+$/{n=$0} END{print n}')"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || die "无法取得远端 monitor pid（${pid}）"
  printf '%s\n' "${pid}"
}

wait_remote_url() {
  local pid="${1:-}" tries=30 i blob
  for ((i = 0; i < tries; i++)); do
    blob="$(remote_ssh "if ! kill -0 ${pid} 2>/dev/null; then echo DEAD; fi; head -n 1 ${REMOTE_DIR}/temp/web/monitor.out 2>/dev/null || true")"
    if printf '%s\n' "${blob}" | grep -qx DEAD; then
      return 1
    fi
    if printf '%s\n' "${blob}" | grep -q '^http://'; then
      printf '%s\n' "${blob}" | grep '^http://' | head -n 1
      return 0
    fi
    sleep 0.4
  done
  return 1
}

start_tunnel() {
  local local_port="$1" remote_port="$2"
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L "${local_port}:${HOST}:${remote_port}" \
    "${REMOTE_SSH_TARGET}" &
  TUNNEL_PID=$!
  disown "${TUNNEL_PID}" 2>/dev/null || true
}

cmd_up_remote() {
  local sf pid tunnel_pid local_port remote_port url rurl
  sf="$(state_file "${SERVER_NAME}")"
  load_server "${SERVER_NAME}"
  ensure_remote_monitor
  mkdir -p "${STATE_DIR}"

  pid="$(read_state_field "${sf}" monitor_pid || true)"
  tunnel_pid="$(read_state_field "${sf}" tunnel_pid || true)"
  local_port="$(read_state_field "${sf}" local_port || true)"
  remote_port="$(read_state_field "${sf}" remote_port || true)"

  if remote_pid_alive "${pid}" && pid_alive "${tunnel_pid}" && [[ -n "${local_port}" ]] && port_listening "${local_port}"; then
    echo "已在运行: http://${HOST}:${local_port}  （${SERVER_NAME}:${remote_port}）"
    return 0
  fi

  if remote_pid_alive "${pid}" && [[ -n "${remote_port}" ]]; then
    echo "==> 远端 monitor 仍在，重建 SSH 隧道"
  else
    if [[ -n "${pid}" ]]; then
      kill_remote_pid "${pid}" || true
    fi
    echo "==> 在 ${SERVER_NAME} 拉起 monitor（--instance remote）..."
    pid="$(start_remote_monitor)"
    rurl="$(wait_remote_url "${pid}")" || {
      echo "远端 monitor 启动失败：" >&2
      remote_ssh "tail -n 50 ${REMOTE_DIR}/temp/web/monitor.err; echo '--- out ---'; tail -n 20 ${REMOTE_DIR}/temp/web/monitor.out" >&2 || true
      kill_remote_pid "${pid}" || true
      exit 1
    }
    remote_port="$(parse_port "${rurl}")"
  fi

  if pid_alive "${tunnel_pid}"; then
    kill_pid "${tunnel_pid}"
  fi
  local_port="$(pick_local_port "${remote_port}")"
  start_tunnel "${local_port}" "${remote_port}"
  tunnel_pid="${TUNNEL_PID}"
  if ! wait_listening "${local_port}" 40; then
    kill_pid "${tunnel_pid}"
    die "SSH 隧道未能在本机 ${local_port} 监听"
  fi
  url="http://${HOST}:${local_port}"
  write_state "${sf}" <<EOF
{"target": "${SERVER_NAME}", "role": "remote", "local_port": ${local_port}, "remote_port": ${remote_port}, "monitor_pid": ${pid}, "tunnel_pid": ${tunnel_pid}, "url": "${url}"}
EOF
  echo "${url}  ←  ${SERVER_NAME} 127.0.0.1:${remote_port}"
}

cmd_down_remote() {
  local sf pid tunnel_pid extra
  load_server "${SERVER_NAME}"
  sf="$(state_file "${SERVER_NAME}")"
  pid="$(read_state_field "${sf}" monitor_pid || true)"
  tunnel_pid="$(read_state_field "${sf}" tunnel_pid || true)"

  if [[ -z "${pid}" ]]; then
    extra="$(remote_ssh "test -f ${REMOTE_DIR}/temp/web/monitor.pid && cat ${REMOTE_DIR}/temp/web/monitor.pid" || true)"
    extra="$(printf '%s' "${extra}" | tr -d '[:space:]')"
    pid="${extra}"
  fi

  if [[ -n "${tunnel_pid}" ]]; then
    kill_pid "${tunnel_pid}"
    echo "已关闭本机隧道 pid=${tunnel_pid}"
  fi
  if [[ -n "${pid}" ]]; then
    kill_remote_pid "${pid}"
    echo "已关闭 ${SERVER_NAME} monitor pid=${pid}"
  else
    echo "没有登记的远端 monitor pid（${SERVER_NAME}）"
  fi
  rm -f "${sf}"
}

main() {
  case "${1:-}" in
    -h|--help|"") usage; [[ -n "${1:-}" ]] && exit 0; exit 1 ;;
  esac
  if [[ $# -lt 2 ]]; then
    usage
    exit 1
  fi
  local target="$1" action="$2"
  case "${action}" in
    up|down) ;;
    *) usage; exit 1 ;;
  esac
  case "${target}" in
    local)
      if [[ "${action}" == up ]]; then cmd_up_local; else cmd_down_local; fi
      ;;
    *)
      SERVER_NAME="${target}"
      if [[ "${action}" == up ]]; then cmd_up_remote; else cmd_down_remote; fi
      ;;
  esac
}

main "$@"
