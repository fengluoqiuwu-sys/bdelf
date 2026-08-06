#!/usr/bin/env bash
# 供 sync.sh / ssh.sh 共用：从 scripts/servers.csv 解析服务行。
# 用法：SCRIPT_DIR 已指向 scripts/ 后 source 本文件；再 load_server <名字>。

SERVERS_CSV="${SERVERS_CSV:-${SCRIPT_DIR}/servers.csv}"

SERVER_NAME=""
REMOTE_HOST=""
REMOTE_USER=""
REMOTE_PORT=""
REMOTE_PASSWORD=""
REMOTE_DIR=""
REMOTE_SSH_TARGET="" # user@host 或 host（用户名为空时不加 user@）
SERVER_SCHEDULER=""
SSH_BASE=(ssh)

list_server_names() {
  python3 - "$SERVERS_CSV" <<'PY'
import csv, sys
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
except FileNotFoundError:
    print("(无 scripts/servers.csv)", file=sys.stderr)
    sys.exit(0)
for row in csv.DictReader(lines):
    name = (row.get("名字") or "").strip()
    if name:
        print(name)
PY
}

# 填充 REMOTE_* / SERVER_* / SSH_BASE；可选校验工作目录非空（默认校验）。
load_server() {
  local name="$1"
  local require_dir="${2:-1}"
  if [[ ! -f "${SERVERS_CSV}" ]]; then
    echo "缺少 ${SERVERS_CSV}（gitignore；请按表头自行创建）" >&2
    return 1
  fi
  local row
  row="$(python3 - "$SERVERS_CSV" "$name" <<'PY'
import csv, shlex, sys
path, want = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
for row in csv.DictReader(lines):
    if (row.get("名字") or "").strip() == want:
        def g(k):
            return (row.get(k) or "").strip()
        host = g("IP") or want
        user = g("用户名")
        for key, val in (
            ("REMOTE_HOST", host),
            ("REMOTE_USER", user),
            ("REMOTE_PORT", g("SSH端口")),
            ("REMOTE_PASSWORD", g("连接密码")),
            ("REMOTE_DIR", g("工作目录")),
            ("SERVER_SCHEDULER", g("调度类型")),
        ):
            print(f"{key}={shlex.quote(val)}")
        sys.exit(0)
sys.exit(2)
PY
)" || {
    local st=$?
    if [[ "${st}" -eq 2 ]]; then
      echo "未知服务名: ${name}" >&2
      echo "可用名字:" >&2
      list_server_names | sed 's/^/  /' >&2
      return 1
    fi
    return "${st}"
  }

  eval "${row}"
  SERVER_NAME="${name}"
  if [[ -n "${REMOTE_USER}" ]]; then
    REMOTE_SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
  else
    REMOTE_SSH_TARGET="${REMOTE_HOST}"
  fi

  if [[ "${require_dir}" == "1" && -z "${REMOTE_DIR}" ]]; then
    echo "服务 ${name} 的「工作目录」为空" >&2
    return 1
  fi

  SSH_BASE=(ssh)
  if [[ -n "${REMOTE_PORT}" ]]; then
    SSH_BASE+=(-p "${REMOTE_PORT}")
  fi
  if [[ -n "${REMOTE_PASSWORD}" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
      echo "服务 ${name} 配置了连接密码，但本机无 sshpass" >&2
      return 1
    fi
    export SSHPASS="${REMOTE_PASSWORD}"
    SSH_BASE=(sshpass -e "${SSH_BASE[@]}")
  fi
  return 0
}

# rsync -e 用的 SSH 命令字符串（须先 load_server）
rsync_rsh_cmd() {
  if [[ -n "${REMOTE_PASSWORD}" ]]; then
    if [[ -n "${REMOTE_PORT}" ]]; then
      printf '%s\n' "sshpass -e ssh -p ${REMOTE_PORT}"
    else
      printf '%s\n' "sshpass -e ssh"
    fi
  elif [[ -n "${REMOTE_PORT}" ]]; then
    printf '%s\n' "ssh -p ${REMOTE_PORT}"
  else
    printf '%s\n' "ssh"
  fi
}
