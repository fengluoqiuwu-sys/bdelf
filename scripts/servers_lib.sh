#!/usr/bin/env bash
# 供 sync.sh / web.sh 共用：从 scripts/servers.csv 解析服务行。
# 用法：SCRIPT_DIR 已指向 scripts/ 后 source 本文件；再 load_server <名字>。
# SSH：一律系统 ssh，目标即「名字」（须已可 ssh <名字>）。

SERVERS_CSV="${SERVERS_CSV:-${SCRIPT_DIR}/servers.csv}"

SERVER_NAME=""
REMOTE_DIR=""
REMOTE_SSH_TARGET="" # 即服务「名字」
SERVER_SCHEDULER=""
SERVER_GPU_MAX=""      # csv「最大使用显卡数量」：AI 合计额度
SERVER_GPU_PER_JOB=""  # csv「单个ai任务最大使用显卡数量」
SSH_BASE=(ssh)

# 须先 load_server；供 sync.sh / web.sh 等共用。
remote_ssh() {
  "${SSH_BASE[@]}" "${REMOTE_SSH_TARGET}" "$@"
}

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
        for key, val in (
            ("REMOTE_DIR", g("工作目录")),
            ("SERVER_SCHEDULER", g("调度类型")),
            ("SERVER_GPU_MAX", g("最大使用显卡数量")),
            ("SERVER_GPU_PER_JOB", g("单个ai任务最大使用显卡数量")),
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
  REMOTE_SSH_TARGET="${name}"
  SSH_BASE=(ssh)

  if [[ "${require_dir}" == "1" && -z "${REMOTE_DIR}" ]]; then
    echo "服务 ${name} 的「工作目录」为空" >&2
    return 1
  fi
  if [[ ! "${SERVER_GPU_MAX}" =~ ^[1-9][0-9]*$ ]]; then
    echo "服务 ${name} 的「最大使用显卡数量」无效（须为正整数）: ${SERVER_GPU_MAX:-<empty>}" >&2
    return 1
  fi
  if [[ ! "${SERVER_GPU_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "服务 ${name} 的「单个ai任务最大使用显卡数量」无效（须为正整数）: ${SERVER_GPU_PER_JOB:-<empty>}" >&2
    return 1
  fi
  return 0
}

# rsync -e 用的 SSH 命令字符串（须先 load_server）
rsync_rsh_cmd() {
  printf '%s\n' "ssh"
}
