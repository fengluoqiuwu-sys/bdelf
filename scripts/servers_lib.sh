#!/usr/bin/env bash
# 供 sync.sh / sync_web.sh / web.sh / launch-*.sh 共用：从 scripts/servers.csv 解析服务行。
# 用法：SCRIPT_DIR 已指向 scripts/ 后 source 本文件；再 load_server <服务名>。
# 「名字」列：服务名，或 别名:SSH主机（如 ovan:ovan-server）。无冒号则服务名=SSH 主机。
# 脚本参数 / --server / logs/<服务名>/ 用服务名；ssh / rsync 用 SSH 主机。

SERVERS_CSV="${SERVERS_CSV:-${SCRIPT_DIR}/servers.csv}"

SERVER_NAME=""
REMOTE_DIR=""
REMOTE_SSH_TARGET="" # csv 冒号右侧；无冒号则与服务名相同
SERVER_SCHEDULER=""
SERVER_GPU_MAX=""      # csv「最大使用显卡数量」：遗留列，不作为合计限额
SERVER_GPU_PER_JOB=""  # csv「单个ai任务最大使用显卡数量」：单次任务上限
SSH_BASE=(ssh)

# 须先 load_server；供 sync.sh / web.sh 等共用。
remote_ssh() {
  "${SSH_BASE[@]}" "${REMOTE_SSH_TARGET}" "$@"
}

# argv: names | load <want>
_servers_csv_query() {
  python3 - "$SERVERS_CSV" "$@" <<'PY'
import csv, shlex, sys

path = sys.argv[1]
cmd = sys.argv[2] if len(sys.argv) > 2 else "names"


def parse_name(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    if ":" in raw:
        nick, ssh = raw.split(":", 1)
        nick, ssh = nick.strip(), ssh.strip()
        if not nick or not ssh:
            return None
        return nick, ssh, raw
    return raw, raw, raw


def read_rows():
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    except FileNotFoundError:
        if cmd == "names":
            print("(无 scripts/servers.csv)", file=sys.stderr)
            sys.exit(0)
        raise
    rows = []
    seen = {}
    for row in csv.DictReader(lines):
        raw = (row.get("名字") or "").strip()
        if not raw:
            continue
        parsed = parse_name(raw)
        if parsed is None:
            print(f"服务「名字」无效（须为 服务名 或 服务名:SSH主机）: {raw}", file=sys.stderr)
            sys.exit(1)
        nick, ssh, _full = parsed
        if "/" in nick or nick in (".", "..") or any(c.isspace() for c in nick):
            print(f"服务名非法: {nick}", file=sys.stderr)
            sys.exit(1)
        if any(c.isspace() for c in ssh):
            print(f"SSH 主机非法: {ssh}", file=sys.stderr)
            sys.exit(1)
        if nick in seen:
            print(f"重复的服务名: {nick}", file=sys.stderr)
            sys.exit(1)
        seen[nick] = True
        rows.append((nick, ssh, _full, row))
    return rows


rows = read_rows()

if cmd == "names":
    for nick, ssh, _full, _row in rows:
        if nick == ssh:
            print(nick)
        else:
            print(f"{nick} (ssh: {ssh})")
    sys.exit(0)

if cmd != "load" or len(sys.argv) < 4:
    sys.exit(1)

want = sys.argv[3]
nick_hits = [r for r in rows if r[0] == want]
if len(nick_hits) > 1:
    sys.exit(3)
if len(nick_hits) == 1:
    chosen = nick_hits[0]
else:
    ssh_hits = [r for r in rows if r[1] == want or r[2] == want]
    if len(ssh_hits) > 1:
        print(f"服务名 {want} 对应多行（请用冒号左侧的服务名）", file=sys.stderr)
        sys.exit(4)
    if len(ssh_hits) != 1:
        sys.exit(2)
    chosen = ssh_hits[0]

nick, ssh, _full, row = chosen


def g(k):
    return (row.get(k) or "").strip()


print(f"SERVER_NAME={shlex.quote(nick)}")
print(f"REMOTE_SSH_TARGET={shlex.quote(ssh)}")
for key, val in (
    ("REMOTE_DIR", g("工作目录")),
    ("SERVER_SCHEDULER", g("调度类型")),
    ("SERVER_GPU_MAX", g("最大使用显卡数量")),
    ("SERVER_GPU_PER_JOB", g("单个ai任务最大使用显卡数量")),
):
    print(f"{key}={shlex.quote(val)}")
PY
}

list_server_names() {
  _servers_csv_query names
}

# 填充 REMOTE_* / SERVER_* / SSH_BASE；可选校验工作目录非空（默认校验）。
# 可用服务名、SSH 主机或「名字」列原文查找。
load_server() {
  local name="$1"
  local require_dir="${2:-1}"
  if [[ ! -f "${SERVERS_CSV}" ]]; then
    echo "缺少 ${SERVERS_CSV}（gitignore；请按表头自行创建）" >&2
    return 1
  fi
  local row
  row="$(_servers_csv_query load "$name")" || {
    local st=$?
    if [[ "${st}" -eq 2 ]]; then
      echo "未知服务名: ${name}" >&2
      echo "可用名字:" >&2
      list_server_names | sed 's/^/  /' >&2
      return 1
    fi
    if [[ "${st}" -eq 4 ]]; then
      return 1
    fi
    return "${st}"
  }

  eval "${row}"
  SSH_BASE=(ssh)

  if [[ "${require_dir}" == "1" && -z "${REMOTE_DIR}" ]]; then
    echo "服务 ${SERVER_NAME} 的「工作目录」为空" >&2
    return 1
  fi
  if [[ ! "${SERVER_GPU_MAX}" =~ ^[1-9][0-9]*$ ]]; then
    echo "服务 ${SERVER_NAME} 的「最大使用显卡数量」无效（须为正整数）: ${SERVER_GPU_MAX:-<empty>}" >&2
    return 1
  fi
  if [[ ! "${SERVER_GPU_PER_JOB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "服务 ${SERVER_NAME} 的「单个ai任务最大使用显卡数量」无效（须为正整数）: ${SERVER_GPU_PER_JOB:-<empty>}" >&2
    return 1
  fi
  return 0
}

# rsync -e 用的 SSH 命令字符串（须先 load_server）
rsync_rsh_cmd() {
  printf '%s\n' "ssh"
}
