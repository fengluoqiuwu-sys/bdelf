#!/usr/bin/env bash
# sync 的 web / monitor 事务：按 hash 合并 cache/monitor/charts.json。
# 由 scripts/sync.sh 调用；也可单独跑。
# 用法: bash scripts/sync_web.sh <服务名> {push|pull}
#
# push：不覆盖远端已有 hash 的图表，只把本机有、远端没有的补上去；
#       instance.json 不推，写完后在远端标为 remote。
# pull：远端有而本机没有的 hash 才写入本机，本机已有的忽略。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=servers_lib.sh
source "${SCRIPT_DIR}/servers_lib.sh"

CHARTS_REL="cache/monitor/charts.json"
INSTANCE_REL="cache/monitor/instance.json"

usage() {
  cat <<EOF
用法: $(basename "$0") <服务名> {push|pull}

  合并 cache/monitor/charts.json（按 hash；已有的不覆盖）：
    push  本机 → 远端：不覆盖远端已保存的 hash 图表，只补远端没有的
    pull  远端 → 本机：远端有而本机没有的 hash 才添加，否则忽略

  instance.json 不参与合并；push 后在远端写成 remote。
EOF
}

# dest 已有的 hash 保留；src 有而 dest 没有的 hash 补上。缺文件视为空库。
merge_charts_json() {
  local dest_path="$1" src_path="$2" out_path="$3"
  python3 - "${dest_path}" "${src_path}" "${out_path}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def empty_store():
    return {
        "version": 1,
        "charts": {},
        "export_prefs": {"invert": False, "width": 960, "height": 360},
    }


def load_store(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return empty_store()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_store()
    if not isinstance(data, dict):
        return empty_store()
    data.setdefault("version", 1)
    data.setdefault("charts", {})
    data.setdefault("export_prefs", {"invert": False, "width": 960, "height": 360})
    if not isinstance(data["charts"], dict):
        data["charts"] = {}
    return data


def hash_maps(bucket: dict) -> tuple[dict, dict]:
    dismissed = bucket.get("dismissed")
    order = bucket.get("order")
    if not isinstance(dismissed, dict):
        dismissed = {}
    if not isinstance(order, dict):
        order = {}
    return dismissed, order


def merge_bucket(dest: dict, src: dict) -> tuple[dict, int]:
    """dest 的 panels / 已有 hash 不动；只补 dest 没有的 hash。"""
    dest = dest if isinstance(dest, dict) else {}
    src = src if isinstance(src, dict) else {}
    d_disc, d_ord = hash_maps(dest)
    s_disc, s_ord = hash_maps(src)
    dest_hashes = set(d_disc) | set(d_ord)
    added = 0
    dismissed = dict(d_disc)
    order = dict(d_ord)
    for h in set(s_disc) | set(s_ord):
        if not isinstance(h, str) or not h or "/" in h or "\\" in h:
            continue
        if h in dest_hashes:
            continue
        if h in s_disc:
            dismissed[h] = s_disc[h]
        if h in s_ord:
            order[h] = s_ord[h]
        added += 1
    if "panels" in dest and isinstance(dest.get("panels"), list):
        panels = dest["panels"]
    else:
        panels = src["panels"] if isinstance(src.get("panels"), list) else []
    return {"panels": panels, "dismissed": dismissed, "order": order}, added


dest_path, src_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
dest = load_store(dest_path)
src = load_store(src_path)
out = empty_store()
out["version"] = dest.get("version") or src.get("version") or 1
if isinstance(dest.get("export_prefs"), dict) and dest.get("export_prefs"):
    out["export_prefs"] = dest["export_prefs"]
elif isinstance(src.get("export_prefs"), dict) and src.get("export_prefs"):
    out["export_prefs"] = src["export_prefs"]

added_total = 0
bucket_added = 0
dest_charts = dest["charts"]
src_charts = src["charts"] if isinstance(src.get("charts"), dict) else {}
keys = list(dest_charts.keys())
for k in src_charts:
    if k not in dest_charts:
        keys.append(k)
for key in keys:
    if not isinstance(key, str) or not key:
        continue
    d_bucket = dest_charts.get(key)
    s_bucket = src_charts.get(key)
    if not isinstance(d_bucket, dict) and isinstance(s_bucket, dict):
        merged, n = merge_bucket({}, s_bucket)
        out["charts"][key] = merged
        added_total += n
        if n:
            bucket_added += 1
        continue
    if not isinstance(d_bucket, dict):
        continue
    merged, n = merge_bucket(d_bucket, s_bucket if isinstance(s_bucket, dict) else {})
    out["charts"][key] = merged
    added_total += n
    if n:
        bucket_added += 1

out["updated_at"] = datetime.now(timezone.utc).isoformat()
dest_p = Path(out_path)
dest_p.parent.mkdir(parents=True, exist_ok=True)
tmp = dest_p.with_name(dest_p.name + ".tmp")
tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp.replace(dest_p)
print(f"{added_total} {bucket_added}")
PY
}

fetch_remote_charts() {
  local dest="$1"
  if remote_ssh "test -f ${REMOTE_DIR}/${CHARTS_REL}"; then
    rsync -e "${RSYNC_RSH}" -az "${REMOTE_SSH_TARGET}:${REMOTE_DIR}/${CHARTS_REL}" "${dest}"
    return 0
  fi
  return 1
}

put_remote_charts() {
  local src="$1"
  remote_ssh "mkdir -p ${REMOTE_DIR}/cache/monitor"
  rsync -e "${RSYNC_RSH}" -az "${src}" "${REMOTE_SSH_TARGET}:${REMOTE_DIR}/${CHARTS_REL}"
}

write_remote_instance() {
  remote_ssh "mkdir -p ${REMOTE_DIR}/cache/monitor && printf '%s\n' '{\"role\": \"remote\"}' > ${REMOTE_DIR}/${INSTANCE_REL}"
}

cmd_push() {
  local tmp remote_copy merged counts added buckets
  echo "==> 合并推送 ${CHARTS_REL}（不覆盖远端已有 hash；instance.json 标为 remote）..."
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '${tmp}'" RETURN
  remote_copy="${tmp}/remote.json"
  merged="${tmp}/merged.json"
  if fetch_remote_charts "${remote_copy}"; then
    echo "    已取远端图表配置，按 hash 合并"
  else
    echo "    远端尚无图表配置，按本机写入"
    : >"${remote_copy}"
  fi
  counts="$(merge_charts_json "${remote_copy}" "${LOCAL_DIR}/${CHARTS_REL}" "${merged}")"
  added="${counts%% *}"
  buckets="${counts##* }"
  put_remote_charts "${merged}"
  write_remote_instance
  if [[ "${added}" -gt 0 ]]; then
    echo "    补了 ${added} 个远端没有的 hash 图表（${buckets} 个 kind/model）"
  else
    echo "    远端已有的 hash 未覆盖；没有可补充的 hash"
  fi
}

cmd_pull() {
  local tmp remote_copy merged counts added buckets
  echo "==> 合并拉取 ${CHARTS_REL}（只补本机没有的 hash，已有的忽略）..."
  if ! remote_ssh "test -f ${REMOTE_DIR}/${CHARTS_REL}"; then
    echo "    （远端尚无图表配置，跳过）"
    return 0
  fi
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '${tmp}'" RETURN
  remote_copy="${tmp}/remote.json"
  merged="${tmp}/merged.json"
  fetch_remote_charts "${remote_copy}"
  mkdir -p "${LOCAL_DIR}/cache/monitor"
  counts="$(merge_charts_json "${LOCAL_DIR}/${CHARTS_REL}" "${remote_copy}" "${merged}")"
  added="${counts%% *}"
  buckets="${counts##* }"
  mv -f "${merged}" "${LOCAL_DIR}/${CHARTS_REL}"
  if [[ "${added}" -gt 0 ]]; then
    echo "    补了 ${added} 个本机没有的 hash 图表（${buckets} 个 kind/model）"
  else
    echo "    本机已有对应 hash，忽略远端图表"
  fi
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

case "${1}" in
  -h|--help)
    usage
    exit 0
    ;;
  push|pull)
    echo "请先指定 servers.csv 中的服务名，例如: $(basename "$0") ovan $1" >&2
    usage >&2
    exit 1
    ;;
esac

load_server "$1" || exit 1
RSYNC_RSH="$(rsync_rsh_cmd)"
shift

case "${1:-}" in
  push) cmd_push ;;
  pull) cmd_pull ;;
  -h|--help) usage; exit 0 ;;
  "") usage >&2; exit 1 ;;
  *)
    echo "未知命令: $1（可用 push / pull）" >&2
    usage >&2
    exit 1
    ;;
esac
