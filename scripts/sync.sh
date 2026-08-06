#!/usr/bin/env bash
# 按 scripts/servers.csv 中的服务名，与远端同步 bdelf：push 推代码 + models/tokenizers；pull 拉 checkpoints。
# 用法: bash scripts/sync.sh <服务名> {push|pull|pull-file} ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=servers_lib.sh
source "${SCRIPT_DIR}/servers_lib.sh"

RSYNC_OPTS=(-az -h --info=progress2)
# 代码 push：镜像同步，删除远端多余文件（排除项如 .venv/cache 不删）
RSYNC_CODE_OPTS=(-az -h --info=progress2 --delete)
# cache 内容：默认按 size+mtime 增量（大文件在 /mnt/d 上 --checksum 要整文件读盘，极慢）
RSYNC_CACHE_OPTS=(-az -h --info=progress2)
RSYNC_CACHE_CHECKSUM_OPTS=(-az -h --info=progress2 --checksum)
# 默认推：models / tokenizers（权重与分词器实体）
CACHE_CONTENT_DIRS=(models tokenizers)
RSYNC_CACHE_CONTENT_EXCLUDES=(
  --exclude='.cache/'
  --exclude='.locks/'
  --exclude='.no_exist/'
  --exclude='*.lock'
  --exclude='CACHEDIR.TAG'
  --exclude='.agent_harnesses.json'
  --exclude='*.incomplete'
  --exclude='tmp/'
  --exclude='__pycache__/'
)
RSYNC_HASH_GUIDE_EXCLUDE=(--exclude='hash_guide.csv')

RSYNC_CODE_FILTERS=(
  --filter='- .venv/'
  --exclude='cache'
  --exclude='temp/'
  --exclude='venv/'
  --exclude='env/'
  --exclude='ENV/'
  --exclude='.git/'
  --exclude='__pycache__/'
  --exclude='*.py[cod]'
  --exclude='.cursor/'
  --exclude='.claude/'
  --exclude='.idea/'
  --exclude='.vscode/'
  --exclude='.env'
  --exclude='.env.*'
  --exclude='*.log'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
  --exclude='build/'
  --exclude='dist/'
  --exclude='*.egg-info/'
)

RSYNC_RSH=""

remote_ssh() {
  "${SSH_BASE[@]}" "${REMOTE_SSH_TARGET}" "$@"
}

rsync_to() {
  rsync -e "${RSYNC_RSH}" "$@"
}

usage() {
  cat <<EOF
用法: $(basename "$0") <服务名> {push|pull|pull-file} ...

  <服务名>  scripts/servers.csv 中「名字」列（如 ovan-server）

  push [--code-only] [--with-datasets] [--checksum]
      强制覆盖推送代码到 <服务>:<工作目录>（删除远端多余文件；.venv/cache/temp 等排除项保留）
      默认再推 cache 内容：models/ tokenizers/（按 size+mtime 增量；排除 .cache/ 等）
      默认不推：datasets/、huggingface/、preprocessed_datasets/、checkpoints/、compile*
      --with-datasets  额外推 datasets/
      --checksum       cache 内容整文件校验后再传（本地 cache 在 /mnt/d 时很慢，慎用）
      --code-only      只推代码，跳过 cache 内容
      temp/ 不同步；cache/checkpoints/hash_guide.csv 仅本地（pull 亦排除）

  pull [--mode MODE] [NAME]
      从远端增量同步 cache/checkpoints/[NAME]/（排除 hash_guide.csv）
      --mode fast（默认）| common | full

  pull-file NAME FILE
      拉取单个文件：cache/checkpoints/NAME/FILE

可用服务名:
$(list_server_names | sed 's/^/  /' 2>/dev/null || echo "  (无 servers.csv)")
EOF
}

push_code() {
  echo "==> 推送代码到 ${REMOTE_SSH_TARGET}:${REMOTE_DIR}（服务=${SERVER_NAME}；强制覆盖）..."
  rsync_to "${RSYNC_CODE_OPTS[@]}" \
    "${RSYNC_CODE_FILTERS[@]}" \
    "${LOCAL_DIR}/" "${REMOTE_SSH_TARGET}:${REMOTE_DIR}/"
}

ensure_remote_cache_dir() {
  # 远端 cache 必须是真实目录；若曾被误推为软链接则先删除
  remote_ssh "d=${REMOTE_DIR}/cache; [ -L \"\$d\" ] && rm \"\$d\"; mkdir -p \"\$d\""
}

push_cache_content() {
  local use_checksum=0
  local dirs=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checksum) use_checksum=1; shift ;;
      *) dirs+=("$1"); shift ;;
    esac
  done
  if [[ ${#dirs[@]} -eq 0 ]]; then
    dirs=("${CACHE_CONTENT_DIRS[@]}")
  fi
  local opts=("${RSYNC_CACHE_OPTS[@]}")
  local mode_msg="size+mtime"
  if [[ "${use_checksum}" -eq 1 ]]; then
    opts=("${RSYNC_CACHE_CHECKSUM_OPTS[@]}")
    mode_msg="checksum（整文件读盘）"
  fi
  echo "==> 推送 cache 内容（${dirs[*]}；${mode_msg}；跳过 .cache/ 等）..."
  if [[ ! -e "${LOCAL_DIR}/cache" ]]; then
    echo "本地 cache 不存在，跳过"
    return 0
  fi
  ensure_remote_cache_dir
  local name local_src remote_dst
  for name in "${dirs[@]}"; do
    local_src="${LOCAL_DIR}/cache/${name}"
    if [[ ! -e "${local_src}" ]]; then
      echo "    跳过 ${name}/（本地不存在）"
      continue
    fi
    remote_dst="${REMOTE_SSH_TARGET}:${REMOTE_DIR}/cache/${name}/"
    echo "    → ${name}/"
    rsync_to "${opts[@]}" \
      "${RSYNC_CACHE_CONTENT_EXCLUDES[@]}" \
      "${local_src}/" "${remote_dst}"
  done
}

pull_filters_for_mode() {
  local mode="$1"
  PULL_FILTERS=()
  case "${mode}" in
    fast)
      PULL_FILTERS=(--exclude='*.pt')
      ;;
    common)
      PULL_FILTERS=(
        --include='checkpoint_latest.pt'
        --exclude='*.pt'
      )
      ;;
    full)
      PULL_FILTERS=()
      ;;
    *)
      echo "未知 --mode: ${mode}（可选: fast|common|full）" >&2
      exit 1
      ;;
  esac
}

pull_checkpoints() {
  local mode="fast"
  local name=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode)
        [[ $# -ge 2 ]] || { echo "缺少 --mode 参数值" >&2; exit 1; }
        mode="$2"
        shift 2
        ;;
      --mode=*)
        mode="${1#--mode=}"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      -*)
        echo "未知选项: $1" >&2
        usage >&2
        exit 1
        ;;
      *)
        if [[ -n "${name}" ]]; then
          echo "多余参数: $1（已指定 NAME=${name}）" >&2
          exit 1
        fi
        name="$1"
        shift
        ;;
    esac
  done

  pull_filters_for_mode "${mode}"

  local remote_src local_dst scope_msg
  if [[ -n "${name}" ]]; then
    remote_src="${REMOTE_SSH_TARGET}:${REMOTE_DIR}/cache/checkpoints/${name}/"
    local_dst="${LOCAL_DIR}/cache/checkpoints/${name}/"
    scope_msg="checkpoints/${name}/"
  else
    remote_src="${REMOTE_SSH_TARGET}:${REMOTE_DIR}/cache/checkpoints/"
    local_dst="${LOCAL_DIR}/cache/checkpoints/"
    scope_msg="checkpoints/"
  fi

  echo "==> 从 ${REMOTE_SSH_TARGET}（服务=${SERVER_NAME}）增量同步 ${scope_msg}（mode=${mode}）..."
  ensure_remote_cache_dir
  mkdir -p "${local_dst}"
  if [[ ${#PULL_FILTERS[@]} -gt 0 ]]; then
    rsync_to "${RSYNC_OPTS[@]}" "${RSYNC_HASH_GUIDE_EXCLUDE[@]}" "${PULL_FILTERS[@]}" \
      "${remote_src}" "${local_dst}"
  else
    rsync_to "${RSYNC_OPTS[@]}" "${RSYNC_HASH_GUIDE_EXCLUDE[@]}" \
      "${remote_src}" "${local_dst}"
  fi
}

pull_file() {
  if [[ $# -ne 2 ]]; then
    echo "用法: $(basename "$0") <服务名> pull-file NAME FILE" >&2
    exit 1
  fi
  local name="$1"
  local file="$2"
  local remote_src="${REMOTE_SSH_TARGET}:${REMOTE_DIR}/cache/checkpoints/${name}/${file}"
  local local_dst="${LOCAL_DIR}/cache/checkpoints/${name}/${file}"

  echo "==> 拉取 ${REMOTE_SSH_TARGET}:cache/checkpoints/${name}/${file}（服务=${SERVER_NAME}）..."
  ensure_remote_cache_dir
  mkdir -p "$(dirname "${local_dst}")"
  rsync_to "${RSYNC_OPTS[@]}" "${remote_src}" "${local_dst}"
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
  push|pull|pull-file)
    echo "请先指定 servers.csv 中的服务名，例如: $(basename "$0") ovan-server $1 ..." >&2
    usage >&2
    exit 1
    ;;
esac

load_server "$1" || exit 1
RSYNC_RSH="$(rsync_rsh_cmd)"
shift

cmd="${1:-}"
case "${cmd}" in
  push)
    shift
    code_only=0
    with_datasets=0
    with_checksum=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --code-only) code_only=1; shift ;;
        --with-datasets) with_datasets=1; shift ;;
        --checksum) with_checksum=1; shift ;;
        --with-cache)
          echo "提示: --with-cache 已废弃；默认推 models/tokenizers（加 --with-datasets 才推数据集）" >&2
          shift
          ;;
        -h|--help) usage; exit 0 ;;
        *)
          echo "未知 push 选项: $1（可用 --code-only / --with-datasets / --checksum）" >&2
          usage >&2
          exit 1
          ;;
      esac
    done
    push_code
    if [[ "${code_only}" -eq 1 ]]; then
      echo "==> 跳过 cache 内容（--code-only）"
    else
      cache_args=()
      [[ "${with_checksum}" -eq 1 ]] && cache_args+=(--checksum)
      cache_args+=("${CACHE_CONTENT_DIRS[@]}")
      if [[ "${with_datasets}" -eq 1 ]]; then
        cache_args+=(datasets)
      fi
      push_cache_content "${cache_args[@]}"
      echo "==> 未推送：huggingface/（缓冲区）、checkpoints/、预处理、compile*；datasets/ 默认跳过（需加 --with-datasets）"
    fi
    echo "==> push 完成（${SERVER_NAME}）"
    ;;
  pull)
    shift
    pull_checkpoints "$@"
    echo "==> pull 完成（${SERVER_NAME}）"
    ;;
  pull-file)
    shift
    pull_file "$@"
    echo "==> pull-file 完成（${SERVER_NAME}）"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
