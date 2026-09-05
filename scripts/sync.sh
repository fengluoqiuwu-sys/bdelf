#!/usr/bin/env bash
# 按 scripts/servers.csv 中的服务名，与远端同步 bdelf：push 推代码 + models/tokenizers；pull 拉 checkpoints。
# 用法: bash scripts/sync.sh <服务名> {push|pull|pull-file} ...
# web / monitor 图表由 scripts/sync_web.sh 处理（主脚本调用）。
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
  # logs/：gitignore；代码 push 不传、且 --delete 不删远端（由 pull 单独拉取）
  --filter='P logs/'
  --exclude='logs/'
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

rsync_to() {
  rsync -e "${RSYNC_RSH}" "$@"
}

usage() {
  cat <<EOF
用法: $(basename "$0") <服务名> {push|pull|pull-file} ...

  <服务名>  servers.csv「名字」冒号左侧（无冒号则整列；如 ovan 或 ovan-server）

  push [--code-only] [--with-datasets] [--checksum] [--checkpoints NAME FILE]...
      强制覆盖推送代码到 <服务>:<工作目录>（删除远端多余文件；.venv/cache/temp 等排除项保留）
      若仓库根 requirements.txt 相对远端内容有变且远端已有 .venv，
      则在推完代码后执行 .venv/bin/pip install -r requirements.txt。
      默认再推 cache 内容：models/ tokenizers/（按 size+mtime 增量；排除 .cache/ 等）
      默认不推：datasets/、huggingface/、preprocessed_datasets/、checkpoints/、compile*
      --with-datasets            额外推 datasets/
      --checksum                 cache 内容（含指定 checkpoint 文件）整文件校验后再传（慢，慎用）
      --code-only                只推代码，跳过 models/tokenizers（不影响 --checkpoints；仍合并 cache/monitor/）
      --checkpoints NAME FILE    额外推单个文件：cache/checkpoints/NAME/FILE（可重复）
                                 NAME 形如 full/odar/<hash>；FILE 如 checkpoint_latest.pt
                                 同时增量推对应 cache/eval/<model>/<hash>/（若本地有；避免远端重复评测）
      temp/ 不同步；logs/ gitignore，push 不覆盖/不删除远端，由 pull 拉取
      cache/monitor/ 由 scripts/sync_web.sh 按 hash 合并 charts.json（push/pull 均调用）
      push 不覆盖远端已有 hash 的图表，只补远端没有的；instance.json 不推，远端写成 remote
      cache/checkpoints/hash_guide.csv 仅本地（push/pull 均排除）

  pull [--mode MODE] [NAME]
      从远端增量同步 cache/checkpoints/[NAME]/（排除 hash_guide.csv）
      并增量拉取 logs/（作业 .out/.err/gpu.log）与 cache/eval/（体量小）
      cache/monitor/charts.json：远端有而本机没有的 hash 才添加，本机已有的忽略
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

# 本地仓库根 requirements.txt 相对远端是否有内容变化（含远端尚无此文件）。
# 无本地文件则视为未更新。须在 rsync 之前调用。
requirements_txt_updated() {
  local local_req="${LOCAL_DIR}/requirements.txt"
  [[ -f "${local_req}" ]] || return 1
  local local_hash remote_hash
  local_hash="$(sha256sum "${local_req}" | awk '{print $1}')"
  remote_hash="$(remote_ssh "if [ -f ${REMOTE_DIR}/requirements.txt ]; then sha256sum ${REMOTE_DIR}/requirements.txt | awk '{print \$1}'; fi")"
  [[ "${local_hash}" != "${remote_hash}" ]]
}

remote_has_venv() {
  remote_ssh "test -d ${REMOTE_DIR}/.venv"
}

# 远端已有 .venv 时按刚推上去的仓库根 requirements.txt 重装依赖；不创建 venv。
install_remote_requirements() {
  echo "==> 远端 requirements.txt 已更新且存在 .venv，执行 pip install..."
  remote_ssh "cd ${REMOTE_DIR} && if [ ! -x .venv/bin/pip ]; then echo '远端 .venv 缺少 .venv/bin/pip' >&2; exit 1; fi && .venv/bin/pip install -r requirements.txt"
}

ensure_remote_cache_dir() {
  # 允许远端 cache 为指向数据盘的软链（如 autodl-tmp）；mkdir -p 会沿软链创建目标。
  # 勿删除软链，否则 push/pull 会把大文件写回系统盘。
  remote_ssh "mkdir -p ${REMOTE_DIR}/cache"
}

sync_web() {
  # web / monitor 图表合并：见 scripts/sync_web.sh（push 不覆盖远端已有 hash，pull 只补本机缺失）
  bash "${SCRIPT_DIR}/sync_web.sh" "${SERVER_NAME}" "$1"
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

# 推送指定 checkpoint 文件（NAME=variant/model/hash，FILE 如 checkpoint_latest.pt）；
# 并对每个 NAME 增量推送对应 cache/eval/{model}/{hash}/（若存在），供远端 eval 跳过已跑组。
# 不做 --delete。
push_checkpoint_files() {
  local use_checksum=0
  local specs=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checksum) use_checksum=1; shift ;;
      *) specs+=("$1"); shift ;;
    esac
  done
  if [[ ${#specs[@]} -eq 0 ]]; then
    return 0
  fi
  if (( ${#specs[@]} % 2 != 0 )); then
    echo "内部错误: push_checkpoint_files 需要成对的 NAME FILE" >&2
    exit 1
  fi
  local opts=("${RSYNC_CACHE_OPTS[@]}")
  local mode_msg="size+mtime"
  if [[ "${use_checksum}" -eq 1 ]]; then
    opts=("${RSYNC_CACHE_CHECKSUM_OPTS[@]}")
    mode_msg="checksum（整文件读盘）"
  fi
  local n_files=$(( ${#specs[@]} / 2 ))
  echo "==> 推送指定 checkpoint 文件（${n_files} 个；${mode_msg}）..."
  ensure_remote_cache_dir
  local i name file local_src remote_dst
  local -A EVAL_NAMES=()
  for (( i = 0; i < ${#specs[@]}; i += 2 )); do
    name="${specs[i]}"
    file="${specs[i + 1]}"
    if [[ "${name}" == *".."* ]] || [[ "${name}" == /* ]]; then
      echo "非法 checkpoints NAME: ${name}" >&2
      exit 1
    fi
    if [[ "${file}" == *".."* ]] || [[ "${file}" == /* ]] || [[ "${file}" == *\/* ]]; then
      echo "非法 checkpoints FILE: ${file}（须为 NAME 目录下的单层文件名）" >&2
      exit 1
    fi
    local_src="${LOCAL_DIR}/cache/checkpoints/${name}/${file}"
    if [[ ! -f "${local_src}" ]]; then
      echo "本地不存在 cache/checkpoints/${name}/${file}" >&2
      exit 1
    fi
    remote_dst="${REMOTE_SSH_TARGET}:${REMOTE_DIR}/cache/checkpoints/${name}/${file}"
    echo "    → checkpoints/${name}/${file}"
    remote_ssh "mkdir -p ${REMOTE_DIR}/cache/checkpoints/${name}"
    rsync_to "${opts[@]}" "${local_src}" "${remote_dst}"
    EVAL_NAMES["${name}"]=1
  done

  echo "==> 推送对应 cache/eval/（与上述 NAME 对齐；无本地目录则跳过）..."
  local variant model model_hash mid eval_rel eval_src eval_dst
  for name in "${!EVAL_NAMES[@]}"; do
    case "${name}" in
      */*/*/*|*".."*)
        echo "    跳过 eval（NAME 非法）: ${name}" >&2
        continue
        ;;
      */*/*)
        variant="${name%%/*}"
        model_hash="${name##*/}"
        mid="${name#*/}"
        model="${mid%/*}"
        ;;
      *)
        echo "    跳过 eval（NAME 须为 variant/model/hash）: ${name}" >&2
        continue
        ;;
    esac
    if [[ -z "${variant}" || -z "${model}" || -z "${model_hash}" || "${model}" == */* ]]; then
      echo "    跳过 eval（NAME 解析失败）: ${name}" >&2
      continue
    fi
    eval_rel="eval/${model}/${model_hash}"
    eval_src="${LOCAL_DIR}/cache/${eval_rel}"
    if [[ ! -d "${eval_src}" ]]; then
      echo "    跳过 ${eval_rel}/（本地不存在）"
      continue
    fi
    eval_dst="${REMOTE_SSH_TARGET}:${REMOTE_DIR}/cache/${eval_rel}/"
    echo "    → ${eval_rel}/"
    remote_ssh "mkdir -p ${REMOTE_DIR}/cache/${eval_rel}"
    rsync_to "${opts[@]}" "${eval_src}/" "${eval_dst}"
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

pull_logs() {
  local remote_src="${REMOTE_SSH_TARGET}:${REMOTE_DIR}/logs/"
  local local_dst="${LOCAL_DIR}/logs/"
  echo "==> 从 ${REMOTE_SSH_TARGET}（服务=${SERVER_NAME}）增量同步 logs/..."
  if ! remote_ssh "test -d ${REMOTE_DIR}/logs"; then
    echo "    （远端尚无 logs/，跳过）"
    return 0
  fi
  mkdir -p "${local_dst}"
  rsync_to "${RSYNC_OPTS[@]}" "${remote_src}" "${local_dst}"
}

pull_eval() {
  local remote_src="${REMOTE_SSH_TARGET}:${REMOTE_DIR}/cache/eval/"
  local local_dst="${LOCAL_DIR}/cache/eval/"
  echo "==> 从 ${REMOTE_SSH_TARGET}（服务=${SERVER_NAME}）增量同步 cache/eval/..."
  if ! remote_ssh "test -d ${REMOTE_DIR}/cache/eval"; then
    echo "    （远端尚无 cache/eval/，跳过）"
    return 0
  fi
  mkdir -p "${local_dst}"
  rsync_to "${RSYNC_OPTS[@]}" "${remote_src}" "${local_dst}"
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
  pull_logs
  pull_eval
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
    echo "请先指定 servers.csv 中的服务名，例如: $(basename "$0") ovan $1 ..." >&2
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
    checkpoint_specs=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --code-only) code_only=1; shift ;;
        --with-datasets) with_datasets=1; shift ;;
        --checksum) with_checksum=1; shift ;;
        --checkpoints)
          [[ $# -ge 3 ]] || { echo "缺少 --checkpoints NAME FILE" >&2; exit 1; }
          checkpoint_specs+=("$2" "$3")
          shift 3
          ;;
        --with-cache)
          echo "提示: --with-cache 已废弃；默认推 models/tokenizers（加 --with-datasets 才推数据集）" >&2
          shift
          ;;
        -h|--help) usage; exit 0 ;;
        *)
          echo "未知 push 选项: $1（可用 --code-only / --with-datasets / --checksum / --checkpoints）" >&2
          usage >&2
          exit 1
          ;;
      esac
    done
    req_updated=0
    if requirements_txt_updated; then
      req_updated=1
    fi
    push_code
    if [[ "${req_updated}" -eq 1 ]]; then
      if remote_has_venv; then
        install_remote_requirements
      else
        echo "==> requirements.txt 已更新，但远端无 .venv，跳过 pip install"
      fi
    fi
    sync_web push
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
      if [[ ${#checkpoint_specs[@]} -eq 0 ]]; then
        echo "==> 未推送：huggingface/（缓冲区）、checkpoints/、预处理、compile*；datasets/ 默认跳过（需加 --with-datasets）"
      else
        echo "==> 未推送：huggingface/（缓冲区）、未指定的 checkpoints/、预处理、compile*；datasets/ 默认跳过（需加 --with-datasets）"
      fi
    fi
    if [[ ${#checkpoint_specs[@]} -gt 0 ]]; then
      ckpt_args=()
      [[ "${with_checksum}" -eq 1 ]] && ckpt_args+=(--checksum)
      ckpt_args+=("${checkpoint_specs[@]}")
      push_checkpoint_files "${ckpt_args[@]}"
    fi
    echo "==> push 完成（${SERVER_NAME}）"
    ;;
  pull)
    shift
    pull_checkpoints "$@"
    sync_web pull
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
