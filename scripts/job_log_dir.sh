#!/usr/bin/env bash
# 作业日志目录：logs/{server-name}/<时间戳>/
# 由 slurm/sbatch-*.sh 与 scripts/launch-train.sh source；勿直接执行。
#
# 用法（先设 ROOT 或传入）::
#   source scripts/job_log_dir.sh
#   job_log_alloc <server-name> [root_dir]
#   → 设置 JOB_LOG_DIR、JOB_LOG_TS；创建目录
#   job_log_pending_dir <server-name> [root_dir]
#   → 打印 pending 目录路径并确保存在

job_log_alloc() {
  local server=$1
  local root=${2:-${ROOT:-}}
  if [[ -z "$server" ]]; then
    echo "job_log_alloc: 需要 server-name" >&2
    return 1
  fi
  if [[ -z "$root" ]]; then
    echo "job_log_alloc: 需要 root 目录（参数或 ROOT）" >&2
    return 1
  fi
  local base="$root/logs/$server"
  mkdir -p "$base"
  local ts
  ts="$(date +%Y%m%dT%H%M%S)"
  local dir="$base/$ts"
  local n=0
  while [[ -e "$dir" ]]; do
    n=$((n + 1))
    dir="$base/${ts}-${n}"
  done
  mkdir -p "$dir"
  JOB_LOG_DIR=$dir
  JOB_LOG_TS=$(basename "$dir")
}

job_log_pending_dir() {
  local server=$1
  local root=${2:-${ROOT:-}}
  if [[ -z "$server" || -z "$root" ]]; then
    echo "job_log_pending_dir: 需要 server-name 与 root" >&2
    return 1
  fi
  local d="$root/logs/$server/pending"
  mkdir -p "$d"
  printf '%s\n' "$d"
}
