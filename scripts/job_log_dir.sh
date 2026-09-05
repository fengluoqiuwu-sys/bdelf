#!/usr/bin/env bash
# 作业日志目录：logs/{服务名}/<时间戳>/（csv 冒号左侧，不是 SSH 主机）
# 由 slurm/sbatch-*.sh 与 scripts/launch-train.sh / launch-eval.sh source；勿直接执行。
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

# AutoDL 等：把 TMPDIR 指到数据盘，避免 overlay 系统盘 /tmp 被 ckpt stage / compile 写满。
# 不改 checkpoint 训练产物路径。已设置的 TMPDIR / SLURM_TMPDIR 不覆盖。
bdelf_export_scratch_tmpdir() {
  if [[ -n "${SLURM_TMPDIR:-}" || -n "${TMPDIR:-}" ]]; then
    return 0
  fi
  if [[ -d /root/autodl-tmp && -w /root/autodl-tmp ]]; then
    mkdir -p /root/autodl-tmp/bdelf-tmp
    export TMPDIR=/root/autodl-tmp/bdelf-tmp
  fi
}

# 只删本 job 的 resume / compile scratch；禁止通配 bdelf-*。
bdelf_rm_job_scratch() {
  local job="${1:-}"
  if [[ ! "$job" =~ ^[A-Za-z0-9._-]+$ ]]; then
    return 0
  fi
  local scratch="${TMPDIR:-/tmp}"
  scratch="${scratch%/}"
  local d
  for d in \
    "${scratch}/bdelf-resume-${job}" \
    "${scratch}/bdelf-compile-${job}" \
    "${scratch}/bdelf-compile-pid${job}" \
    "/tmp/bdelf-resume-${job}" \
    "/tmp/bdelf-compile-${job}" \
    "/tmp/bdelf-compile-pid${job}"
  do
    if [[ -d "$d" ]]; then
      rm -rf -- "$d"
    fi
  done
}
