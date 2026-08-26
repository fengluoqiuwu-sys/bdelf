#!/usr/bin/env bash
# BELF/RELF 显存探针（idea 夹脚本）。common 须经 launch-train；勿直接占 GPU。
# 默认跑全部格子（每格子进程）；传 --models / --scenarios 只跑一格。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "找不到 Python（请创建 .venv 或激活环境）" >&2
  exit 1
fi

GPUS=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      [[ $# -ge 2 ]] || { echo "--gpus 需要参数" >&2; exit 1; }
      GPUS=$2
      shift 2
      ;;
    --gpus=*)
      GPUS=${1#--gpus=}
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:False}"
export CUDA_DISABLE_UNIFIED_MEMORY="${CUDA_DISABLE_UNIFIED_MEMORY:-1}"
export CUDA_MANAGED_FORCE_DEVICE_ALLOC="${CUDA_MANAGED_FORCE_DEVICE_ALLOC:-1}"

PROBE="$ROOT/temp/ideas/belf-relf/source/probe_vram.py"
if [[ ! -f "$PROBE" ]]; then
  echo "缺少探针脚本: $PROBE（先把 idea source rsync 到远端）" >&2
  exit 1
fi

exec "$PY" "$PROBE" "${EXTRA[@]}"
