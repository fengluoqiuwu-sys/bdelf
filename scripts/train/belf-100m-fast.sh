#!/usr/bin/env bash
# BELF 100m fast 冒烟。本机调试用；--preprocess owt-seg512（与主训同切 512）。
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

exec "$PY" train.py \
  --model belf \
  --config 100m-fast \
  --dataset owt \
  --preprocess owt-seg512 \
  --generate eval \
  "$@"
