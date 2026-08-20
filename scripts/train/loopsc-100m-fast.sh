#!/usr/bin/env bash
# LoopSC 100m fast 冒烟（ELF 复制；训练展开闭环 SC）。本机调试用。
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
  --model loopsc \
  --config 100m-fast \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  "$@"
