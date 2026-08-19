#!/usr/bin/env bash
# ResidW 100m fast 冒烟（ELF 复制；残差 SC-CFG 老师）。本机调试用。
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
  --model residw \
  --config 100m-fast \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  "$@"
