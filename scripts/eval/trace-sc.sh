#!/usr/bin/env bash
# TrACE 主表：sc0.5 / sc1 / sc3，ACE=off。须经 launch-eval / sbatch-eval。
# 传入 --run full/trace/<hash> 或 --checkpoint path。
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

exec "$PY" eval.py \
  --table trace-sc \
  --micro-bs 16 \
  "$@"
