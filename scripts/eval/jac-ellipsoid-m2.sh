#!/usr/bin/env bash
# JacEllipsoid M2 TriFluency：config/eval/tables/jac-ellipsoid-m2.yaml
# 须经包装器或本机传入 --run full/jac_ellipsoid/<hash>。
# 远端默认 micro-bs=16；本机可 --micro-bs 8。四对照各评一次再 score_main。
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
  --table jac-ellipsoid-m2 \
  --micro-bs 16 \
  "$@"
