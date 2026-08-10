#!/usr/bin/env bash
# ODAR TriFluency 扫参：config/eval/tables/odar-sc-ace.yaml
# 须经包装器或本机传入 --run full/odar/<hash>（及可选 --checkpoint）。
# 远端默认 micro-bs=16（显存高于本机 5080）；本机可 --micro-bs 8 覆盖。
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
  --table odar-sc-ace \
  --micro-bs 16 \
  "$@"
