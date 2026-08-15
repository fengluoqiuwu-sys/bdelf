#!/usr/bin/env bash
# I-3 M5 一次性包装（不进 git，用完删除）。表在 temp/。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY=python
fi
exec "$PY" eval.py \
  --table temp/research-scout/2026-08-13-elf-arch/m5_nfe.yaml \
  --micro-bs 16 \
  "$@"
