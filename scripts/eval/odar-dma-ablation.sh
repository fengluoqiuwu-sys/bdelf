#!/usr/bin/env bash
# ODAR 采样 DMA 消融：config/eval/tables/odar-dma-ablation.yaml
# 须经包装器或本机传入 --run / --checkpoint。
# 远端默认 micro-bs=16；本机可 --micro-bs 8 覆盖。
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
  --table odar-dma-ablation \
  --micro-bs 16 \
  "$@"
