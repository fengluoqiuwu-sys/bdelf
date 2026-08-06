#!/usr/bin/env bash
# Cola Stage-2 100m full；需已有 Stage-1 VAE checkpoint。
# 可用 COLA_VAE_CHECKPOINT 或 train --set 指向 full/cola_vae/<hash>。
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
  --model cola \
  --config 100m-full \
  --dataset owt \
  --preprocess default \
  --generate eval \
  "$@"
