#!/usr/bin/env bash
# I-1 M2 墙钟拆解（去噪 / ZSBD / native g / BGEE）。须经 launch-eval。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY=python
fi
exec "$PY" scripts/hybrid_readout_probe.py \
  --run full/elf/official-owt-b \
  --stage m2 \
  --num-samples 16 \
  --num-tokens 1024 \
  --micro-bs 16 \
  --repeats 5 \
  --seed 42 \
  --generate eval \
  --out temp/auto-research/hybrid-i1 \
  "$@"
