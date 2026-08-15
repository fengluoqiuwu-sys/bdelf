#!/usr/bin/env bash
# I-1 Hybrid bulk readout：M0 终态探针 + M1 结构/随机/类型门（4 卡分片）。
# 须经 launch-eval / sbatch-eval；勿直接占 GPU。
# common::
#   bash scripts/launch-eval.sh hybrid-m0 --server train-server-2 --gpus 0,1,2,3 \
#     --holder auto-train:hybrid-i1
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

exec "$PY" scripts/hybrid_readout_probe.py \
  --run full/elf/official-owt-b \
  --stage m0m1 \
  --num-samples 1024 \
  --num-tokens 1024 \
  --micro-bs 16 \
  --seed 42 \
  --generate eval \
  --out temp/auto-research/hybrid-i1 \
  "$@"
