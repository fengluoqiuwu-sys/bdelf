#!/usr/bin/env bash
# ODAR 5428154e7c817e11：补齐 step 360k/400k 的 ACE=on × SC∈{0.5,1,2,3}。
# 同一作业内顺序跑两步（先估 ACE 方向再评测）；同参已有 summary.json 会跳过。
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

CKPT_DIR="cache/checkpoints/full/odar/5428154e7c817e11"
TABLE="${TABLE:-odar-ace-on}"
MICRO_BS="${MICRO_BS:-16}"

for step_pt in \
  checkpoint_step_0360000.pt \
  checkpoint_step_0400000.pt
do
  ckpt="${CKPT_DIR}/${step_pt}"
  if [[ ! -f "$ckpt" ]]; then
    echo "缺少 checkpoint: $ckpt" >&2
    exit 1
  fi
  echo "=== eval ACE-on: $ckpt (table=$TABLE micro_bs=$MICRO_BS) ==="
  "$PY" eval.py \
    --checkpoint "$ckpt" \
    --table "$TABLE" \
    --micro-bs "$MICRO_BS" \
    "$@"
done
