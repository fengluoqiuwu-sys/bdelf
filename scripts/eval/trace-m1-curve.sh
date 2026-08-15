#!/usr/bin/env bash
# M1 曲线：对稀疏 opt-step 快照只评 sc0.5（ACE=off）。
# save_step=128 → 文件名按 2048 micro 对齐：128 / 512 / 1024 / 2048 / 5000 opt。
# 传入 --run full/trace/<hash>。缺文件则跳过。
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

RUN=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      [[ $# -ge 2 ]] || { echo "--run 需要参数" >&2; exit 1; }
      RUN=$2
      shift 2
      ;;
    --run=*)
      RUN=${1#--run=}
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$RUN" ]]; then
  echo "需要 --run full/trace/<hash>" >&2
  exit 1
fi

CKPT_DIR="cache/checkpoints/${RUN}"
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "找不到 run 目录: $CKPT_DIR" >&2
  exit 1
fi

# accum=16 且每 128 opt 存盘：对齐到实际文件名（约 128/512/1k/2k/5k opt）。
STEPS=(0002048 0008192 0016384 0032768 0080000)

eval_one() {
  local ckpt=$1
  echo "=== eval sc0.5: $ckpt ==="
  "$PY" eval.py \
    --checkpoint "$ckpt" \
    --name sc0.5 \
    --generate eval \
    --set self_cond_cfg_scale=0.5 \
    --set ace=false \
    --num-samples 1024 \
    --micro-bs 16 \
    "${EXTRA[@]+"${EXTRA[@]}"}"
}

found=0
for step in "${STEPS[@]}"; do
  ckpt="${CKPT_DIR}/checkpoint_step_${step}.pt"
  if [[ -f "$ckpt" ]]; then
    found=1
    eval_one "$ckpt"
  else
    echo "跳过缺失: $ckpt"
  fi
done

latest="${CKPT_DIR}/checkpoint_latest.pt"
if [[ -f "$latest" ]]; then
  found=1
  eval_one "$latest"
fi

if [[ "$found" -eq 0 ]]; then
  echo "没有可评的 checkpoint: $CKPT_DIR" >&2
  exit 1
fi
