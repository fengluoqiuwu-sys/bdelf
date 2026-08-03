#!/usr/bin/env bash
# elf-cfg 续训专用：锁定 fingerprint → full/elf/19de90b094488c46
# 勿随意加 --set；勿在未核对 temp/auto-research/elf-cfg/README.md「续训指纹冻结」
# 的情况下改 config/train/model/elf/full.yaml 的 batch/optimizer。
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

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

exec "$PY" train.py \
  --model elf \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  --set eval.gen_eval_samples=32
