#!/usr/bin/env bash
# DenoiserChart 100m full（ELF 复制：隔离 W_t + 中段冻 unembed CE）。
# 默认两者格（chart_warp=true, chart_weight=0.1, freeze_unembed=true）。
# 续训 ELF-100m 时加：--set extra.init_ckpt=cache/checkpoints/full/elf/<hash>/checkpoint_latest.pt
# 消融：
#   仅 CE    --set model.chart_warp=false
#   仅 W_t   --set model.chart_weight=0
#   泄漏     --set model.chart_leak_decode=true
# 经 slurm/sbatch-train.sh 或 launch-train.sh 提交；本脚本不直接占远端 GPU。
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
  --model denoiser_chart \
  --config 100m-full \
  --dataset owt \
  --preprocess elf \
  --generate eval \
  "$@"
