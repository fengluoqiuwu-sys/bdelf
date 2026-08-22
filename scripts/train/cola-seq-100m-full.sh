#!/usr/bin/env bash
# Cola 两阶段顺序 full：先 cola_vae（Stage-1），再 cola（Stage-2，加载 artifacts VAE（或 COLA_VAE_CHECKPOINT））。
# 经 slurm/sbatch-train.sh 或 launch-train.sh 提交；本脚本不直接占远端 GPU。
# 同一作业内连续跑两段；Slurm 默认 --time=2-00:00:00 可能不够，提交时可追加例如
#   bash slurm/sbatch-train.sh cola-seq-100m-full --time=4-00:00:00
# 已有 Stage-1 权重时：export COLA_VAE_CHECKPOINT=.../checkpoint_latest.pt 可跳过 VAE。
# "$@" 原样传给两段（含 --gpus / --set）。VAE 已跑满则 train.py 会立刻退出再进 Stage-2。
# Stage-2 日程在 cola-100m-full.sh（冻 VAE、AdamW 4e-4）；Stage-1 VAE 仍用 cola-vae 默认 50B 日程。
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

# 抽出 --set，供 resolve_checkpoint 与 Stage-1 入参对齐（--gpus 不进哈希）。
SET_ARGS=()
_args=("$@")
_i=0
while [[ $_i -lt ${#_args[@]} ]]; do
  _a="${_args[$_i]}"
  if [[ "$_a" == --set ]]; then
    _i=$((_i + 1))
    if [[ $_i -ge ${#_args[@]} ]]; then
      echo "--set 需要 SECTION.KEY=VALUE" >&2
      exit 1
    fi
    SET_ARGS+=(--set "${_args[$_i]}")
  elif [[ "$_a" == --set=* ]]; then
    SET_ARGS+=("$_a")
  fi
  _i=$((_i + 1))
done

resolve_vae_ckpt() {
  "$PY" scripts/resolve_checkpoint.py \
    --model cola_vae \
    --config 100m-full \
    --dataset owt \
    --preprocess default \
    --generate eval \
    "${SET_ARGS[@]}" \
    --json \
  | "$PY" -c "
import json, sys
from pathlib import Path
d = json.load(sys.stdin)
p = Path(d['checkpoint_latest'])
if not p.is_file():
    raise SystemExit(f'Stage-1 checkpoint 不存在: {p}')
print(p)
"
}

if [[ -n "${COLA_VAE_CHECKPOINT:-}" ]]; then
  if [[ ! -f "$COLA_VAE_CHECKPOINT" ]]; then
    echo "COLA_VAE_CHECKPOINT 不是文件: $COLA_VAE_CHECKPOINT" >&2
    exit 1
  fi
  echo "=== cola-seq: 跳过 Stage-1（已设置 COLA_VAE_CHECKPOINT=$COLA_VAE_CHECKPOINT）==="
else
  echo "=== cola-seq Stage-1 cola_vae 100m-full $(date -Is) ==="
  bash "$ROOT/scripts/train/cola-vae-100m-full.sh" "$@"
  export COLA_VAE_CHECKPOINT="$(resolve_vae_ckpt)"
  echo "=== cola-seq Stage-1 完成: COLA_VAE_CHECKPOINT=$COLA_VAE_CHECKPOINT ==="
fi

echo "=== cola-seq Stage-2 cola 100m-full $(date -Is) ==="
echo "COLA_VAE_CHECKPOINT=$COLA_VAE_CHECKPOINT"
bash "$ROOT/scripts/train/cola-100m-full.sh" "$@"
echo "=== cola-seq 全部完成 $(date -Is) ==="
