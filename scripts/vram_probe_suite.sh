#!/usr/bin/env bash
# 在单卡上依次对多个模型跑 scripts/vram_probe.py。
# 跳过 bdelf（模型仍在修）与 cola 系列（缺 VAE / 暂不测）。
# 由 slurm/sbatch-vram-probe.sh --suite 提交；也可在已分配 GPU 的作业内直接跑。
# 结果供 AI 填入本地 temp/vram-probe/alloc.md（model×batch→GiB）；不按 global_bs 过滤。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  echo "找不到 .venv/bin/python" >&2
  exit 1
fi

# 元数据（不用于过滤）；测量时探针固定 accum=1
WORLD_SIZE="${BDELF_VRAM_PROBE_WORLD_SIZE:-4}"
BATCHES="${BDELF_VRAM_PROBE_BATCHES:-1,2,4,8,16,24,32,48,64,96,128}"

# model|preprocess
MODELS=(
  "ar|default"
  "ar1_5|default"
  "ar2|default"
  "bd3lm|default"
  "elf|elf"
)

echo "=== vram_probe_suite start: $(date -Is) host=$(hostname) ==="
echo "world_size(metadata)=$WORLD_SIZE batches=$BATCHES"
echo "skip: bdelf (模型仍在修); cola, cola_vae (cola 系列暂不测)"

overall=0
for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"
  pre="${entry##*|}"
  echo
  echo "################################################################"
  echo "# MODEL=$model preprocess=$pre"
  echo "################################################################"
  set +e
  "$PY" scripts/vram_probe.py \
    --model "$model" \
    --config 100m-full \
    --dataset owt \
    --preprocess "$pre" \
    --generate eval \
    --batches "$BATCHES" \
    --world-size "$WORLD_SIZE"
  rc=$?
  set -e
  echo "# MODEL=$model exit=$rc"
  if [[ $rc -ne 0 && $rc -ne 2 ]]; then
    # 2 = 某档 OOM 停（仍有部分结果）；其它非 0 视为失败
    overall=$rc
  fi
done

echo
echo "=== vram_probe_suite end: $(date -Is) overall_exit=$overall ==="
exit "$overall"
