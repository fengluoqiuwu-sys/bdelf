#!/usr/bin/env bash
# BDELF 100m full；经 slurm/sbatch-train.sh 提交或本地直接跑。
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

# eval 条数 = ELF 默认（1024 / 256）的 1/4；经 --set 覆盖共享 eval/default.yaml
exec "$PY" train.py \
  --model bdelf \
  --config 100m-full \
  --dataset owt \
  --preprocess default \
  --generate eval \
  --set eval.eval_sample_count=256 \
  --set eval.gen_eval_samples=64 \
  "$@"
