#!/usr/bin/env bash
# =============================================================================
# 远端 / 本机 eval 启动脚本模板（复制为 scripts/eval/<name>.sh 后修改）
# =============================================================================
# - 工作目录须为仓库根
# - 只写「怎么评」：eval.py 参数（--table / --micro-bs 等）
# - --run / --checkpoint 等随作业变化的参数经包装器 ``--`` 传入
# - 不要写 #SBATCH；资源由 slurm/eval.slurm + sbatch-eval.sh 或 launch-eval.sh 负责
#
# 本机::
#   bash scripts/eval/<name>.sh --run full/<model>/<hash>
# Slurm::
#   bash slurm/sbatch-eval.sh <name> -- --run full/<model>/<hash>
# common::
#   bash scripts/launch-eval.sh <name> --server <服务名> --gpus 0,1,2,3 -- \
#     --run full/<model>/<hash>
# =============================================================================
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

exec "$PY" eval.py \
  --table <table-name> \
  --micro-bs 8 \
  "$@"
