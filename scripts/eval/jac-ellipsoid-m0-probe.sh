#!/usr/bin/env bash
# JacEllipsoid M0 探针：覆盖率 / unembed 几何对齐。
# 实现在 temp/ideas/jac-ellipsoid/source/m0_probe.py；结果写 result/m0/。
# 须经 launch-eval.sh / sbatch-eval.sh 若占 GPU；本机 CPU 也可直接跑 coverage。
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

PROBE="$ROOT/temp/ideas/jac-ellipsoid/source/m0_probe.py"
if [[ ! -f "$PROBE" ]]; then
  echo "缺少探针脚本: $PROBE（temp/ 不同步，须拷到目标机）" >&2
  exit 1
fi

exec "$PY" "$PROBE" "$@"
