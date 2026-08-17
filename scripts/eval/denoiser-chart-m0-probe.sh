#!/usr/bin/env bash
# DenoiserChart M0 中段探针（冻 unembed CE / ZSBD / p10 margin）。
# 实现在 temp/ideas/denoiser-chart/source/m0_probe.py；本题结果写 result/m0_probe_adam/。
# 单格：--cell NAME --hash HEX（训完一格接一格）；须经 launch-eval.sh / sbatch-eval.sh。
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

PROBE="$ROOT/temp/ideas/denoiser-chart/source/m0_probe.py"
if [[ ! -f "$PROBE" ]]; then
  echo "缺少探针脚本: $PROBE（temp/ 不同步，须拷到目标机）" >&2
  exit 1
fi

exec "$PY" "$PROBE" "$@"
