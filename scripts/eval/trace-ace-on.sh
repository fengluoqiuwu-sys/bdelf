#!/usr/bin/env bash
# TrACE ACE=on 诊断：sc0.5 / sc1，与同 step 的 ACE=off 对照。
# 须经 launch-eval / sbatch-eval。--checkpoint 经包装器 -- 传入。
# 两组共用一次 Alg.1 估 d，故放同一作业、一张卡。
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
  --table trace-ace-on \
  --micro-bs 16 \
  "$@"
