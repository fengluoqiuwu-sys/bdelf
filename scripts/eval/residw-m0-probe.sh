#!/usr/bin/env bash
# ResidW M0：加载 ELF 权重，扫 sc∈{0.5,1,2,3} × ACE on/off（8 组）。
# 实现在 temp/ideas/resid-w/source/m0_sc_ace_eval.sh。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PROBE="$ROOT/temp/ideas/resid-w/source/m0_sc_ace_eval.sh"
if [[ ! -f "$PROBE" ]]; then
  echo "缺少探针脚本: $PROBE（temp/ 不同步，须拷到目标机）" >&2
  exit 1
fi

exec bash "$PROBE" "$@"
