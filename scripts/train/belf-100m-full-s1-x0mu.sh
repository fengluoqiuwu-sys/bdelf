#!/usr/bin/env bash
# BELF 100m Stage1 主训：显式 x0_source=mu（现为默认）。其余同 belf-100m-full-s1.sh。
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/belf-100m-full-s1.sh" \
  --set model.x0_source=mu \
  "$@"
