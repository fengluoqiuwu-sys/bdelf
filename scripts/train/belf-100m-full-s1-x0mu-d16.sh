#!/usr/bin/env bash
# BELF 100m Stage1 主训：x0_source=mu + 入口 latent 100m-b32-d16（encoder 块长 16）。
# 块因果入口只允许 frozen（mid/full 会启动拒绝）。
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/belf-100m-full-s1.sh" \
  --set model.x0_source=mu \
  --set model.tag=100m-b32-d16 \
  --set model.latent_tune=frozen \
  "$@"
