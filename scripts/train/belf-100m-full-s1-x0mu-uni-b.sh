#!/usr/bin/env bash
# BELF 100m Stage1 主训：x0_source=mu + 入口 latent_t5/100m-uni-b（单向因果、readout=b、B=32）。
# 入口块长为 1；默认 latent_tune=frozen，亦可改 mid/full。
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/belf-100m-full-s1.sh" \
  --set model.x0_source=mu \
  --set model.latent_model=latent_t5 \
  --set model.tag=100m-uni-b \
  "$@"
