#!/usr/bin/env bash
# 兼容短名：RELF 100m full = Stage1 主训（45B@512）。Stage2 用 relf-100m-full-s2.sh。
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/relf-100m-full-s1.sh" "$@"
