#!/usr/bin/env bash
# Cola Stage-2 100m full（含训练期 gen-eval）；经 slurm/sbatch-train.sh 提交或本地直接跑。
# 需已有 Stage-1 VAE checkpoint：COLA_VAE_CHECKPOINT 或 --set 指向 artifacts/cola_vae/<tag> 或 COLA_VAE_CHECKPOINT。
# 两阶段连跑用 scripts/train/cola-seq-100m-full.sh。
# Stage-2：峰值/衰减按 100m·45.2B 缩放的官方 AdamW；warmup 与 elf-cfg 相同：
#   - 峰值 lr=4e-4：官方 1.5e-4 是 ~2B、宽 2048、~721B token 的；100m 宽 768
#     按 μP（×2048/768）≈4.0e-4，与 GPT-2 124m 的 6e-4 再按 batch 256/512 开方 ≈4.2e-4 一致
#     原样 1.5e-4 + 余弦到 1e-5 在 45.2B 上平均 LR≈1e-4，对 AdaLN-Zero DiT 偏保守
#   - warmup_ratio=0.1（与 elf-cfg 一致；本次 ~17243 / 172425 opt-step）
#   - min_lr_ratio=1e-5/4e-4=0.025，余弦在本次最后一步落到 1e-5
#   - use_muon=false（官方论文 AdamW）
#   - 在线 gen-eval：16 步 ODE、T=0 argmax、CFG=7
# Stage-1 VAE 仍用默认 50B / Muon 0.002 日程，勿把下列 --set 传给 cola_vae。
# 微批 8、全局批 256（4 卡 accum 仍为 8）。
# 预处理仍为 default（GPT-2），不用 elf（T5）。
#   - gen_eval_samples=32（控在线评测开销）
# fingerprint：resolve_checkpoint.py … 下列 --set → full/cola/9063fab557ec5667
# Rank0 gen-eval 时 peer 会卡在短 all_reduce，拉长 NCCL 超时避免误杀。
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

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

exec "$PY" train.py \
  --model cola \
  --config 100m-full \
  --dataset owt \
  --preprocess default \
  --generate eval \
  --set eval.gen_eval_samples=32 \
  --set schedule.use_muon=false \
  --set schedule.warmup_ratio=0.1 \
  --set schedule.min_lr_ratio=0.025 \
  --set schedule.target_tokens=45200000000 \
  "$@"
