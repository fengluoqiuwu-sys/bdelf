---
description: Claude 禁止使用远端服务器（硬约束）
---

# 禁止使用远端

Claude **不得**使用远端，包括但不限于：

- `ssh` 任意远端（含 ovan-server 与 `servers.csv` 中 common 机）
- Slurm（`sbatch` / `scancel` / `squeue` / `slurm/sbatch-train.sh` 等）
- `scripts/sync.sh <服务名>` / `scripts/ssh.sh <服务名>`
- `slurm/remote_status.sh`、`slurm/gpu_availability.py`、`slurm/tail_remote_logs.py`
- 读写远端任意路径（含远端 `temp/`）

需要远端权重或日志时：拒绝代操作，请用户用 Cursor 侧 skill（`sync` / `train-ops`）自行拉取后再在本机继续。
