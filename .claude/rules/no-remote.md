---
description: Claude 禁止使用远端服务器（硬约束）
---

# 禁止使用远端

Claude **不得**使用远端服务器，包括但不限于：

- `ssh ovan-server` / 任何远端 SSH
- Slurm（`sbatch` / `scancel` / `squeue` 等）
- `scripts/sync-ovan-server.sh` push / pull / pull-file
- 读写远端 `temp/` 或其它远端路径

训练、评测、generate、调试只在本机进行（见 rule「本机计算约束」）。若用户要求操作远端，应拒绝并说明 Claude 侧不允许使用远端。
