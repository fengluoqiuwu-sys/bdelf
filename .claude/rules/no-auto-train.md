# 禁止自动训练

Claude **不得**使用自动训练闭环，包括但不限于：

- 调用 `auto-train` skill 或相关 subagent
- 自主创建训练实验并提交到远端
- 自动化的训练-评测-调参循环

**允许**的远端工作（须用户明确授权）：

- 手动提交作业：Slurm 用 `ssh ovan-server '~/bin/sar task add …'`（**禁止**直接 `sbatch` / `slurm/sbatch-*.sh` / 任何 `sar project`）；common 用 `bash scripts/launch-train.sh`
- 查看远端状态：`ssh ovan-server '~/bin/sar gpus'` / `'~/bin/sar status'`（本机包装 `bash slurm/remote_status.sh` 即 `sar gpus`）；common 扫该机 `temp/agent/active/`
- 同步代码与权重（`bash scripts/sync.sh <服务名> push|pull`）
- 读取远端日志（`sar task show ID` 或 `slurm/tail_remote_logs.py`）
- 手动评测：Slurm 经 `sar task add`；common 经 `bash scripts/launch-eval.sh`

若用户要求自动训练闭环：拒绝，并说明应使用 **Cursor** 侧 skill `auto-train`。
