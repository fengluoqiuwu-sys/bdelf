# 禁止自动训练

Claude **不得**使用自动训练闭环，包括但不限于：

- 调用 `auto-train` skill 或相关 subagent
- 自主创建训练实验并提交到远端
- 自动化的训练-评测-调参循环

**允许**的远端工作：

- 手动提交训练作业（`bash slurm/sbatch-train.sh` / `bash scripts/launch-train.sh`）
- 查看远端状态（`bash slurm/remote_status.sh` / 扫 `temp/agent/active/`）
- 同步代码与权重（`bash scripts/sync.sh <服务名> push|pull`）
- 读取远端日志（`slurm/tail_remote_logs.py`）
- 手动评测（`bash slurm/sbatch-eval.sh` / `bash scripts/launch-eval.sh`）

若用户要求自动训练闭环：拒绝，并说明应使用 **Cursor** 侧 skill `auto-train`。
