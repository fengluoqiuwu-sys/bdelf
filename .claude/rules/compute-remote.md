---
description: 远端（ovan-server / Slurm）计算与自动运行硬约束
---

# 远端计算约束

## 环境

- **主机**：`ovan-server`，最多 4× RTX 4090。通过 `ssh ovan-server` 操作。
- **用途**：只跑 **full**（或同等完整训练配置）；禁止 ultra / preprocess 作业。
- **Python**：由 `slurm/full/*.slurm` 内 `source .venv/bin/activate`；不要在远端交互式乱跑训练/评测。

## 只读项目树 + 仅 Slurm

- 远端项目代码/配置/脚本：**只读**。禁止用 ssh 直接改 `~/source/bdelf` 下除 `temp/` 外的任何文件。
- 改动一律在本地完成 → `bash sync-ovan-server.sh push` → 再 `sbatch`。
- 远程任务只能通过 **Slurm** 提交；脚本放在仓库 `slurm/` 下并随代码同步。
- 禁止提交 `slurm/ultra/`、`slurm/preprocess-*` 等非 full 作业；只用现有 `slurm/full/` 或等同 full 配置。
- `slurm/logs/` 等仅允许只读查看。
- **唯一可写远端路径**：`~/source/bdelf/temp/`（agent 任务登记）。`temp/` 在 push/pull 中均被屏蔽。

## AI 任务互斥

- 只约束 **本 agent 在 `temp/` 中登记过的任务**；同账号他人手动任务不擅自取消。
- 同时只跑一个 AI 登记任务。提交前检查登记；可自行 `scancel` 自己拉起的上一个任务，或确认其已结束后再提交。
- 拉起/取消后必须更新 `temp/` 登记文件。

## 禁止在远端做的事

- 看训练效果 / generate / eval / 调试推理：**不要在远端跑**；拉回本机测（见 rule「本机计算约束」）。
- 提交与登记流程见 skill `train-ops`；同步见 skill `sync-ovan-server`。
