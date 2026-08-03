---
description: Claude 禁止训练，只允许本机推理/评测
---

# 禁止训练（只允许推理）

Claude **不得**启动或续跑训练，包括但不限于：

- 运行 `train.py`（含 fast / full）
- 运行 `scripts/train/*.sh` 或 `slurm/sbatch-train.sh`
- 修改训练超参并开训、冒烟训、auto-train 闭环

**允许**的本机工作：

- `generate.py` 推理 / 采样 / 续写（见 skill `generate`）
- 只读查看本机已有 checkpoint、`train_log.csv` / `eval_log.csv`、配置 YAML
- `scripts/resolve_checkpoint.py`（仅定位本机已有 run 的 hash 路径）
- 代码阅读与不涉及开训的编辑（遵守项目规范）

若用户要求训练或远端作业：拒绝，并说明应使用 **Cursor** 侧 skill（`train` / `train-ops` / `auto-train`）。
