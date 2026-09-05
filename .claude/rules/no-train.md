# 禁止训练（只允许推理）

Claude **不得**启动或续跑训练，包括但不限于：

- 运行 `train.py`（含 fast / full）
- 运行 `scripts/train/*.sh`（除非用户明确授权远端手动提交，且走包装器 / `sar`，禁止直接 sbatch）
- 修改训练超参并开训、冒烟训、auto-train 闭环

**允许**的本机工作：

- `generate.py` 推理 / 采样 / 续写（见 skill `generate`）
- 只读查看本机已有 checkpoint、`train_log.csv` / `eval_log.csv`、配置 YAML
- `scripts/resolve_checkpoint.py`（仅定位本机已有 run 的 hash 路径）
- 代码阅读与不涉及开训的编辑（遵守项目规范）

若用户要求本机开训或自动训练：拒绝，并说明应使用 **Cursor** 侧 skill（`train` / `train-ops` / `auto-train`）。  
用户明确授权的远端手动提交见 rule「禁止自动训练」与远端计算约束（Slurm 走 `sar`，禁止直接 `sbatch`）。
