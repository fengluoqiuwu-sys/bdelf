# 本机计算约束

- **硬件**：1× RTX 5080（16GB）；用途：**仅**本地调试 / generate，**不是** `servers.csv` 中的计算服务。
- **调度**：本机 ≠ Slurm、≠ common 远端；不入 `servers.csv`。正式 full 训练在远端（Slurm 或 common），交给 **Cursor**。
- **Python**：见 rule「Python 虚拟环境」。
- **GPU 互斥**：同时只跑一个占 GPU 的进程；可停本会话拉起的作业，勿杀用户自启进程。
- **分支**：改代码 / generate 均在 **`master`** 上进行（`git branch --show-current` 须为 `master`；被切走则切回）。**禁止**为实验另开实现分支；思路隔离用 `temp/`。**不**抢工作区锁。
- **改动兼容性**：默认只做向前兼容、且不影响其他模型训练/推理的改动；破坏性 / 修实现错误会改旧数值等须向用户二次确认（见 rule「Checkpoint 路径与配置哈希」）。
- **Claude 范围**：只做本机 generate / 只读检查；**禁止训练**（见 rule「禁止训练」）；**禁止自动训练**（见 rule「禁止自动训练」）；远端操作需用户明确授权。
- **推理/评测**：本机可直接 `generate.py`；本机 `eval.py` 仅在用户明确要求且 GPU 空闲时。远端 TriFluency eval：Slurm 经 `sar`，common 经 `scripts/launch-eval.sh`，默认卡数与 csv「单个ai任务最大使用显卡数量」相同；结果经 `sync pull`（含 `cache/eval/`）拉回查看。`generate.py` 交互式续写仍只在本机。
- 推理命令见 skill `generate`。

## 本机作业登记

本机启动的占 CPU/GPU **调试**作业，在本地 `temp/agent/` 登记（`scheduler: "local"`）。**不要**在本机登记 Slurm 远端作业（common 的登记写在那台机的 `temp/agent/`）。

- 路径：`temp/agent/active/<id>.json`（未结束）+ `launched/<id>.json`（历史）；`<id>` 建议 `pid<PID>`。
- **必填**：`pid`、`cpus`、`gpus`、`holder`、`started_at`、`state`、`scheduler: "local"`；建议含 `job_name` / `cmdline` / `script`。
- 启动前：扫 `active/`，对已死 PID 清掉 active；再确认本机 GPU 空闲。
- 结束后：更新 launched、**删除**对应 active；勿动他人 `holder` 的登记。
