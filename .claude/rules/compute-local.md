---
description: 本机（RTX 5080）计算与自动运行硬约束
---

# 本机计算约束

## 环境

- **硬件**：1× RTX 5080（16GB）。
- **用途**：`fast` 训练与任意快速调试。
- **Python**：一律用仓库 `.venv`（见 rule「Python 虚拟环境」）。

## GPU 互斥

- 不要让两个占用 GPU 的进程同时跑；需要时串行。
- 允许结束 **本会话拉起** 的旧 GPU 进程；不要随意杀掉用户自己启动的进程。

## 测试与评测

- generate / eval / 调试推理：只在本机跑。
- 本机调度与训练命令见 skill `train`；生成见 skill `generate`。
