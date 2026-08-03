---
description: 本机 RTX 5080 计算硬约束（Claude：仅推理）
---

# 本机计算约束

- **硬件**：1× RTX 5080（16GB）。
- **Python**：见 rule「Python 虚拟环境」。
- **GPU 互斥**：同时只跑一个占 GPU 的进程；可停本会话拉起的作业，勿杀用户自启进程。
- **Claude 范围**：只做本机 generate / 只读检查；**禁止训练**（见 rule「禁止训练」）；**禁止远端**（见 rule「禁止使用远端」）。
- 推理命令见 skill `generate`。
