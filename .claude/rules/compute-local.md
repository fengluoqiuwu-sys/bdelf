# 本机计算约束

- **硬件**：1× RTX 5080（16GB）；**仅**本地调试 / generate，不入 `servers.csv`（≠ common 远端）。
- **Python**：见 rule「Python 虚拟环境」。
- **GPU 互斥**：同时只跑一个占 GPU 的进程；可停本会话拉起的作业，勿杀用户自启进程。
- 若启动占资源进程：在 `temp/agent/` 登记 `pid` / `cpus` / `gpus`，`scheduler: "local"`。
- **Claude 范围**：只做本机 generate / 只读检查；**禁止训练**（见 rule「禁止训练」）；**禁止自动训练**（见 rule「禁止自动训练」）；远端操作需用户明确授权。
- 推理命令见 skill `generate`。
