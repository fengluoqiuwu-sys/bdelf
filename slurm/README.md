# slurm/

实验室通用状态/日志脚本（ovan-server）。**AI 提交 Slurm 作业必须用 `slurm-auto-run`（`~/bin/sar`）**，不要用本目录的 `sbatch-*.sh` 当提交闸。`remote_status.sh` 只是本机包装 `sar gpus`。

详见 skill `slurm-auto-run`。

复制到新仓库后：

1. Slurm 占卡：`ssh ovan-server '~/bin/sar …'`（daemon / task / gpus；`project` 仅人执行）。
2. **`sbatch-*.sh`、`prototype.slurm` 等作业模板未收录**；即使项目里有，AI 也不得直接 `sbatch`。
3. `gpu_availability.py` 视为遗留；查卡用 `sar gpus`，本机也可用 `bash slurm/remote_status.sh`。

| 文件 | 用途 |
|------|------|
| `remote_status.sh` | 本机包装：`ssh ovan-server '~/bin/sar gpus'`（可 `--json` / `--cluster`） |
| `gpu_availability.py` | 遗留；在目标机跑、不发起 SSH |
| `tail_remote_logs.py` | 读 `logs/<服务>/<时间戳>/`；不发起 SSH |
