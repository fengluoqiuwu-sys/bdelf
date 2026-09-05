# 脚本约定

## 放置

| 位置 | 内容 |
|------|------|
| 仓库根 | 入口：`train.py`、`generate.py`、`eval.py`（Claude **不跑** `train.py`） |
| `scripts/` | 通用辅助（resolve、download、sync、`sync_web.sh`、`web.sh`、`workspace_lock.py`、`agent_wakeup.py`、`preprocess.py`、`export_latent_artifact.py`、`compute_latent_whiten.py`、`vram_probe.py`、`launch-train.sh`、`launch-eval.sh`、`job_log_dir.sh` 等） |
| `scripts/train/` | 训练启动 `.sh`（Claude **不执行**，除非用户明确授权远端手动提交） |
| `scripts/eval/` | 离线评测启动 `.sh`（远端经 `sar` / `launch-eval`） |
| `slurm/` | `gpu_availability.py` / `tail_remote_logs.py`（遗留）；Slurm 提交走 `sar`。仓库内 `sbatch-*.sh` / `prototype.slurm` 仅人用，AI 不得直接 sbatch |
| `logs/<服务名>/<时间戳>/` | 作业日志（gitignore；`sync pull` 拉取，push 不删远端）：`.out` / `.err` / `gpu-*.log` |

勿再为每个模型复制一份 `.slurm`。

## 执行目录

- **工作目录一律是仓库根**（含 `train.py` / `config/` / `.venv/`）。
- Claude 常用：`.venv/bin/python generate.py`、`.venv/bin/python scripts/resolve_checkpoint.py`、`.venv/bin/python scripts/foo.py`。
- 用户授权的远端：`ssh ovan-server '~/bin/sar …'`、`bash scripts/launch-train.sh <name> --server <服务名> --gpus …`、`bash scripts/launch-eval.sh <name> --server <服务名> --gpus … -- --run …`、`bash scripts/sync.sh <服务名> push|pull|…`、`ssh <服务名> 'cd ~/source/bdelf && …'`（服务名见 `scripts/servers.csv`；均可系统 SSH）。
- 禁止 `cd scripts && python foo.py`。
- `scripts/` 下 Python 用 `repo_env.ensure_repo_root()`；`slurm/*.py` 自行 `chdir` 到仓库根。

## SSH 边界

- `slurm/gpu_availability.py`、`slurm/tail_remote_logs.py` 等**不得在进程内发起 SSH**；在目标机直接跑。
- 本机查 ovan-server Slurm 队列/卡：`ssh ovan-server '~/bin/sar gpus'` / `'~/bin/sar status'`。本机包装查卡：`bash slurm/remote_status.sh`（即 `sar gpus`）。不要用它提交作业。
- common 远端：`ssh <名字>`；占 GPU 须 `scripts/launch-train.sh` / `scripts/launch-eval.sh`（`--server` + `--gpus`），写 `temp/agent` 与 `logs/<名字>/<时间戳>/`；无 `remote_status.sh`（见 rule「远端 common 计算约束」）。
- 本机读远端日志：push（如需）→ `ssh <服务名> 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py …'`（可加 `--server <服务名>`）。
