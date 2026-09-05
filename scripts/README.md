# scripts/

实验室通用辅助脚本。复制到新仓库后：

1. 把 `servers.csv.example` 复制为 `servers.csv`（gitignore），**改「工作目录」为本仓库远端路径**。
2. **`sync.sh` 必须对照新项目重写产物规则**（见该文件头 `TODO(PROJECT)`）。模板只推代码、只拉 `logs/`。
3. 作业入口（`scripts/train/*.sh`、`launch-*.sh` 等）在本仓库；common 接 `launch-train.sh` / `launch-eval.sh`，Slurm 经 `sar task add` 的 `-- CMD`。

| 文件 | 用途 |
|------|------|
| `servers_lib.sh` | 解析 `servers.csv` |
| `servers.csv.example` | 服务名（可 `别名:SSH主机`）/ 调度 / 工作目录 / 单任务卡数上限（「最大使用显卡数量」为遗留列） |
| `sync.sh` | 代码 push + `logs/` pull；根目录 `requirements.txt` 变更且远端有 `.venv` 则 pip install；默认再推 models/tokenizers |
| `sync_web.sh` | `sync.sh` 调用：按 hash 合并 `cache/monitor/charts.json` |
| `web.sh` | `local\|服务名` + `up\|down`：本机或 SSH 隧道打开 monitor |
| `job_log_dir.sh` | `logs/<服务名>/<时间戳>/` |
| `workspace_lock.py` | 遗留；AI 改代码不调用 |
| `agent_wakeup.py` | 延时打印 `AGENT_WAKEUP` |
| `repo_env.py` | `scripts/*.py` 切到仓库根 |
