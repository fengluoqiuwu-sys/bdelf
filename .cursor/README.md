# `.cursor` 研究模板

从具体实验仓库抽出来的、**与项目实现无关**的 Cursor 规则与 skill。实验室 GPU / 单任务卡数约定保留；不含自动训练闭环。

复制到新仓库的 `.cursor/` 后，按该仓库补项目专用规则（配置、checkpoint、训练入口等）。

## 规则（`rules/`，均 alwaysApply）

| 文件 | 内容 |
|------|------|
| `compute-local.mdc` | 本机 5080 只调试；GPU 互斥；改代码须 `master`（不抢锁）；本机 `temp/agent` |
| `compute-remote-slurm.mdc` | ovan-server；Slurm 作业必须经 `sar`；禁止直接 sbatch；无合计卡数限额 |
| `compute-remote-common.mdc` | 非 Slurm 远端；占卡须包装器 + `gpu_ids`；只卡单次张数 |
| `python-venv.mdc` | 一律仓库根 `.venv` |
| `temp-layout.mdc` | `ideas` / `papers` / `research-scout` / 本机与 common `agent` |
| `scripts.mdc` | 工作目录=仓库根；SSH 边界 |
| `project-conventions.mdc` | 通用目录、中文注释、skill 索引 |
| `git-commit.mdc` | 人说才提交；禁机密；多逻辑先问再拆 |
| `subagent-model.mdc` | 三类模型须用户指定、写入每层 Task prompt 并原样向内传；禁默认与 fast |

未纳入：自动训练、具体模型/配置/checkpoint 哈希、YAML 行内注释等同构。

## Skills（`skills/`）

| Skill | 用途 |
|-------|------|
| `research-scout` | 文献探索 → `temp/research-scout/<run>/ideas.md` 与 `ideas/I-{n}/`（失败为 `D-{n}/`；单条深化见同目录 `idea-explore.md`） |
| `idea-kickoff` | 人确认可行后拷贝到 `temp/ideas/<name>/`；开题：综述→规格→报告 |
| `idea-experiment` | 实验规范：`source/` 本题代码、`result/` 数据与结果；不要求 AI 跑实验 |
| `git-commit` | 人说才提交；类型 + 中文说明；多逻辑先提案再拆 |
| `paper-ingest` | 下载一篇论文并写 `INDEX.md`（供 scout / idea-explore / idea-kickoff subagent） |
| `compute-ops` | 本机/common 占卡作业与 `temp/agent` 登记；Slurm 见 `slurm-auto-run` |
| `slurm-auto-run` | ovan-server `~/bin/sar` 提交/取消/看卡；禁止直接 sbatch；禁止 `sar project` |
| `sync` | `scripts/sync.sh` 推代码 / 拉 `logs/`；产物规则须按新项目重写 |

`train-ops` 在本仓库仍为训练/评测/VRAM 专用 skill；Slurm 提交走 `slurm-auto-run`。

## 脚本（仓库根 `scripts/` / `slurm/`）

实验室通用工具。详见各目录 README。

| 路径 | 用途 |
|------|------|
| `scripts/sync.sh` | 代码 push + `logs/` pull；根目录 `requirements.txt` 变更且远端有 `.venv` 则 pip install；产物路径见文件头 |
| `scripts/servers.csv.example` | 复制为 `servers.csv` 并改「工作目录」；「名字」可 `别名:SSH主机` |
| `scripts/servers_lib.sh` | 解析 csv |
| `scripts/job_log_dir.sh` | `logs/<服务>/<时间戳>/` |
| `scripts/workspace_lock.py` | 遗留工具；AI 改代码**不**调用 |
| `scripts/agent_wakeup.py` | `AGENT_WAKEUP` |
| `scripts/repo_env.py` | 切到仓库根 |
| `slurm/remote_status.sh` | 本机包装 `sar gpus`；提交仍走 `sar task add` |
| `slurm/gpu_availability.py` | 遗留；查卡用 `sar gpus` |
| `slurm/tail_remote_logs.py` | 读作业日志（common / 仓库 logs/） |

仓库内若仍有 `sbatch-*.sh`、`prototype.slurm`、`launch-*.sh`、`scripts/train/`：人可用；**AI 提交 Slurm 走 `sar`，不要直接 sbatch**。
