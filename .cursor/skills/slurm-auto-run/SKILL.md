---
name: slurm-auto-run
description: >-
  Operate slurm-auto-run (alias sar) on ovan-server: start/stop the daemon,
  enqueue Slurm tasks with primary/backup cluster+GPU pairs, inspect GPUs,
  check status, and cancel tasks. AI must not run sar project (add/list/show/rm);
  missing or broken projects are handed back to the user. Use for every Slurm
  GPU/CPU job on ovan-server; never raw sbatch, slurm/sbatch-*.sh, or
  scancel. Also when the user mentions sar, 主集群/附属, 主用/备用.
---

# slurm-auto-run

登录节点上的常驻队列（已装到用户 `~/bin`）：按 qos 串行 `sbatch`，并在主集群与附属集群之间同步代码 / 指定 cache / logs。

**本模板：凡 `调度类型=slurm`（ovan-server）的占 CPU/GPU 作业，AI 必须经本工具。**  
禁止：直接 `sbatch`、`bash slurm/sbatch-*.sh`、手写 `scancel`、手写集群间 rsync。  
**禁止 AI 执行任何 `sar project`（add / list / show / rm）。** 项目未登记、路径不对或其它 project 报错 → **停下来交给人**，不要自行补登记。  
common 远端仍走 `scripts/launch-*.sh`（该机 `temp/agent` 登记占 GPU）；本机调试走 `temp/agent`。**Slurm 不写**项目 `temp/agent/`（见 `compute-ops`）。

入口：`slurm-auto-run` 或缩写 `sar`。非交互 SSH 用绝对路径：

```bash
ssh ovan-server '~/bin/sar …'
# 或
ssh ovan-server '~/bin/slurm-auto-run …'
```

参数细节：`ssh ovan-server '~/bin/sar help'` / `'~/bin/sar help task add'`。  
安装根默认 `~/slurm-auto-run`（`.env` / `config.json` / `state/tasks/`）。

## 角色

- **主集群 / 附属**：写在安装根 `.env`（`PRIMARY`、`SECONDARIES`），不能用 CLI 改。本环境示例：主=`cls1`，附属=`cls2`。代码与依赖 cache 只主→附属；结束后 logs 与指定 cache 从实际跑过的附属→主。
- **主用 / 备用**：单条任务的择路。`--cluster` + `--gpu-type` 是主用；可选再给一对 `--backup-cluster` + `--backup-gpu-type`。提交时主用卡够走主用，不够且备用够走备用，都不够仍提交主用给 Slurm 排队。

路径由同目录 `config.json` 模板生成。`project add` 只有名字、无 `--path`（**仅人执行**）。daemon 只读 `.env`，不改写。

## 安装与 daemon

未 `daemon start` 时只允许 `help`、`gpus`、`daemon start`。CLI 与 daemon 须在同一台登录机（ovan-server）。

```bash
ssh ovan-server '~/bin/sar daemon start'
ssh ovan-server '~/bin/sar daemon status'
ssh ovan-server '~/bin/sar daemon stop'
```

`daemon stop` 不 `scancel` 已提交作业，不改未提交队列。改 `.env` 后须 `stop` 再 `start`。

## 提交前

1. `daemon status` 确认在跑。
2. **不要** `project add` / `list` / `show` / `rm`。`task add` 因项目不存在或 project 报错失败 → 把错误原文交给人，停止。
3. `gpus` 或 `gpus --cluster ID` 看可用卡（只读 `scontrol`，不占卡、不依赖 daemon）。本机也可用 `bash slurm/remote_status.sh`（包装同一命令）。
4. `task add` 必填 `--project --cluster --gpu-type --gpus` 和 `-- CMD`。
5. 备用必须成对。有备用则主用、备用两侧数据都要就绪；主集群缺 `--cache` 所列路径会报错，不从附属回补。
6. 每个 qos 名同时最多 1 个本服务已提交且仍在 `squeue` 的作业。
7. 登录节点不跑重 Python、不交互占卡；作业经 daemon 的 `sbatch`。
8. `--gpus` 张数不要超过该机 csv「单个ai任务最大使用显卡数量」（现场读 `servers.csv`，不要默记）。未指定则用该上限。**不要**用「最大使用显卡数量」当合计限额。

sar 自己的账本在安装根 `state/tasks/`；本仓库 **不写** `temp/agent/` 记 **Slurm** 作业（本机也不记 Slurm）。作业脚本由 daemon 写到项目 `temp/sar-jobs/<id>/`（gitignore；勿手改、勿当 sbatch 模板）。

本机改代码：先 `bash scripts/sync.sh ovan-server push` 推到主集群工作目录；集群间同步由 sar 做，不要手写 rsync。

## 命令

```bash
ssh ovan-server '~/bin/sar help'
ssh ovan-server '~/bin/sar help task add'

# 以下 project 指令仅人执行；AI 禁止
# ssh ovan-server '~/bin/sar project add NAME'
# ssh ovan-server '~/bin/sar project list'
# ssh ovan-server '~/bin/sar project show NAME'
# ssh ovan-server '~/bin/sar project rm NAME'

ssh ovan-server '~/bin/sar gpus'
ssh ovan-server '~/bin/sar gpus --cluster ID'
ssh ovan-server '~/bin/sar gpus --json'

ssh ovan-server '~/bin/sar task add \
  --project NAME \
  --cluster ID --gpu-type TYPE --gpus N \
  [--backup-cluster ID --backup-gpu-type TYPE] \
  [--qos long] [--time 24:00:00] \
  [--cache REL] [--set KEY=VALUE] \
  -- CMD…'

ssh ovan-server '~/bin/sar task list [--project NAME] [--qos QOS] [--state STATE] [--cluster ID]'
ssh ovan-server '~/bin/sar task show ID'
ssh ovan-server '~/bin/sar task cancel ID'
ssh ovan-server '~/bin/sar status'
```

- `task add` 的 `--` 之后是作业 argv。工作目录 = 实际选中集群上的项目根。内置 `{project-dir}` 在提交时换成该侧物理路径；`--set KEY=VALUE` 替换 `{KEY}` 并写入作业环境。
- 任务 id 为 0–65535 随机整数。取消用 `task cancel ID`，只取消本服务登记的作业。
- 有未结束任务时 `project rm` 会拒绝（**仅人**执行 project）；不删磁盘上的源码 / cache / logs。
- 只在主集群跑、不要备用时，去掉 `--backup-*`。

## 同步（不要手写 rsync）

- 代码、依赖 cache：只主 → 目标附属。mtime 未变则跳过。
- 附属上项目第一次不存在：建目录与软链 → 同步代码 → `~/make-venv.sh` → `pip install -r`。
- 结束后：实际跑过的附属 → 主（成功/失败拉 logs 和指定 cache；取消只拉 logs）。
- 附属之间不同步。
