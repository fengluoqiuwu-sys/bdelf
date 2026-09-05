---
name: compute-ops
description: >-
  Local GPU debug jobs and remote GPU jobs: Slurm on ovan-server via
  slurm-auto-run (sar; never raw sbatch); common via launch wrappers
  (--server + --gpus). Local/common agent registry in temp/agent; Slurm
  does not use it. Per-job GPU cap only (no aggregate AI remote quota).
  Hard limits in compute-local / compute-remote-slurm / compute-remote-common.
  Slurm details: skill slurm-auto-run. For push/pull use skill sync.
---

# compute-ops

操作流程 skill。硬约束见 rule「本机计算约束」「远端 Slurm 计算约束」「远端 common 计算约束」「脚本约定」「Python 虚拟环境」。  
Slurm 提交/取消/看卡见 `slurm-auto-run`。同步见 `sync`。

**卡数**：**没有**合计 AI 远端限额；只卡 **单次**张数（csv「单个ai任务最大使用显卡数量」）。`temp/agent` 只给本机调试与 **common** 占 GPU 用；**Slurm 不用**（不要写、不要扫它当 Slurm 账本）。

## 本机调试占卡

本机为**本地调试机**（非 `servers.csv` 服务、非 common）：本机占资源作业须写本地 `temp/agent/`（`scheduler: "local"`），只记本机进程，**不要**把 Slurm 远端作业登记进来。

1. 扫本地 `temp/agent/active/`：死 PID 清 active；再 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` 确认本机 GPU 空闲。
2. 本会话旧 GPU 进程可停；用户自启进程勿杀。同时只跑一个占 GPU 任务。
3. 启动后立即登记 `active/pid<PID>.json` + `launched/`（含 `pid` / `cpus` / `gpus`，`scheduler: "local"`）。
4. 冒烟看到**首批指标正常**即可尽快停，勿占满本机卡做正式重计算。结束后删对应 active、更新 launched。

## 本机分支（AI）

1. 改代码前：`git branch --show-current` 必须为 **`master`**；否则 `git switch master`。
2. **禁止**为实验另开实现分支；思路隔离用 `temp/`。
3. 改动默认向前兼容；破坏性改动须用户二次确认。
4. **不**抢工作区锁（不要调用 `scripts/workspace_lock.py`）。

## 远端状态（Slurm / ovan-server）

对 **ovan-server** 的 Slurm 作业：用 `slurm-auto-run`（`~/bin/sar`），**不要**直接 `sbatch` / `scancel`，**不要**在本机或项目 `temp/agent` 登记 Slurm 任务。查卡：`ssh ovan-server '~/bin/sar gpus'`，或本机 `bash slurm/remote_status.sh`。

```bash
ssh ovan-server '~/bin/sar daemon status'
ssh ovan-server '~/bin/sar gpus'            # 不依赖 daemon
ssh ovan-server '~/bin/sar status'
```

## common 远端

对 `servers.csv` 中 `调度类型=common` 的远端机：

- 允许重 CPU，**不**登记 CPU、不经 GPU 包装器。
- 占 GPU 须用户授权；**禁止**直接跑占卡脚本，须经 `scripts/launch-*.sh`：

```bash
ssh <名字> 'cd <工作目录> && bash scripts/launch-*.sh <name> --server <名字> --gpus 0,1 [--holder WHO]'
```

- `--gpus` 张数不得超过 csv「单个ai任务最大使用显卡数量」（现场读，不要默记）。不要用「最大使用显卡数量」当合计闸。
- 包装器自动写该机 `temp/agent/active|launched/pid<PID>.json`（含 `gpu_ids`）与 `logs/<名字>/<时间戳>/` 下三个日志文件。
- 作业前扫该机 `active/` 的 `gpu_ids` + 可选 `nvidia-smi`；无 `remote_status.sh`。

## 远端提交（Slurm / ovan-server）

必须经 skill `slurm-auto-run`。禁止 `bash slurm/sbatch-*.sh`、禁止写 `temp/agent` 记 Slurm 作业。

### 检查清单

```text
- [ ] 用户已确认服务名 ovan-server（调度类型 slurm）
- [ ] bash scripts/sync.sh ovan-server push
- [ ] ssh ovan-server '~/bin/sar daemon status'   # 未跑则 daemon start
- [ ] ssh ovan-server '~/bin/sar gpus'
- [ ] ssh ovan-server '~/bin/sar task add --project <短名> --cluster ID --gpu-type TYPE --gpus N -- CMD'
```

`--project` 用实例短名。**禁止** `sar project add/list/show/rm`；项目不存在或报错 → 把错误原文交给人，停止。`--gpus` 默认取 csv「单个ai任务最大使用显卡数量」。看任务：`sar task list` / `task show ID` / `status`。取消：`sar task cancel ID`。

### Agent 登记（`temp/agent/`：本机与 common；**Slurm 不用**）

本机调试与 **common** 占 GPU 写该机 `temp/agent/`。**Slurm 作业账本在 sar 安装根 `state/tasks/`**，不要再写项目 `temp/agent/`。

两边（本机 / common）目录结构相同：

| 路径 | 含义 |
|------|------|
| `active/<id>.json` | **当前未结束**的每个 AI 作业（可多个） |
| `launched/<id>.json` | 历史 |
| `current.json` | **已废弃**单槽位；仅兼容旧状态，新作业勿再写 |

#### local（本机调试：PID + CPU + GPU）

本机 RTX 5080 **仅调试**，不入 `servers.csv`；占资源作业须本地登记；`<id>` 用 `pid<PID>`；`scheduler: "local"`。不要把 Slurm 远端作业登记进来。

```json
{
  "job_id": "pid12345",
  "pid": 12345,
  "job_name": "<name>",
  "script": "scripts/<script>.sh",
  "cmdline": ".venv/bin/python …",
  "cpus": 8,
  "gpus": 1,
  "started_at": "2026-08-06T09:00:00+08:00",
  "state": "RUNNING",
  "holder": "<who>",
  "scheduler": "local"
}
```

- **必填**：`pid`、`cpus`、`gpus`、`holder`、`started_at`、`state`、`scheduler: "local"`。
- 启动前：读 `active/`；`kill -0 <pid>` 失败则视为僵死登记，删 active。
- 进程退出后：更新 launched 的 `state`，**删除** `active/pid<PID>.json`。

```bash
mkdir -p temp/agent/active temp/agent/launched
PID=<PID>
cat > temp/agent/active/pid${PID}.json <<EOF
{"job_id":"pid${PID}","pid":${PID},"job_name":"<NAME>","script":"scripts/<script>.sh","cmdline":"<CMD>","cpus":<N>,"gpus":1,"started_at":"$(date -Is)","state":"RUNNING","holder":"<WHO>","scheduler":"local"}
EOF
cp temp/agent/active/pid${PID}.json temp/agent/launched/pid${PID}.json
# 结束：改 launched state 后 rm temp/agent/active/pid${PID}.json
```

#### common 远端（仅占 GPU；登记卡号；不登记 CPU）

纯 CPU 不写 agent；占 GPU 用 launch 包装器（自动登记）。

```json
{
  "job_id": "pid12345",
  "pid": 12345,
  "job_name": "<name>",
  "cmdline": "bash scripts/launch-*.sh <name> --server <名字> --gpus 0,1",
  "gpus": 2,
  "gpu_ids": [0, 1],
  "log_dir": "logs/<名字>/20260806T100000",
  "started_at": "2026-08-06T10:00:00+08:00",
  "state": "RUNNING",
  "holder": "<who>",
  "scheduler": "common"
}
```

- **必填**：`pid`、`gpus`、`gpu_ids`、`holder`、`started_at`、`state`、`scheduler: "common"`（不要 `cpus`）。
- `gpu_ids` 与其它 active **不得重叠**。
- 单任务张数不得超过 csv「单个ai任务最大使用显卡数量」；不要用「最大使用显卡数量」当合计闸。

## 作业日志布局

common 与历史作业仍可用：

```text
logs/<server-name>/<时间戳>/
  <job-name>-<job_id>.out
  <job-name>-<job_id>.err
  gpu-<job_id>.log
  meta.json
```

- `<server-name>`：`servers.csv`「名字」。
- Slurm（sar）：以 `ssh ovan-server '~/bin/sar task show ID'` 的日志目录为准；附属跑完后 sar 会把 logs 拉回主集群。
- `logs/` gitignore；**pull 增量拉取**；push 不上传且 `--delete` 不删远端。

## 远端只读：日志

Slurm：优先 `sar task show ID` / `sar status`。需要读项目 `logs/` 时，`slurm/tail_remote_logs.py` 在远端执行（**无内嵌 SSH**）。脚本有更新时先 push。

```bash
# Slurm（若仍走仓库 logs/<服务>/<ts>/）
ssh ovan-server 'cd <工作目录> && .venv/bin/python slurm/tail_remote_logs.py <JOB_ID>'
# common
ssh <名字> 'cd <工作目录> && .venv/bin/python slurm/tail_remote_logs.py pid12345 --server <名字>'
```

看日志不要靠 pull；勿手写长串 `ssh ... tail`，除非脚本不可用。  
查 GPU/队列用 `sar gpus` / `sar status`（仅 ovan Slurm）。
