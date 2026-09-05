---
name: compute-ops
description: >-
  Local GPU debug jobs and remote GPU jobs: Slurm via sbatch wrappers
  (ovan-server; mandatory remote_status.sh) or common via launch wrappers
  (--server + --gpus). Job logs under logs/<server>/<timestamp>/. Agent
  registry in temp/agent/active. Use when starting/stopping jobs, sbatch/
  scancel, or checking queues. Hard limits in compute-local /
  compute-remote-slurm / compute-remote-common. For push/pull use skill sync.
---

# compute-ops

操作流程 skill。硬约束见 rule「本机计算约束」「远端 Slurm 计算约束」「远端 common 计算约束」「脚本约定」「Python 虚拟环境」。  
同步见 `sync`。

## 本机调试占卡

本机为**本地调试机**（非 `servers.csv` 服务、非 common）：占资源作业须写本地 `temp/agent/`（`scheduler: "local"`）。

1. 扫 `temp/agent/active/`：死 PID 清 active；再 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` 确认 GPU 空闲。
2. 本会话旧 GPU 进程可停；用户自启进程勿杀。同时只跑一个占 GPU 任务。
3. 启动后立即登记 `active/pid<PID>.json` + `launched/`（含 `pid` / `cpus` / `gpus`，`scheduler: "local"`）。
4. 冒烟看到**首批指标正常**即可尽快停，勿占满本机卡做正式重计算。结束后删对应 active、更新 launched。

## 本机分支（AI）

1. 改代码前：`git branch --show-current` 必须为 **`master`**；否则 `git switch master`。
2. **禁止**为实验另开实现分支；思路隔离用 `temp/`。
3. 改动默认向前兼容；破坏性改动须用户二次确认。
4. **不**抢工作区锁（不要调用 `scripts/workspace_lock.py`）。

## 远端状态工具（Slurm / ovan-server 强制）

对 **ovan-server** 做作业相关操作（`sbatch` / `scancel`、改 agent 登记）之前，在本机**先**执行：

```bash
bash slurm/remote_status.sh          # 可读表；机器用加 --json
```

一次 ssh，汇总：`gpu_availability` + `squeue` + `temp/agent/active/*.json`（及 `agent_gpu_sum`、`agent_gpu_budget`）。  
确认 **AI 登记合计 GPU + 本作业申请 ≤ `agent_gpu_budget`**（csv「最大使用显卡数量」，现场读，不要默记）后再 `sbatch`。`AVAIL` 仅作参考——不足时仍应提交让 Slurm 排队，**不要**空等空闲卡。不要手拼多条 ssh 代替本工具。

## common 远端

对 `servers.csv` 中 `调度类型=common` 的远端机：

- 允许重 CPU，**不**登记 CPU、不经 GPU 包装器。
- 占 GPU 须用户授权；**禁止**直接跑占卡脚本，须经 `scripts/launch-*.sh`：

```bash
ssh <名字> 'cd <工作目录> && bash scripts/launch-*.sh <name> --server <名字> --gpus 0,1 [--holder WHO]'
```

- 包装器自动写该机 `temp/agent/active|launched/pid<PID>.json`（含 `gpu_ids`）与 `logs/<名字>/<时间戳>/` 下三个日志文件。
- 作业前扫 active 的 `gpu_ids` + 可选 `nvidia-smi`；无 `remote_status.sh`。

## 远端提交（Slurm / ovan-server）

### 检查清单

```text
- [ ] 包装器 / 作业脚本已就绪
- [ ] bash scripts/sync.sh ovan-server push
- [ ] bash slurm/remote_status.sh   # 强制；看 GPU / 队列 / AI 登记合计与 csv 额度
- [ ] agent_gpu_sum + 本作业 gpus ≤ agent_gpu_budget（csv「最大使用显卡数量」；额度满则等；AVAIL 不足仍可排队提交）
- [ ] bash slurm/sbatch-*.sh …      # GPU 数须与 csv 单任务上限一致
- [ ] 写 temp/agent/active/<job_id>.json + launched/<job_id>.json
```

```bash
ssh ovan-server 'cd <工作目录> && bash slurm/sbatch-*.sh <name>'
# stdout 含 Submitted batch job <id> 与 log_dir=logs/ovan-server/<时间戳>
```

模板默认 GPU 数须与 csv「单个ai任务最大使用显卡数量」一致（16 CPU / 128G）。日志目录：`logs/ovan-server/<时间戳>/`（`.out` / `.err` / `gpu-<job_id>.log`）。  
一次性手动任务额度满（`agent_gpu_sum` + 本作业 > `agent_gpu_budget`）则向用户说明后停下。

### Agent 登记（`temp/agent/`，本机/远端各自一份，不同步）

主机清单见 `scripts/servers.csv`（`slurm` | `common`）。两边目录结构相同：

| 路径 | 含义 |
|------|------|
| `active/<id>.json` | **当前未结束**的每个 AI 作业（可多个） |
| `launched/<id>.json` | 历史 |
| `current.json` | **已废弃**单槽位；仅兼容旧状态，新作业勿再写 |

#### slurm（远端 job_id）

```json
{
  "job_id": "1234567",
  "job_name": "<name>",
  "script": "scripts/<script>.sh",
  "gpus": 4,
  "started_at": "2026-08-01T12:00:00+08:00",
  "state": "SUBMITTED",
  "holder": "<who>",
  "scheduler": "slurm"
}
```

- `gpus`：本作业申请卡数（默认取 csv「单个ai任务最大使用显卡数量」）；`remote_status` 用其算 `agent_gpu_sum`。
- `holder`：哪个 AI/思路登记的，便于区分多任务。
- `state`：`SUBMITTED` | `RUNNING` | `COMPLETED` | `CANCELLED` | `FAILED`。
- 结束/取消后：更新 `launched/`，**删除**对应 `active/<job_id>.json`。
- 勿 `scancel` 非本 agent 登记（`holder` 不属于自己）的作业。
- `scancel` 前同样先跑 `remote_status.sh`。

#### local（本机调试：PID + CPU + GPU）

本机 RTX 5080 **仅调试**，不入 `servers.csv`；占资源作业须本地登记；`<id>` 用 `pid<PID>`；`scheduler: "local"`。

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

## 作业日志布局（Slurm / common 统一）

```text
logs/<server-name>/<时间戳>/
  <job-name>-<job_id>.out
  <job-name>-<job_id>.err
  gpu-<job_id>.log
  meta.json                 # 可选：server / job_id / script / …
```

- `<server-name>`：`servers.csv`「名字」（Slurm 默认 `ovan-server`）。
- `<job_id>`：Slurm 数字 job id，或 common 的进程 PID（文件名用裸 PID；agent 键为 `pid<PID>`）。
- `logs/` gitignore；**pull 增量拉取**；push 不上传且 `--delete` 不删远端。

## 远端只读：日志

`slurm/tail_remote_logs.py` 在远端执行（**无内嵌 SSH**）。脚本有更新时先 push。

```bash
# Slurm
ssh ovan-server 'cd <工作目录> && .venv/bin/python slurm/tail_remote_logs.py <JOB_ID>'
ssh ovan-server 'cd <工作目录> && .venv/bin/python slurm/tail_remote_logs.py <JOB_ID> --which err -n 120'
# common
ssh <名字> 'cd <工作目录> && .venv/bin/python slurm/tail_remote_logs.py pid12345 --server <名字>'
```

看日志不要靠 pull；勿手写长串 `ssh ... tail`，除非脚本不可用。  
查 GPU/队列/登记用 `remote_status.sh`（仅 ovan）。
