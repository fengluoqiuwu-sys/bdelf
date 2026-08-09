---
name: train-ops
description: >-
  Local fast smoke training on RTX 5080 and remote full training: Slurm via
  sbatch-train (ovan-server; mandatory remote_status.sh) or common via
  scripts/launch-train.sh (--server + --gpus). Job logs unified under
  logs/<server>/<timestamp>/ (.out/.err/gpu.log). Agent registry in
  temp/agent/active. Use when starting/stopping jobs, sbatch/scancel,
  checking queues, or evaluating after pull. Hard limits in compute-local /
  compute-remote-slurm / compute-remote-common; CLI in skill train.
---

# train-ops

操作流程 skill。硬约束见 rule「本机计算约束」「远端 Slurm 计算约束」「远端 common 计算约束」「脚本约定」「Python 虚拟环境」。  
训练参数/配置见 skill `train`；同步见 `sync`；生成见 `generate`；显存定档见 `vram-probe`；
自动闭环见 `auto-train`。

## 本机 fast 冒烟

本机为**本地调试机**（非 `servers.csv` 服务、非 common）：占资源作业须写本地 `temp/agent/`（见下「本机登记」；`scheduler: "local"`）。

1. 扫 `temp/agent/active/`：死 PID 清 active；再 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` 确认 GPU 空闲。
2. 本会话旧 GPU 进程可停；用户自启进程勿杀。同时只跑一个占 GPU 任务。
3. 启动后立即登记 `active/pid<PID>.json` + `launched/`（含 `pid` / `cpus` / `gpus`，`scheduler: "local"`）。
4. 用 `100m-fast` + `.venv/bin/python train.py ...`（或对应 `scripts/train/`）。
5. 看到**首批 loss 正常**即可；**2–3 分钟内主动停**，勿跑满 fast token 预算。正式训练只在远端 full。结束后删对应 active、更新 launched。

## 本机分支与工作区锁（AI）

多 AI / 人工并行时，本机工作树易被切走或改乱。

### 分支（一律 master）

1. 改代码 / generate 前：`git branch --show-current` 必须为 **`master`**；否则 `git switch master`。
2. **禁止**为自动训练或实验另开实现分支；思路隔离用 `temp/auto-research/<idea>/`（见 skill `auto-train`）。
3. 改动默认向前兼容、不影响其他模型；破坏性改动须用户二次确认（见 `auto-train`「改动兼容性」）。

### 工作区锁（仅非 temp 改动；generate **不**占锁）

锁路径：`temp/local-workspace.lock/`（**目录存在即占锁**；见 rule「temp/ 布局」）。  
用仓库根工具（勿手写 mkdir）：

```bash
WHO="auto-train:<idea>"   # 或 human
.venv/bin/python scripts/workspace_lock.py acquire --holder "$WHO" --purpose "<简述>"
# …改代码 / commit…
.venv/bin/python scripts/workspace_lock.py release --holder "$WHO"
.venv/bin/python scripts/workspace_lock.py status   # 可选
```

- AI：`acquire` 失败（exit 1）→ 睡 **30 分钟** 再试（auto-train「资源等待」）。
- 适用：编辑仓库非 `temp/` 文件、commit、改配置/代码、fast 冒烟改动等。
- **不适用**：只读、只动 `temp/`、`generate.py` 推理（仍须在 `master`）。
- 勿删他人仍在用的锁；若确认 AI 已死且 `status` 显示僵死，可手动 `rm -rf temp/local-workspace.lock`（慎用）。

## 远端状态工具（Slurm / ovan-server 强制）

对 **ovan-server** 做作业相关操作（`sbatch` / `sbatch-train` / `sbatch-vram-probe` / `scancel`、改 agent 登记）之前，在本机**先**执行（见 rule「远端 Slurm 计算约束」）：

```bash
bash slurm/remote_status.sh          # 可读表；机器用加 --json
```

一次 ssh，汇总：`gpu_availability` + `squeue` + `temp/agent/active/*.json`（及 `agent_gpu_sum`）。  
确认 **AI 登记合计 GPU + 本作业申请 ≤ 4** 后再 `sbatch`。`AVAIL` 仅作参考——不足时仍应提交让 Slurm 排队，**不要**空等空闲卡。不要手拼多条 ssh 代替本工具。

## common 远端

对 `servers.csv` 中 `调度类型=common` 的远端机（见 rule「远端 common 计算约束」）：

- 允许重 CPU，**不**登记 CPU、不经 `launch-train`。
- 占 GPU 须用户授权；**禁止**直接跑 `scripts/train/*.sh`，须：

```bash
bash scripts/ssh.sh <名字> -- \
  bash scripts/launch-train.sh <name> --server <名字> --gpus 0,1 [--holder WHO]
```

- `launch-train` 自动写该机 `temp/agent/active|launched/pid<PID>.json`（含 `gpu_ids`）与 `logs/<名字>/<时间戳>/` 下三个日志文件。
- 作业前扫 active 的 `gpu_ids` + 可选 `nvidia-smi`；无 `remote_status.sh`。

## 远端提交 full（Slurm / ovan-server）

### 检查清单

```text
- [ ] scripts/train/<name>.sh 已就绪（full）
- [ ] bash scripts/sync.sh ovan-server push
- [ ] bash slurm/remote_status.sh   # 强制；看 GPU / 队列 / AI 登记合计
- [ ] agent_gpu_sum + 本作业 gpus ≤ 4（额度满则等；AVAIL 不足仍可排队提交）
- [ ] bash slurm/sbatch-train.sh <name> […]   # prototype 默认 4 GPU
- [ ] 写 temp/agent/active/<job_id>.json + launched/<job_id>.json
```

### 提交

```bash
ssh ovan-server 'cd ~/source/bdelf && bash slurm/sbatch-train.sh <name>'
# 例：bash slurm/sbatch-train.sh elf-100m-full --name elf-cfg-100m-full --exclude=cls1-srv2
# 若要 2 卡：追加 --gpus-per-node=2 --mem=64G
# stdout 含 Submitted batch job <id> 与 log_dir=logs/ovan-server/<时间戳>
```

禁止 AI 提交预处理作业（`slurm/sbatch-preprocess.sh`）。模板：`slurm/prototype.slurm`（**默认 4 GPU / 16 CPU / 128G**）。  
日志目录：`logs/ovan-server/<时间戳>/`（`.out` / `.err` / `gpu-<job_id>.log`）。

AI 合计将超 4：auto-train 按「资源等待」睡 **60 分钟**再 `remote_status`（等本侧额度）；`AVAIL` 不足则**先 sbatch 排队**，再 60m 看是否 RUNNING。一次性手动任务额度满则向用户说明后停下。

## VRAM 探针（填 alloc 表；开训查表选型）

测 rank0 峰值（训练模型 + 优化器 + EMA + `gpt2-large`），把各档 GiB 填入本地 `temp/vram-probe/alloc.md`（model×batch）。  
**开训时查表**：`alloc ≤ 当前卡 total−2`，且满足本次 `global_batch_size` 整除（细则见 skill **`vram-probe`**）。

- 同属 AI **占 GPU** 作业：须 `remote_status`、写入 `temp/agent/active/`（计入合计 GPU；探针默认 1 卡）。
- 提交：`bash slurm/sbatch-vram-probe.sh [--nodelist=…] -- <vram_probe.py 参数…>`（1 卡模板；节点不写死）。
- 本机可跑探针但不建议；勿与正在跑的本地训练抢卡。

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
  "job_name": "elf-100m-full",
  "script": "scripts/train/elf-100m-full.sh",
  "gpus": 4,
  "started_at": "2026-08-01T12:00:00+08:00",
  "state": "SUBMITTED",
  "holder": "auto-train:<idea>",
  "scheduler": "slurm"
}
```

- `gpus`：本作业申请卡数（训练默认 **4**，vram-probe **1**）；`remote_status` 用其算 `agent_gpu_sum`。
- `holder`：哪个 AI/思路登记的，便于区分多任务。
- `state`：`SUBMITTED` | `RUNNING` | `COMPLETED` | `CANCELLED` | `FAILED`。
- 结束/取消后：更新 `launched/`，**删除**对应 `active/<job_id>.json`。
- 勿 `scancel` 非本 agent 登记（`holder` 不属于自己）的作业。
- `scancel` 前同样先跑 `remote_status.sh`。

```bash
ssh ovan-server 'mkdir -p ~/source/bdelf/temp/agent/active ~/source/bdelf/temp/agent/launched && cat > ~/source/bdelf/temp/agent/active/<JOB_ID>.json <<EOF
{"job_id":"<JOB_ID>","job_name":"<NAME>","script":"scripts/train/<name>.sh","gpus":4,"started_at":"<ISO>","state":"SUBMITTED","holder":"auto-train:<idea>","scheduler":"slurm"}
EOF
cp ~/source/bdelf/temp/agent/active/<JOB_ID>.json ~/source/bdelf/temp/agent/launched/<JOB_ID>.json'
```

#### local（本机调试：PID + CPU + GPU）

本机 RTX 5080 **仅调试**，不入 `servers.csv`；占资源作业须本地登记；`<id>` 用 `pid<PID>`；`scheduler: "local"`（见「本机计算约束」）。

```json
{
  "job_id": "pid12345",
  "pid": 12345,
  "job_name": "elf-100m-fast",
  "script": "scripts/train/elf-100m-fast.sh",
  "cmdline": ".venv/bin/python train.py ...",
  "cpus": 8,
  "gpus": 1,
  "started_at": "2026-08-06T09:00:00+08:00",
  "state": "RUNNING",
  "holder": "auto-train:<idea>",
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
{"job_id":"pid${PID}","pid":${PID},"job_name":"<NAME>","script":"scripts/train/<name>.sh","cmdline":"<CMD>","cpus":<N>,"gpus":1,"started_at":"$(date -Is)","state":"RUNNING","holder":"<WHO>","scheduler":"local"}
EOF
cp temp/agent/active/pid${PID}.json temp/agent/launched/pid${PID}.json
# 结束：改 launched state 后 rm temp/agent/active/pid${PID}.json
```

#### common 远端（仅占 GPU；登记卡号；不登记 CPU）

`servers.csv` 中 `调度类型=common`：纯 CPU 不写 agent；占 GPU 用 `scripts/launch-train.sh`（自动登记）。硬约束见「远端 common 计算约束」。

```json
{
  "job_id": "pid12345",
  "pid": 12345,
  "job_name": "elf-100m-full",
  "cmdline": "bash scripts/launch-train.sh elf-cfg-100m-full --server train-server-1 --gpus 0,1",
  "gpus": 2,
  "gpu_ids": [0, 1],
  "log_dir": "logs/train-server-1/20260806T100000",
  "started_at": "2026-08-06T10:00:00+08:00",
  "state": "RUNNING",
  "holder": "auto-train:<idea>",
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
  gpu-<job_id>.log          # 训练包装器后台 nvidia-smi；探针/预处理可无
  meta.json                 # 可选：server / job_id / script / …
```

- `<server-name>`：`servers.csv`「名字」（Slurm 默认 `ovan-server`）。
- `<job_id>`：Slurm 数字 job id，或 common 的进程 PID（文件名用裸 PID；agent 键为 `pid<PID>`）。
- `logs/` gitignore；**pull 增量拉取**；push 不上传且 `--delete` 不删远端。
- 旧路径 `slurm/logs/` 仍被 `tail_remote_logs.py` 兼容扫描。

## 远端只读：日志

`slurm/tail_remote_logs.py` 在远端执行（**无内嵌 SSH**）。脚本有更新时先 push。

```bash
# Slurm
ssh ovan-server 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py <JOB_ID>'
ssh ovan-server 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py <JOB_ID> --which err -n 120'
ssh ovan-server 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py --server ovan-server --list'
# common
bash scripts/ssh.sh train-server-1 -- \
  .venv/bin/python slurm/tail_remote_logs.py pid12345 --server train-server-1
bash scripts/ssh.sh train-server-1 -- \
  .venv/bin/python slurm/tail_remote_logs.py pid12345 --which gpu
```

看日志不要靠 pull；勿手写长串 `ssh ... tail`，除非脚本不可用。  
查 GPU/队列/登记用 `remote_status.sh`（仅 ovan），不要再手拼 `gpu_availability` + `squeue` + `cat current.json`。

## 效果评测（拉回本机）

1. `bash scripts/sync.sh ovan-server pull --mode fast [NAME]`
2. `bash scripts/sync.sh ovan-server pull-file NAME FILE`（或 `pull --mode common` 取 latest）
3. 本机 `generate` / 分析；遵守 GPU 互斥、在 **`master`** 上跑（generate 不占工作区锁）。

**禁止** AI 主动 `pull --mode full`。`NAME` 为 `{fast|full}/{model}/{hash}`（见 skill `train` / rule checkpoint）。
