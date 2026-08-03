---
name: train-ops
description: >-
  Local fast smoke training on RTX 5080 and remote full training on ovan-server
  via Slurm: mandatory scripts/remote_status.sh before remote job ops,
  sbatch-train, agent registry under temp/agent, remote logs over ssh. Use when
  starting/stopping jobs, sbatch/scancel, checking queues, or evaluating after
  pull. Hard limits in compute-local / compute-remote; CLI in skill train.
---

# train-ops

操作流程 skill。硬约束见 rule「本机计算约束」「远端计算约束」「脚本约定」「Python 虚拟环境」。  
训练参数/配置见 skill `train`；同步见 `sync-ovan-server`；生成见 `generate`。

## 本机 fast 冒烟

1. `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` 确认 GPU 空闲。
2. 本会话旧 GPU 进程可停；用户自启进程勿杀。同时只跑一个占 GPU 任务。
3. 用 `100m-fast` + `.venv/bin/python train.py ...`（或对应 `scripts/train/`）。
4. 看到**首批 loss 正常**即可；**2–3 分钟内主动停**，勿跑满 fast token 预算。正式训练只在远端 full。

## 远端状态工具（强制）

任何远端**作业操作**（`sbatch` / `sbatch-train` / `scancel`、改 agent 登记）之前，在本机**先**执行：

```bash
bash scripts/remote_status.sh          # 可读表；机器用加 --json
```

一次 ssh，汇总：`gpu_availability` + `squeue` + `temp/agent/current.json`。  
根据输出确认有足够 `AVAIL`、无未结束的本 AI job（或已处理）后，再继续。不要手拼多条 ssh 代替本工具。

## 远端提交 full

### 检查清单

```text
- [ ] scripts/train/<name>.sh 已就绪（full）
- [ ] bash scripts/sync-ovan-server.sh push
- [ ] bash scripts/remote_status.sh   # 强制；看 GPU / 队列 / 登记
- [ ] 无未结束 AI job（或已 scancel 自己的）
- [ ] bash slurm/sbatch-train.sh <name> […]
- [ ] 写 current.json + launched/<job_id>.json
```

### 提交

```bash
ssh ovan-server 'cd ~/source/bdelf && bash slurm/sbatch-train.sh <name>'
# 例：bash slurm/sbatch-train.sh elf-100m-full --name elf-cfg-100m-full --exclude=cls1-srv2
```

禁止 AI 提交 preprocess 作业。模板：`slurm/prototype.slurm`。

### Agent 登记（远端 `temp/agent/`，不同步）

| 文件 | 含义 |
|------|------|
| `current.json` | 当前任务；空闲时 `{"job_id": null}` 或删除 |
| `launched/<job_id>.json` | 历史 |

```json
{
  "job_id": "1234567",
  "job_name": "elf-100m-full",
  "script": "scripts/train/elf-100m-full.sh",
  "started_at": "2026-08-01T12:00:00+08:00",
  "state": "SUBMITTED"
}
```

`state`：`SUBMITTED` | `RUNNING` | `COMPLETED` | `CANCELLED` | `FAILED`。  
结束/取消后更新 `launched/` 并清空 `current`。勿 `scancel` 非本 agent 登记的作业。  
`scancel` 前同样先跑 `remote_status.sh`。

### 登记写入示例

```bash
ssh ovan-server 'mkdir -p ~/source/bdelf/temp/agent/launched && cat > ~/source/bdelf/temp/agent/current.json <<EOF
{"job_id":"<JOB_ID>","job_name":"<NAME>","script":"scripts/train/<name>.sh","started_at":"<ISO>","state":"SUBMITTED"}
EOF
cp ~/source/bdelf/temp/agent/current.json ~/source/bdelf/temp/agent/launched/<JOB_ID>.json'
```

## 远端只读：日志

`slurm/tail_remote_logs.py` 在远端执行（**无内嵌 SSH**）。脚本有更新时先 push。

```bash
ssh ovan-server 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py <JOB_ID>'
ssh ovan-server 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py <JOB_ID> --which err -n 120'
ssh ovan-server 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py --list'
```

看日志不要靠 pull；勿手写长串 `ssh ... tail`，除非脚本不可用。  
查 GPU/队列/登记用 `remote_status.sh`，不要再手拼 `gpu_availability` + `squeue` + `cat current.json`。

## 效果评测（拉回本机）

1. `bash scripts/sync-ovan-server.sh pull --mode fast [NAME]`
2. `bash scripts/sync-ovan-server.sh pull-file NAME FILE`（或 `pull --mode common` 取 latest）
3. 本机 `generate` / 分析；遵守 GPU 互斥。

**禁止** AI 主动 `pull --mode full`。`NAME` 为 `{fast|full}/{model}/{hash}`（见 skill `train` / rule checkpoint）。
