---
name: train-ops
description: >-
  Local fast smoke training on RTX 5080 and remote full training on ovan-server
  via Slurm: mandatory scripts/remote_status.sh before remote job ops,
  sbatch-train (default 2 GPU), multi-job agent registry under temp/agent/active,
  remote logs over ssh. Use when starting/stopping jobs, sbatch/scancel,
  checking queues, or evaluating after pull. Hard limits in compute-local /
  compute-remote; CLI in skill train.
---

# train-ops

操作流程 skill。硬约束见 rule「本机计算约束」「远端计算约束」「脚本约定」「Python 虚拟环境」。  
训练参数/配置见 skill `train`；同步见 `sync-ovan-server`；生成见 `generate`；显存定档见 `vram-probe`；
自动闭环见 `auto-train`。

## 本机 fast 冒烟

1. `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` 确认 GPU 空闲。
2. 本会话旧 GPU 进程可停；用户自启进程勿杀。同时只跑一个占 GPU 任务。
3. 用 `100m-fast` + `.venv/bin/python train.py ...`（或对应 `scripts/train/`）。
4. 看到**首批 loss 正常**即可；**2–3 分钟内主动停**，勿跑满 fast token 预算。正式训练只在远端 full。

## 本机分支与工作区锁（AI）

多 AI / 人工并行时，本机工作树易被切走。

### 分支（generate 与改代码都要）

1. 操作前：`git branch --show-current` 必须等于本任务分支（如 `<idea>`）；不对则 `git switch <idea>`。
2. 操作完成后：`git switch master`（把工作树交还默认分支；commit 留在任务分支上）。

### 工作区锁（仅非 temp 改动；generate **不**占锁）

锁路径：`temp/local-workspace.lock/`（**目录存在即占锁**；见 rule「temp/ 布局」）。  
`meta.json` 的 `holder` 必须写明是谁（如 `auto-train:<idea>` / `human`）。

```bash
# 抢锁：mkdir → 写 holder → 睡 1s → 读回校验
WHO="auto-train:<idea>"   # 或 human
mkdir temp/local-workspace.lock || { echo "锁被占用"; exit 1; }
cat > temp/local-workspace.lock/meta.json <<EOF
{"holder":"$WHO","purpose":"<简述>","acquired_at":"$(date -Is)"}
EOF
sleep 1
# 读回：holder 必须仍是自己，否则视为未抢到
got=$(.venv/bin/python -c "import json; print(json.load(open('temp/local-workspace.lock/meta.json'))['holder'])")
if [[ "$got" != "$WHO" ]]; then
  echo "校验失败: holder=$got（期望 $WHO）"
  # 若目录是自己 mkdir 的且 holder 已是他人，不要 rm；直接放弃
  exit 1
fi

# 释放（仅 holder 为自己时）
rm -rf temp/local-workspace.lock
```

- AI：`mkdir` 失败，或写后 1s 读回 `holder` ≠ 自己 → 睡 **30 分钟** 再试（auto-train「资源等待」）。
- 适用：编辑仓库非 `temp/` 文件、commit、改配置/代码、fast 冒烟改动等。
- **不适用**：只读、只动 `temp/`、`generate.py` 推理（仍要切分支/恢复 master）。

### 人工上锁（自己改代码时）

```bash
WHO=human
mkdir temp/local-workspace.lock || { echo "锁被占用"; exit 1; }
cat > temp/local-workspace.lock/meta.json <<EOF
{"holder":"$WHO","purpose":"manual edit","acquired_at":"$(date -Is)"}
EOF
sleep 1
got=$(.venv/bin/python -c "import json; print(json.load(open('temp/local-workspace.lock/meta.json'))['holder'])")
[[ "$got" == "$WHO" ]] || { echo "校验失败: holder=$got"; exit 1; }
# …自己改完…
rm -rf temp/local-workspace.lock
```

勿删他人仍在用的锁；若确认 AI 已死可手动 `rm -rf`。

## 远端状态工具（强制）

任何远端**作业操作**（`sbatch` / `sbatch-train` / `sbatch-vram-probe` / `scancel`、改 agent 登记）之前，在本机**先**执行：

```bash
bash scripts/remote_status.sh          # 可读表；机器用加 --json
```

一次 ssh，汇总：`gpu_availability` + `squeue` + `temp/agent/active/*.json`（及 `agent_gpu_sum`）。  
确认：集群有足够 `AVAIL`、**AI 登记合计 GPU + 本作业申请 ≤ 4** 后，再继续。不要手拼多条 ssh 代替本工具。

## 远端提交 full

### 检查清单

```text
- [ ] scripts/train/<name>.sh 已就绪（full）
- [ ] bash scripts/sync-ovan-server.sh push
- [ ] bash scripts/remote_status.sh   # 强制；看 GPU / 队列 / AI 登记合计
- [ ] agent_gpu_sum + 本作业 gpus ≤ 4；有足够 AVAIL
- [ ] bash slurm/sbatch-train.sh <name> […]   # prototype 默认 2 GPU
- [ ] 写 temp/agent/active/<job_id>.json + launched/<job_id>.json
```

### 提交

```bash
ssh ovan-server 'cd ~/source/bdelf && bash slurm/sbatch-train.sh <name>'
# 例：bash slurm/sbatch-train.sh elf-100m-full --name elf-cfg-100m-full --exclude=cls1-srv2
# 人工若要 4 卡：追加 --gpus-per-node=4 --mem=128G（AI 自动训练保持默认 2）
```

禁止 AI 提交 preprocess 作业。模板：`slurm/prototype.slurm`（**默认 2 GPU**）。

无足够空闲卡或 AI 合计将超 4：auto-train 按「资源等待」睡 **60 分钟**再 `remote_status`；一次性手动任务则向用户说明后停下。

## VRAM 探针（填 alloc 表；开训查表选型）

测 rank0 峰值（训练模型 + 优化器 + EMA + `gpt2-large`），把各档 GiB 填入本地 `temp/vram-probe/alloc.md`（model×batch）。  
**开训时查表**：`alloc ≤ 当前卡 total−2`，且满足本次 `global_batch_size` 整除（细则见 skill **`vram-probe`**）。

- 同属 AI **占 GPU** 作业：须 `remote_status`、写入 `temp/agent/active/`（计入合计 GPU；探针默认 1 卡）。
- 提交：`bash slurm/sbatch-vram-probe.sh [--nodelist=…] -- <vram_probe.py 参数…>`（1 卡模板；节点不写死）。
- 本机可跑探针但不建议；勿与正在跑的本地训练抢卡。

### Agent 登记（远端 `temp/agent/`，不同步）

| 路径 | 含义 |
|------|------|
| `active/<job_id>.json` | **当前未结束**的每个 AI 作业（可多个） |
| `launched/<job_id>.json` | 历史 |
| `current.json` | **已废弃**单槽位；仅兼容旧状态，新作业勿再写 |

```json
{
  "job_id": "1234567",
  "job_name": "elf-100m-full",
  "script": "scripts/train/elf-100m-full.sh",
  "gpus": 2,
  "started_at": "2026-08-01T12:00:00+08:00",
  "state": "SUBMITTED",
  "holder": "auto-train:<idea>"
}
```

- `gpus`：本作业申请卡数（训练默认 **2**，vram-probe **1**）；`remote_status` 用其算 `agent_gpu_sum`。
- `holder`：哪个 AI/思路登记的，便于区分多任务。
- `state`：`SUBMITTED` | `RUNNING` | `COMPLETED` | `CANCELLED` | `FAILED`。
- 结束/取消后：更新 `launched/`，**删除**对应 `active/<job_id>.json`。
- 勿 `scancel` 非本 agent 登记（`holder` 不属于自己）的作业。
- `scancel` 前同样先跑 `remote_status.sh`。

### 登记写入示例

```bash
ssh ovan-server 'mkdir -p ~/source/bdelf/temp/agent/active ~/source/bdelf/temp/agent/launched && cat > ~/source/bdelf/temp/agent/active/<JOB_ID>.json <<EOF
{"job_id":"<JOB_ID>","job_name":"<NAME>","script":"scripts/train/<name>.sh","gpus":2,"started_at":"<ISO>","state":"SUBMITTED","holder":"auto-train:<idea>"}
EOF
cp ~/source/bdelf/temp/agent/active/<JOB_ID>.json ~/source/bdelf/temp/agent/launched/<JOB_ID>.json'
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
3. 本机 `generate` / 分析；遵守 GPU 互斥、分支切回约定（见上；generate 不占工作区锁）。

**禁止** AI 主动 `pull --mode full`。`NAME` 为 `{fast|full}/{model}/{hash}`（见 skill `train` / rule checkpoint）。
