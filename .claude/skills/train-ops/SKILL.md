---
name: train-ops
description: >-
  Run local fast/debug training on RTX 5080 and submit remote full training via
  Slurm on ovan-server. Use when starting/stopping training, sbatch/scancel,
  checking GPU jobs, or evaluating checkpoints after pull.
---

# train-ops

配合 rule「本机计算约束」/「远端计算约束」、rule「Python 虚拟环境」与 skill `sync-ovan-server`。
本机 Python 一律用 `.venv/bin/python`（或先 `source .venv/bin/activate`）。

## 本机（5080 / fast + 调试）

1. 检查 GPU 是否已被占用：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

2. 若已有占用：
   - 本会话拉起的进程 → 可结束后再开新任务
   - 用户自行启动的进程 → 不杀；改为等待或询问用户
3. 一次只跑一个占 GPU 的进程（训练 / generate / 显存探测等串行）。
4. 使用 `fast` / `fast-16gb` 等本机配置；不要在本机跑远端 full 规模多卡作业。
5. 本机命令用 `.venv/bin/python`（见 train / generate skill），不要用系统 python。
6. `fast` 是**冒烟验证**，不是正式训练：起进程后看到**首批 loss 正常打印、无崩溃**即可，
   **2–3 分钟内主动停掉**（kill），不要跑满 fast 的 token 预算（正式训练只在远端 full 跑）。

## 远端（4×4090 / 仅 full + Slurm）

### 硬限制（再强调）

- 只 `sbatch`：`slurm/full/*.slurm` 或同等 full；**禁止** ultra / preprocess。
- 不改远端项目文件；只写 `~/source/bdelf/temp/`。
- 先本地改脚本 → `bash sync-ovan-server.sh push` → 再提交。

### Agent 任务登记（`temp/`）

远端路径：`~/source/bdelf/temp/agent/`（push/pull 均不同步）。

| 文件 | 含义 |
|------|------|
| `current.json` | 当前 AI 任务；无任务时删除或 `{"job_id": null}` |
| `launched/<job_id>.json` | 历史：本 agent 拉起过的每个 job |

`current.json` / `launched/<id>.json` 字段示例：

```json
{
  "job_id": "1234567",
  "job_name": "elf-100m-full",
  "script": "slurm/full/elf-100m-full.slurm",
  "started_at": "2026-08-01T12:00:00+08:00",
  "state": "RUNNING"
}
```

`state` 建议：`SUBMITTED` | `RUNNING` | `COMPLETED` | `CANCELLED` | `FAILED`。

### 提交前检查清单

```text
- [ ] 本地 slurm 脚本已就绪且为 full
- [ ] bash sync-ovan-server.sh push 已完成
- [ ] 读取远端 temp/agent/current.json
- [ ] 若有未结束的 AI job：scancel 或确认已结束，并更新登记
- [ ] sbatch，写入 current.json + launched/<job_id>.json
```

### 常用 SSH 命令

```bash
# 查看队列（本账号）
ssh ovan-server 'squeue -u $USER'

# 读当前 AI 任务登记
ssh ovan-server 'mkdir -p ~/source/bdelf/temp/agent/launched && cat ~/source/bdelf/temp/agent/current.json 2>/dev/null || echo none'

# 取消自己拉起的 job
ssh ovan-server 'scancel <JOB_ID>'

# 提交（在 push 之后）
ssh ovan-server 'cd ~/source/bdelf && sbatch slurm/full/<name>.slurm'
```

登记写入示例（提交成功拿到 JOB_ID 后）：

```bash
ssh ovan-server 'mkdir -p ~/source/bdelf/temp/agent/launched && cat > ~/source/bdelf/temp/agent/current.json <<EOF
{"job_id":"<JOB_ID>","job_name":"<NAME>","script":"slurm/full/<file>.slurm","started_at":"<ISO>","state":"SUBMITTED"}
EOF
cp ~/source/bdelf/temp/agent/current.json ~/source/bdelf/temp/agent/launched/<JOB_ID>.json'
```

任务结束或取消后：更新 `launched/<id>.json` 的 `state`，并清空 `current.json`。

### 与他人任务

- `squeue` 里非本 agent 登记的 job：**不要** `scancel`。
- 资源紧张时告知用户，由用户决定。

### 只读日志

```bash
ssh ovan-server 'tail -n 80 ~/source/bdelf/slurm/logs/<job>-<id>.out'
```

## 测试效果（拉回本机）

不要在远端跑 generate/eval 做效果检查。

1. `bash sync-ovan-server.sh pull --mode fast [NAME]` — 同步目录与小文件  
2. `bash sync-ovan-server.sh pull-file NAME FILE` — 拉具体步数或所需 `.pt`  
3. 在本机跑测试；注意本机 GPU 互斥  

需要最新权重时可改用 `pull --mode common [NAME]`（见 sync skill）；**禁止** AI 主动 `pull --mode full`。
