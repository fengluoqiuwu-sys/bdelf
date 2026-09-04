# 远端 common 计算约束

适用范围：`scripts/servers.csv` 中 `调度类型=common` 的**远端**主机（SSH 目标；单机、无 Slurm）。  
本机仅本地调试，见 rule「本机计算约束」（**不**算 common）。Slurm / ovan-server 见「远端 Slurm 计算约束」。

## 主机与额度

- 以 `scripts/servers.csv`（gitignore）为准：名字、调度类型、工作目录、显卡、单卡GB、最大使用显卡数量、单个 ai 任务最大使用显卡数量。
- 所有服务均可系统 SSH：`ssh <名字>`（「名字」即 SSH 主机名）。
- 同步：`bash scripts/sync.sh <名字> …`；登录/执行：`ssh <名字> 'cd <工作目录> && …'`（工作目录见 csv，通常 `~/source/bdelf`）。
- 占 GPU 时：张数与**具体卡号**均须合规；合计张数不得超过该机 csv **「最大使用显卡数量」**，单任务张数不得超过 **「单个ai任务最大使用显卡数量」**（每次启动前读 csv，不要默记数字）；`gpu_ids` 不得与已登记作业重叠。

## 状态（占 GPU 前强制）

无 `slurm/remote_status.sh`。在启动/结束**占 GPU** 作业或改该机 `temp/agent` 之前：

1. 扫该机 `temp/agent/active/*.json`：死 `pid` 清 active；收集已占用的 `gpu_ids`，并汇总 `gpus`。
2. 可选：`ssh <名字> 'nvidia-smi -L'` / `nvidia-smi …` 看物理卡与占用。
3. 确认：本作业 `gpu_ids` 与 active 无交集；张数未超 csv 上限；不与他人 `holder` 冲突。

纯 CPU 任务**不**走上述登记/额度检查。

## 提交与运行

- **禁止**直接 `bash scripts/train/<name>.sh --gpus …` 或 `bash scripts/eval/<name>.sh` 占 GPU；须经包装器（类比 `slurm/sbatch-train.sh` / `sbatch-eval.sh`）：

```bash
ssh <名字> 'cd ~/source/bdelf && bash scripts/launch-train.sh <name> --server <名字> --gpus 0,1 [--holder WHO]'
# 离线 eval（默认与训练同为 4 卡）：
ssh <名字> 'cd ~/source/bdelf && bash scripts/launch-eval.sh <name> --server <名字> --gpus 0,1,2,3 -- --run full/<model>/<hash>'
```


 `--gpus` 为物理卡号（逗号分隔）；张数须为 1/2/4/8。包装器后台拉起，并写入与 Slurm 同构的三个日志文件到 `logs/<名字>/<时间戳>/`：
 `<job-name>-<pid>.out` / `.err` / `gpu-<pid>.log`（另有 `meta.json`）。
- **允许** AI 做重 **CPU** 任务（如 `scripts/preprocess.py`）；**不**登记 CPU、不经 `launch-train` / `launch-eval`。
- **禁止** AI **擅自**占 **GPU**；须用户明确授权，并指定 `--gpus`（`launch-train` / `launch-eval` 会写 `temp/agent`，含 `gpu_ids`）。
- **generate.py**：禁止在远端交互式跑；离线 eval 经 `launch-eval`；权重/产物 `scripts/sync.sh <名字> pull …` 后本机查看。
- 可停本 `holder` 登记且仍存活的进程；勿杀用户自启 / 他人 holder。

## 文件系统

- 项目树只读；改代码只在本地 → `bash scripts/sync.sh <名字> push` → 再跑。
- 远端 AI **可写**：该机 `temp/`（agent 登记；不同步）与 `logs/`（作业日志；**pull 拉取**，push 不删远端）。
- generate / eval 产物：`scripts/sync.sh <名字> pull …`（pull 含 `logs/` 与 `cache/eval/`）后本机查看。

## AI 作业登记（仅占 GPU；登记用了哪几张卡）

路径：该机 `temp/agent/active/<id>.json` + `launched/<id>.json`；`<id>` 为 `pid<PID>`。  
`scripts/launch-train.sh` / `scripts/launch-eval.sh` **自动**写 active + launched + `logs/.../meta.json`。  
**仅**占 GPU 需要登记；纯 CPU **不写** agent。**不**登记 `cpus`。

```json
{
  "job_id": "pid12345",
  "pid": 12345,
  "job_name": "elf-100m-full",
  "script": "scripts/train/elf-cfg-100m-full.sh",
  "cmdline": "bash scripts/launch-train.sh elf-cfg-100m-full --server train-server-1 --gpus 0,1",
  "gpus": 2,
  "gpu_ids": [0, 1],
  "log_dir": "logs/train-server-1/20260806T130015",
  "started_at": "2026-08-06T10:00:00+08:00",
  "state": "RUNNING",
  "holder": "auto-train:<idea>",
  "scheduler": "common"
}
```

- **必填**：`pid`、`gpus`（张数）、`gpu_ids`（物理卡号列表，与 `--gpus` 一致）、`holder`、`started_at`、`state`、`scheduler: "common"`。
- `gpu_ids` 与其他 active 作业**不得重叠**。
- 结束后更新 launched、删对应 active。
- 读日志：`ssh <名字> 'cd ~/source/bdelf && .venv/bin/python slurm/tail_remote_logs.py pid12345 --server <名字>'`。
