---
name: eval
description: >-
  TriFluency offline eval via eval.py: local multi-generate tables, skip
  existing results, remote launch (sbatch-eval / launch-eval), and sync of
  cache/eval. Use when scoring checkpoints, sweeping generate params, or
  submitting eval jobs on Slurm/common.
---

# eval

仓库根；Python：`.venv/bin/python`。硬约束见 rule「本机计算约束」「远端 Slurm / common 计算约束」。  
同步见 skill `sync`；占卡登记 / `remote_status` 见 skill `train-ops`；交互式续写见 `generate`。

## 何时用

- 对本机 / 远端 `full` checkpoint 跑 TriFluency（`eval.py`）
- 同一 checkpoint 扫多组 generate 参数（`--table`）
- 远端提交 eval 作业（Slurm / common）
- 推 checkpoint 时带上已有 `cache/eval/`，避免远端重复跑

## 输出布局

```text
cache/eval/{model}/{model-hash}/{step}/
  results.csv                      # 同 step 各组 name × 指标（浮点四位小数；自动刷新）
  results.png                      # 主指标柱状图（柱顶四位小数）
  results_table.png                # 给人看的汇总表图（四位小数）
  {generate-hash}/
    fingerprint.json               # 含 name（name 不进 generate-hash）
    samples.txt
    summary.txt
    summary.json
```

- `generate-hash`：生成配置 + 样本参数（含 seed / micro_bs；**不含** step / **name**）
- 同 `generate-hash` 已有 `summary.json` → **默认跳过**（仍会写 name 并刷新 CSV/图）；`--force` 才重跑
- 仅刷新已有结果的 CSV / 图 / 补 name（不占 GPU）：`.venv/bin/python eval.py --rebuild-csv`

## 本机

### 单组

```bash
.venv/bin/python eval.py --run full/<model>/<hash> --name sc0.5 --micro-bs 8
.venv/bin/python eval.py --run full/<model>/<hash> --name ace-sc2 --generate eval \
  --set self_cond_cfg_scale=2.0 --num-samples 1024 --seed 42 --micro-bs 8
```

### 多组扫参（模型与 gpt2/CoLA 各加载一次）

```bash
.venv/bin/python eval.py --run full/<model>/<hash> \
  --table odar-sc-ace --micro-bs 8
```

| 参数 | 说明 |
|------|------|
| `--name` | 单组**必填**显示名（CSV 首列；**不进** generate-hash）；不可与 `--table` 同用 |
| `--table` | `config/eval/tables/<name>.yaml`，或显式 `*.yaml` 路径；扫参模式**必填** `--micro-bs` |
| `--micro-bs` | 生成 micro-batch（进 generate-hash）；本机 5080 常用 8 |
| `--force` | 忽略已有 `summary.json` |
| `--set` | 仅单组；**不可**与 `--table` 同用（覆盖写在表的 `runs:`） |
| `--rebuild-csv` | 扫描 `cache/eval`，补 name 并重写各 `results.csv` |

流程：按表逐组生成 → 释放生成模型 → 加载 gpt2-large + CoLA 一次打分 → 刷新 `{step}/results.csv`。

### 扫参表

路径：`config/eval/tables/`（`prototype.yaml` 不可直接 `--table`）。  
字段：`generate` / `num_samples` / `num_tokens` / `seed` + `runs:`（每项须含 `name` + 相对基线的覆盖）。

示例：`config/eval/tables/odar-sc-ace.yaml`；启动包装：`scripts/eval/odar-sc-ace.sh`。

## 远端提交（默认卡数与训练相同：4）

须先把权重推到目标机。`--checkpoints NAME FILE` 会**同时**推对应 `cache/eval/<model>/<hash>/`（若本地有），供跳过已跑组（见 skill `sync`）。

配方脚本：`scripts/eval/<name>.sh`（`--run` 等经 `--` 传入）。

### Slurm（ovan-server）

```bash
bash scripts/sync.sh ovan-server push
bash slurm/remote_status.sh    # 强制；agent_gpu_sum + 本作业 ≤ 4
ssh ovan-server 'cd ~/source/bdelf && bash slurm/sbatch-eval.sh odar-sc-ace -- --run full/odar/<hash>'
# 少卡：追加 --gpus-per-node=1 --mem=64G
```

模板：`slurm/eval.slurm`（默认 4 GPU / 16 CPU / 128G，与 `prototype.slurm` 对齐）。  
日志：`logs/ovan-server/<时间戳>/`；AI 须登记 `temp/agent/active/<job_id>.json`（`gpus: 4`）。

### common

```bash
ssh <名字> 'cd ~/source/bdelf && bash scripts/launch-eval.sh odar-sc-ace --server <名字> --gpus 0,1,2,3 -- --run full/odar/<hash>'
```

禁止直接 `bash scripts/eval/*.sh` 占 GPU；`launch-eval` 自动写 agent + 三日志文件。

### 拉回结果

```bash
bash scripts/sync.sh <服务名> pull --mode fast
```

`pull` 增量同步 `logs/` 与 `cache/eval/`。禁止 AI 主动 `pull --mode full`。

## 与 generate 的边界

| | `eval.py` | `generate.py` |
|--|-----------|-----------------|
| 用途 | TriFluency 离线协议 | 交互/续写采样 |
| 远端 | 经 `sbatch-eval` / `launch-eval` | **禁止**远端交互式跑 |
| 工作区锁 | 改代码才占；纯跑评测不占 | 不占 |

## 注意

- 仅 `variant=full` checkpoint。
- GPU 互斥与 agent 额度见 `train-ops` / 计算约束 rule。
- 远端需已缓存 gpt2-large / CoLA（`push` 默认含 `cache/models` `tokenizers`）。
