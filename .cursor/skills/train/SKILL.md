---
name: train
description: >-
  bdelf train.py CLI, config composition (recipe/schedule/eval/generate), and
  checkpoint hash resolution via scripts/resolve_checkpoint.py. Use when
  launching or inspecting training args, picking configs, or locating a run
  directory. For Slurm submit/register/logs use train-ops; for push/pull use
  sync-ovan-server.
---

# train

聚焦**训练命令与配置**。硬约束见 rule「本机/远端计算」「Checkpoint 路径」「Python 虚拟环境」。  
提交作业、登记、读日志 → skill `train-ops`；同步 → `sync-ovan-server`。

## CLI

```bash
.venv/bin/python train.py \
  --model     <model>       # ar | ar1_5 | ar2 | bd3lm | bdelf | elf | cola | cola_vae …
  --config    100m-fast|100m-full
  --dataset   <dataset>     # config/datasets/
  --preprocess <pre>        # config/preprocess/
  --generate  eval          # 训练在线评测用 eval；正式生成见 generate skill
  --set SECTION.KEY=VALUE   # 可重复；写入 config_refs.overrides
```

`--help` epilog 列出可用 model / dataset / preprocess。  
`--set` section：`optimizer` / `batch` / `schedule` / `eval` / `generate`。

## 配置从哪来

| 路径 | 角色 |
|------|------|
| `config/train/model/<model>/{fast,full}.yaml` | optimizer + batch（含 `global_batch_size`） |
| `config/train/schedule/{fast,full}.yaml` | token 预算、eval/save 步、resume、muon 等 |
| `config/train/eval/default.yaml` | 在线评测 |
| `config/generate/<model>/{generate,eval}.yaml` | 采样字段（因模型而异） |

- `--config 100m-fast|full` → 加载对应 `model/<model>/fast|full.yaml`。
- `grad_accum` 由 `global_batch_size / (batch_size * world_size)` 推导，须整除。
- schedule 的 `{eval,save,snapshot,log_plot}_step` 以**优化器步**计；改时长优先调 `target_tokens`。

## 定位 checkpoint 目录

与 `train.py` **相同入参**（仓库根）：

```bash
.venv/bin/python scripts/resolve_checkpoint.py \
  --model ar --config 100m-fast \
  --dataset owt --preprocess default --generate eval
# --hash-only | --json | 同样可加 --set ...
```

布局与哈希规则见 rule「Checkpoint 路径与配置哈希」。不要猜目录名。

## 启动方式

| 场景 | 命令 |
|------|------|
| 本机 fast 冒烟 | `.venv/bin/python train.py ... --config 100m-fast ...` 或 `bash scripts/train/<name>.sh`（若脚本已是 fast） |
| 远端 full | `bash slurm/sbatch-train.sh <name>`（短名 → `scripts/train/<name>.sh`；`--name` 改 job-name） |

`world_size` 按可见 GPU 探测（∈ {1,2,4,8}）。AI 远端 full 默认 2 卡（`prototype.slurm`）。本机不要跑 full 规模。

## 产物

`cache/checkpoints/{fast\|full}/{model}/{hash}/`：

- `checkpoint_latest.pt` / `checkpoint_step_*.pt`
- `config.json`、`hardware.json`、`train_log.csv`、`eval_log.csv`、曲线图
- 训练时 upsert 本地 `hash_guide.csv`（不同步）

`resume` 默认 true：有 `checkpoint_latest.pt` 则续训。进度查看与评测拉数见 `train-ops` / `sync-ovan-server`；**不要在远端跑 generate/eval**。
