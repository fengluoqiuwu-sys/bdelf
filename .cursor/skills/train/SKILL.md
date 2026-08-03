---
name: train
description: >-
  Run bdelf pretraining via train.py: local fast/debug training on RTX 5080 and
  remote full training on ovan-server. Understand the train config layout
  (per-model recipe + schedule/eval), construct train.py CLI args,
  and read/verify checkpoints and train/eval logs. Use when the user wants to
  launch or inspect training, pick a train config, or check train/eval progress.
---

# train

训练入口：本机用 `.venv/bin/python train.py`（见 rule「Python 虚拟环境」）。配合
rule「本机计算约束」/「远端计算约束」、skill `train-ops`（调度/互斥/远端 Slurm 登记）与
`sync-ovan-server`（push/pull）使用。本 skill 聚焦**命令与配置**本身；要不要跑
full、提交到远端、拉 checkpoint 等流程决策见 `train-ops` 与 `auto-train`。

## 命令格式

```bash
.venv/bin/python train.py \
  --model     <model>      # 必须：ar | ar1_5 | ar2 | bd3lm | bdelf | elf
  --config    <name>       # 必须：100m-fast / 100m-full（选 batch 剖面）
  --dataset   <dataset>    # 必须：config/datasets/，如 owt / wikitext / arxiv
  --preprocess <pre>       # 必须：config/preprocess/，如 default / elf
  --generate  <gen>        # 必须：config/generate/<model>/；训练用 eval，正式生成用 generate
  --set SECTION.KEY=VALUE  # 可选：覆盖 YAML 超参（可重复）
```

可用值都可直接查：`--help` 的 epilog 会列出 Available models / datasets / preprocess。
校验失败会 `SystemExit` 并提示可用项。

定位 checkpoint 目录（与 train 相同入参）::

```bash
.venv/bin/python scripts/resolve_checkpoint.py \
  --model ar --config 100m-fast \
  --dataset owt --preprocess default --generate eval
```

`--set` 在加载 recipe / schedule / eval / generate 之后、推导 accum / token 预算之前生效。section 为：
`optimizer` / `batch` / `schedule` / `eval` / `generate`。例：

```bash
--set optimizer.learning_rate=1e-3 \
--set batch.batch_size=16 \
--set schedule.target_tokens=1000000000 \
--set eval.gen_eval_samples=64 \
--set generate.temperature=0.8
```

覆盖写入 checkpoint 的 `config_refs.overrides`，便于复现。
### 运行位置与规模

- 本机 5080（fast）：`.venv/bin/python train.py --model elf --config 100m-fast --dataset owt --preprocess elf --generate eval`
- 远端 4×4090（full）：`python train.py --config 100m-full ...`（由 `slurm/full/*.slurm` 内 `source .venv/bin/activate` 后调用）
- `world_size` 按可见 GPU 数自动探测（须 ∈ {1,2,4,8}）；本机**不要**跑 full。

## 训练配置（config/train/）

`get_train_config` 把 **per-model recipe**、全局 schedule/eval 与 **generate** 组合成 `FL_TrainConfig`。

| 路径 | 说明 | 关键字段 |
|------|------|----------|
| `config/train/model/<model>/{fast,full}.yaml` | 模型配方：optimizer + batch | `learning_rate`, `batch_size`, `global_batch_size` |
| `schedule/{fast,full}.yaml` | 全局训练计划 | `target_tokens`, `warmup_ratio`, `{eval,save,snapshot,log_plot}_step`, `resume`, `use_muon` |
| `eval/default.yaml` | 全局在线评测 | `eval_sample_count`, `gen_eval_samples` |
| `config/generate/<model>/{generate,eval}.yaml` | 每模型采样（字段因模型而异） | 如 `temperature` / `top_k` / `num_steps` / `use_fast_infer` … |

- `--config 100m-fast|100m-full` 对应加载 `model/<model>/fast.yaml` 或 `full.yaml`。
- `--generate eval`：训练在线 gen-eval；`generate.py --generate generate`：正式生成。
- 梯度累积由 `global_batch_size / (batch_size * world_size)` 推导；须整除。
- schedule 的 `{eval,save,snapshot,log_plot}_step` 以**优化器步**为单位，内部按推导的
  accum 换算成微步。改计划时长优先调 `target_tokens`。

## checkpoint 与日志

落在 `cache/checkpoints/{fast|full}/{model}/{config-hash}/`（无别名；见 rule「Checkpoint 路径与配置哈希」）：

- `checkpoint_latest.pt` — 最新可续训权重（含 model/optimizer/EMA/step）
- `checkpoint_step_XXXXXXX.pt` — 历史快照
- `config.json` — 本次 `{"train": <FL_TrainConfig dict>, "model": <model_meta>}`
- `train_log.csv` / `eval_log.csv` — 训练/评测曲线；`*_ppl.png` 由 `update_ppl_plots` 绘制

用 `scripts/resolve_checkpoint.py`（与 train 相同入参）解析 `config-hash` / 目录；`generate.py --run` 填 `{variant}/{model}/{hash}`。

## 检查进度 / 续训

- `resume` 默认 true：`checkpoint_latest.pt` 存在则自动续训（CSV 按 step 截断，恢复 RNG）。
- 看曲线/评测：`pull --mode fast [NAME]` 拉回 `train_log.csv`/`eval_log.csv` 后分析，或在远端只读 `tail` 日志（见 train-ops），不要在远端跑 generate/eval。
- 已到 `max_steps`（由 `target_tokens` 折算）会直接结束；训练结束总是落最终 checkpoint。
