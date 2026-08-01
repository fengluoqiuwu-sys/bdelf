---
name: train
description: >-
  Run bdelf pretraining via train.py: local fast/debug training on RTX 5080 and
  remote full training on ovan-server. Understand the train config layout
  (model/batch/eval/hardware/optimizer/schedule), construct train.py CLI args,
  and read/verify checkpoints and train/eval logs. Use when the user wants to
  launch or inspect training, pick a train config, or check train/eval progress.
---

# train

训练入口：`python train.py`。配合 rule「本机与远端计算约束」、skill `train-ops`
（调度/互斥/远端 Slurm 登记）与 `sync-ovan-server`（push/pull）使用。本 skill
聚焦**命令与配置**本身；要不要跑 full、提交到远端、拉 checkpoint 等流程决策见
`train-ops` 与 `auto-train`。

## 命令格式

```bash
python train.py \
  --model     <model>      # 必须：ar | ar1_5 | ar2 | bd3lm | bdelf | elf
  --config    <name>       # 必须：config/train/batch/<model>/<name>.yaml，如 100m-fast / 900m-full
  --dataset   <dataset>    # 必须：config/datasets/，如 owt / wikitext / arxiv
  --preprocess <pre>       # 必须：config/preprocess/，如 default / elf
  --run-name  <dir>        # 可选：checkpoint 目录名，默认为 train config 的 name
```

可用值都可直接查：`--help` 的 epilog 会列出 Available models / datasets / preprocess。
校验失败会 `SystemExit` 并提示可用项。

### 运行位置与规模

- 本机 5080（fast）：`python train.py --model elf --config 100m-fast --dataset owt --preprocess elf`
- 远端 4×4090（full）：`python train.py --config 100m-full ...`（由 `slurm/full/*.slurm` 内调用）
- 本机**不要**跑 full/ultra；full 自动探测 GPU 数（须 ∈ {1,2,4,8}），本机只有 1 卡默认配 world_size=4 会启动失败。

## 训练配置（config/train/）

`get_train_config` 把多个子 YAML 组合成一份 `FL_TrainConfig`，`--config` 的名字取自
`batch/<model>/`，其余子目录全局共享。

| 子目录 | 说明 | 关键字段（示例） |
|--------|------|------------------|
| `batch/<model>/<name>.yaml` | 每 GPU 微批；微批或全局批二选一 | `batch_size`, `grad_accum_steps` **或** `global_batch_size` |
| `schedule/{fast,full,ultra}.yaml` | 训练计划 | `target_tokens`, `warmup_ratio`, `{eval,save,snapshot,log_plot}_step`, `resume`, `use_muon` |
| `optimizer/<model>/<name>.yaml` | 优化器 | `learning_rate`, `muon_learning_rate`, `weight_decay`, `grad_clip` |
| `eval/default.yaml` | 评测 | `eval_sample_count`, `gen_eval_model`, `gen_eval_batches` |
| `hardware/{fast-16gb,full-4x4090,full-8x4090}.yaml` | 硬件 | `world_size`, `num_workers`, `gpu_memory_gb` |

- 名字一律用 `/` 形式列出/选用：如头里 `batch/elf/{100m,300m,900m}-{fast,full,ultra}`。
- `_YAML_REQUIRED` 缺失会报错退出；非属性信息写 YAML 键（如 `_doc`）进 `extra`。
- schedule 的 `{eval,save,snapshot,log_plot}_step` 以**优化器步**为单位，内部乘以
  `grad_accum_steps` 换算成微步。改计划时长优先调 `target_tokens`。

## checkpoint 与日志

落在 `cache/checkpoints/<run>/`（`CHECKPOINT_ROOT = "cache/checkpoints"`）：

- `checkpoint_latest.pt` — 最新可续训权重（含 model/optimizer/EMA/step）
- `checkpoint_step_XXXXXXX.pt` — 历史快照
- `config.json` — 本次 `{"train": <FL_TrainConfig dict>, "model": <model_meta>}`
- `train_log.csv` / `eval_log.csv` — 训练/评测曲线；`*_ppl.png` 由 `update_ppl_plots` 绘制

`--run-name` 指定 `<run>` 目录名；不指定则取 train config 的 `name` 字段（可能与其他模型同名冲突，跨模型跑建议显式 `--run-name`）。

## 检查进度 / 续训

- `resume` 默认 true：`checkpoint_latest.pt` 存在则自动续训（CSV 按 step 截断，恢复 RNG）。
- 看曲线/评测：`pull --mode fast [NAME]` 拉回 `train_log.csv`/`eval_log.csv` 后分析，或在远端只读 `tail` 日志（见 train-ops），不要在远端跑 generate/eval。
- 已到 `max_steps`（由 `target_tokens` 折算）会直接结束；训练结束总是落最终 checkpoint。