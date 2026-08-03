---
name: vram-probe
description: >-
  Single-GPU VRAM probe on ovan-server (or local, not recommended): load
  gpt2-large + train model/optimizer/EMA, measure peak memory across micro-batch
  sizes, stop on OOM. Fills model×batch alloc table under temp/vram-probe/;
  train-time queries the table against the target GPU budget and global_batch_size.
  Mutex via train-ops agent registry (multi-job; AI GPU sum ≤ 4); 2 GiB safety margin is an AI selection rule,
  not coded into the probe.
---

# vram-probe

单卡显存探针：模拟 **rank0 峰值**（训练模型 + 优化器 + EMA（若开）+ 常驻 `gpt2-large`），按候选 micro-batch 升序测峰，**首次 OOM 即停**。只打印，不写 YAML / checkpoint。

**探针职责**：测各 `(model, batch)` 的 `alloc_peak_GiB`，填入本地 `temp/vram-probe/alloc.md`。  
**开训职责**：查表 + 目标卡容量 + 本次 `global_batch_size`，选出具体 `batch_size`。

硬约束见 rule「本机/远端计算」「脚本约定」；提交互斥与登记见 skill `train-ops`；auto-train 强制前置见 skill `auto-train`。

## 产物：`temp/vram-probe/alloc.md`

二维表：**行 = 模型，列 = micro-batch**，单元格 = **`alloc_peak_GiB`**（或 `oom` / 未测）。

- 峰值是**绝对占用**，与显卡总容量无关；不同机器 GPU 大小不一，**选型时再和当前卡比**，表里不存「max_safe」。
- 改动影响显存后须重跑探针并更新对应行。勿写回 recipe 默认 YAML。
- 旧口径（只记 max_ok / 按某张卡写死安全档）作废。

## 开训时查表选型

候选集合：`1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128`。

已知：目标卡 `total_memory_GiB`、本次 `global_batch_size`、`world_size`（AI full 默认 **2**）、表中该模型各档 `alloc`：

```text
budget = total_memory_GiB − 2
chosen = max { b ∈ 候选 |
               table[model][b] 为数字
               ∧ table[model][b] ≤ budget
               ∧ global_batch_size % (b * world_size) == 0 }
```

- 缺测档（`—`）或 `oom`：不可选。
- 若 recipe 默认 `batch_size` 已等于 `chosen` → 可直接训，不必 `--set`。
- 否则在专用 `scripts/train/<name>.sh`（或当次 CLI）加 `--set batch.batch_size=<chosen>`。
- **不要**因探针改 `config/train/model/*/full.yaml` 默认 batch；换卡或换 `global_bs` 时重新查表。

## 探针行为

- 测**全** `--batches`（默认全候选）；**不**按 recipe 的 `global_batch_size` 过滤
- 测量时 `grad_accum_steps=1`（单 micro-step 峰值与真实 accum / global_bs 无关）
- `--world-size` 仅元数据（AI full 默认传 **2**）；探针进程始终 1 卡
- `torch.compile` **跟随 schedule**（full 为 true）；可用 `--no-compile` 覆盖
- 其它与训练对齐：bf16 autocast、`(loss/accum).backward()`、Muon/AdamW、EMA（若开）、常驻 gpt2-large、TF32 matmul
- 填表：`ok` 行写入 `alloc_peak_GiB`；首次 OOM 的档写 `oom`，更大档保持 `—`（未测）

## 本机

`.venv/bin/python scripts/vram_probe.py …` **能跑但不建议**；以远端目标卡为准。本机 GPU 互斥仍适用。

## 远端提交（计入 AI GPU 合计）

```text
- [ ] bash scripts/sync-ovan-server.sh push   # 探针/脚本有更新时
- [ ] bash scripts/remote_status.sh           # 强制
- [ ] agent_gpu_sum + 1 ≤ 4；有空闲 GPU（否则 auto-train 睡 60m）
- [ ] ssh 后 bash slurm/sbatch-vram-probe.sh …
- [ ] 写 temp/agent/active/<job_id>.json（gpus:1）+ launched/<job_id>.json
- [ ] 结束后更新 launched、删除 active/<job_id>.json
- [ ] 读日志 → 更新 temp/vram-probe/alloc.md 对应行 + 测量记录
```

```bash
ssh ovan-server 'cd ~/source/bdelf && bash slurm/sbatch-vram-probe.sh \
  --nodelist=cls1-srv3 \
  -- \
  --model elf --config 100m-full \
  --dataset owt --preprocess elf --generate eval \
  --batches 8,16,24,32'
```

- `--nodelist` / `--gpus-per-node` 等在 `--` **之前**（模板默认 1 GPU，**不写死节点**）
- `--` **之后**为 `scripts/vram_probe.py` 参数
- 读日志：`slurm/tail_remote_logs.py <JOB_ID>`（见 train-ops）

登记示例：

```json
{
  "job_id": "<JOB_ID>",
  "job_name": "vram-probe",
  "script": "slurm/sbatch-vram-probe.sh",
  "started_at": "<ISO>",
  "state": "SUBMITTED"
}
```

## 探针 CLI

```bash
.venv/bin/python scripts/vram_probe.py \
  --model <model> --config 100m-full \
  --dataset <ds> --preprocess <pre> --generate eval \
  [--batches 8,16,24,32] [--world-size 4] [--set SECTION.KEY=VALUE ...]
```

输出：主机 / GPU / `total_memory_GiB`、候选列表、每档 `alloc_peak_GiB` / `reserved_peak_GiB` / `smi_used_GiB` / `status`（`ok`|`oom`）、`max_ok_batch_size`。OOM 时退出码非 0，已测档仍打印。

## 与 auto-train

改动会影响显存时：**先本 skill 探针 → 更新 `alloc.md` → 开训查表选型（必要时 `--set`）→ 再 sbatch-train**。禁止未探针 / 表缺测直接 full。

`bdelf` 修好后：再跑探针填表后再训。
