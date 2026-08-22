# OWT 数据集与句段预处理（工程说明）

本文档只描述**仓库内实现**：配置、代码路径、缓存布局与构建命令。动机、文献与分析见 [`latex/owt-preprocess.tex`](latex/owt-preprocess.tex)。

## 数据源

| 项 | 值 |
|---|---|
| 配置 | [`config/datasets/owt.yaml`](../../config/datasets/owt.yaml) |
| 实现 | [`dataset/owt.py`](../../dataset/owt.py)（注册于 `dataset`） |
| 本地路径 | `cache/datasets/owt/`（parquet，`plain_text/train-*.parquet`） |
| HF | `Skylion007/openwebtext`，`revision: main` |

下载：

```bash
.venv/bin/python scripts/download_dataset.py owt
```

## 逻辑划分（行级 holdout）

OWT 仅下载 HF `train` split；在** parquet 行号顺序**上做一次随机划分（与旧 `eval_count=10000` **无继承**）：

| Split | 行数 | 配置字段 |
|---|---|---|
| `test` | 1024 | `test_count` |
| `dev` | 1024 | `dev_count` |
| `train` | 其余 | — |

- `holdout_seed: 42`：打乱 `0..N-1` 后，**先**取 1024 行 → `test`，**再** 1024 行 → `dev`。
- 实现：[`dataset/dataset.py`](../../dataset/dataset.py) 中 `_get_tri_holdout_sets()`、`holdout_row_split()`。
- 预处理扫 parquet 时按行号打 `train|dev|test` 标签（[`preprocess/preprocess.py`](../../preprocess/preprocess.py) `_iter_tagged_doc_batches`）。

## 预处理配置

### 新：句段切分（`strategy: owt_segment`）

| 名 | 文件 | `process_d` | Pad |
|---|---|---|---|
| `owt-seg512` | [`config/preprocess/owt-seg512.yaml`](../../config/preprocess/owt-seg512.yaml) | 512 | 统一 pad 到 512 |
| `owt-bucket` | [`config/preprocess/owt-bucket.yaml`](../../config/preprocess/owt-bucket.yaml) | 2048 | 桶：256 / 512 / 1024 / 2048 |

公共参数：

- `tokenizer: gpt2`；encode `add_special_tokens=False`；BOS/EOS 由 [`preprocess/owt_split.py`](../../preprocess/owt_split.py) `wrap_chunk` 手动添加。
- 特殊 token ID：[`config/tokenizers/gpt2.yaml`](../../config/tokenizers/gpt2.yaml) 的 `<|bos|>` / `<|eos|>` / `<|pad|>`。
- `min_chunk_len: 128`：包装后（含 BOS/EOS）长度 ≤ 128 的段丢弃。
- `shuffle_seed: 42`：每个 split 内 chunk 写盘前 **块 shuffle**（seed 与 split 名混合；块大小 65536 行，块内保持源顺序）。
- 切分完成后写 `manifest`（split `status: built`）；**block shuffle** 带进度条 `[preprocess] shuffle {split}`；完成后 `status: complete`。切分已完成但 shuffle 未做/中断时，重跑同命令会**跳过 tokenize、只 shuffle**（无 manifest 时从磁盘推断 `built`）。

### 旧：流式切块（`strategy: stream`）

| 名 | 说明 |
|---|---|
| `default-old` / `elf-old` / `cola-old` | 跨文档 token 流拼接 → 定长 chunk，行首 BOS；语义与历史 `default`/`elf` 一致 |

现行 [`default`](../../config/preprocess/default.yaml) 仍为 `stream`，**未**改为句段逻辑。

## 句段算法（代码入口）

| 模块 | 职责 |
|---|---|
| [`preprocess/owt_split.py`](../../preprocess/owt_split.py) | 分隔符检测、`split_token_ranges`、`chunk_document`、`bucket_pad_length` |
| [`preprocess/owt_segment_build.py`](../../preprocess/owt_segment_build.py) | 多进程按文档切分、pad、block shuffle、写 shard |
| [`preprocess/preprocess.py`](../../preprocess/preprocess.py) | `get_preprocess` / `get_preprocessed`、指纹、`manifest`、stream / owt_segment 分支 |

切分要点（实现细节）：

- 内容上限 `content_cap = d - 2`（预留 BOS/EOS）。
- 在 `target = start + content_cap` 前 **lookback = min(256, d/8)** 个 token 内选分隔符；P0 段落空行 → P1 换行 → P2 句末标点；找不到则硬切，下一段从切点继续。
- 分隔符在原文上匹配，token 边界用 Fast tokenizer 的 `offset_mapping`（[`encode_doc`](../../preprocess/owt_split.py)）。

## 构建缓存

```bash
# 定长 512
.venv/bin/python scripts/preprocess.py --dataset owt --preprocess owt-seg512

# 变长分桶
.venv/bin/python scripts/preprocess.py --dataset owt --preprocess owt-bucket

# 旧流式（GPT-2 1024）
.venv/bin/python scripts/preprocess.py --dataset owt --preprocess default-old
```

入口：[`scripts/preprocess.py`](../../scripts/preprocess.py)。要求 raw parquet 已下载且体积 ≥ 1 GiB。

并行：主进程顺序 `iter_parquet_rows`；`ProcessPoolExecutor`（spawn）按文档 `chunk_document`；worker 数 = 可见 CPU 数 − 1（`sched_getaffinity`，Slurm 作业内为分配核数）；`max_inflight ≈ 2×workers`。

## 缓存目录

根路径：`cache/preprocessed_datasets/`（**不同步** push）。

目录名：`{dataset}_{preprocess}_{16hex_fingerprint}/`

示例：`owt_owt-seg512_c661b6d8c3f996d0`

| 文件 | 含义 |
|---|---|
| `manifest.yaml` | 版本、指纹、`split_counts`、各 split 元数据 |
| `{split}.{shard:05d}.bin` | `int32` memmap，形状 `(N, chunk_length)` |
| `{split}.len` | 有效 token 数（含 BOS/EOS），用于 loss mask |

- `owt_segment`：`manifest.version = 3`，含 `process_d`、`pad_mode`、`bucket_lengths`、`shuffle_seed` 等。
- `stream`：`manifest.version = 2`，含 `overflow_mode`。

`owt-bucket` 的 split 元数据可含 `bucket_counts`（各 pad 桶样本数）。

指纹：[`_fingerprint`](../../preprocess/preprocess.py) — 预处理 YAML 字段 + 数据集 holdout 字段（`dev_count` / `test_count` / `holdout_seed`）的 JSON 哈希前 16 hex。改名或改参数会**新目录**，不软链旧缓存。

## 消费侧（当前状态）

- `get_preprocessed("owt-seg512", "owt")` → `load_split("train"|"dev"|"test")` 返回 `input_ids` + 可选 `length`。
- **训练脚本尚未接线** `dev`/`test` 与变长分桶 batching；续训旧 run 仍用 `--preprocess default` / `elf` 等 stream 配置。

## 与历史缓存的关系

| 旧目录 | 说明 |
|---|---|
| `owt_default_*` | `eval_count=10000` 二划分 + stream，**不兼容**新三分割 |
| `owt_owt-seg512_*` | 新指纹；需完整跑完 `scripts/preprocess.py` 得到 `status: complete` |

勿将旧目录软链为新名。
