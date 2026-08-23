# latent_t5 / latent_vae（工程说明）

本文档只描述**仓库内实现**：目录、配置、训练命令与代码入口。设计动机、DLM 语义空间优劣分析、公式、读出变体与**长度课程**见 [`latex/latent-model.tex`](latex/latent-model.tex)（编译得 PDF；第 2--3 节为动机与 DLM 分析，长度课程见对应节）。

## 模型族概览

| CLI `--model` | 目录 | 类型 | 简述 |
|---|---|---|---|
| `latent_t5` | [`models/latent/latent_t5/`](../../models/latent/latent_t5/) | latent | T5-small 维数；encoder 可双向/单向 + AR cross-attn decoder |
| `latent_vae` | [`models/latent/latent_vae/`](../../models/latent/latent_vae/) | latent | 同维数；因果 encoder + 并行 decoder |
| `cola_vae` | [`models/latent/cola_vae/`](../../models/latent/cola_vae/) | latent | 官方 Cola TextVAE（**未改**；Stage-2 仍只认此族） |

共享算子：[`models/latent/encdec/`](../../models/latent/encdec/)（**不是**独立 `--model`）。

二者与 ELF 冻结 `t5-small` encoder **无关**；内部为手写 RoPE Transformer，维数对齐 T5-small（`E=512, d_kv=64, d_ff=2048, 6+6 层`）。

## 参数表（`100m.yaml`）

| 键 | `latent_t5` | `latent_vae` |
|---|---|---|
| `n_embd` / `d_kv` / `d_ff` / 层数 | 512 / 64 / 2048 / enc6+dec6 | 同左 |
| `latent_dim`（瓶颈 B） | 64 | 64 |
| `max_seq_len` | 1024 | 1024 |
| `tokenizer` | gpt2 | gpt2 |
| encoder 注意力 | `bidirectional: true`（默认）或 `false`（逐 token 单向） | **禁止双向**；`block_size: 1` 单向（默认）或 `>1` 块因果 |
| 读出 | `readout: e`（可选 `b`） | 固定 B 读出 |
| 辅助损失 | `lambda_span` + span 腐蚀 | `lambda_mask` + BERT-mask |
| `beta_kl` | 0.1 | 0.1 |

## 配置路径

### 模型架构

| 文件 | 说明 |
|---|---|
| [`config/models/latent/latent_t5/prototype.yaml`](../../config/models/latent/latent_t5/prototype.yaml) | 模板 |
| [`config/models/latent/latent_t5/100m.yaml`](../../config/models/latent/latent_t5/100m.yaml) | 默认可训实例 |
| [`config/models/latent/latent_vae/prototype.yaml`](../../config/models/latent/latent_vae/prototype.yaml) | 模板 |
| [`config/models/latent/latent_vae/100m.yaml`](../../config/models/latent/latent_vae/100m.yaml) | 默认可训实例 |

### 训练配方（optimizer + batch）

| 变体 | `latent_t5` | `latent_vae` |
|---|---|---|
| fast | [`config/train/model/latent/latent_t5/fast.yaml`](../../config/train/model/latent/latent_t5/fast.yaml) | [`.../latent_vae/fast.yaml`](../../config/train/model/latent/latent_vae/fast.yaml) |
| full | [`.../full.yaml`](../../config/train/model/latent/latent_t5/full.yaml) | [`.../full.yaml`](../../config/train/model/latent/latent_vae/full.yaml) |

在线评测：[`config/train/eval/latent/default.yaml`](../../config/train/eval/latent/default.yaml)（held-out 重建 CE；latent 不跑 gen-PPL）。

### 生成 / 采样

| 用途 | 路径 |
|---|---|
| `latent_t5` generate / eval | [`config/generate/latent/latent_t5/`](../../config/generate/latent/latent_t5/) |
| `latent_vae` generate / eval | [`config/generate/latent/latent_vae/`](../../config/generate/latent/latent_vae/) |

## 代码地图

```
models/latent/
├── encdec/                 # 共享 block / encoder / 读出
│   ├── layers.py           # SelfAttention, TransformerBlock, DecoderBlock, CrossAttention
│   ├── encoder.py          # LatentEncoder（三路 mask）
│   └── readout.py          # PosteriorBReadout, PosteriorEReadout, KL
├── latent_t5/
│   ├── config.py           # FL_LatentT5Config
│   ├── model.py            # _LatentT5Backbone, FL_LatentT5Model
│   └── span.py             # span corruption（L_aux）
└── latent_vae/
    ├── config.py           # FL_LatentVAEConfig
    └── model.py            # _LatentVAEBackbone, FL_LatentVAEModel
```

### 注意力模式（`LatentEncoder`）

| 条件 | 模式 | 实现 |
|---|---|---|
| `bidirectional=true` | 全双向 | SDPA 无 mask |
| `bidirectional=false`, `block_size=1` | 逐 token 因果 | `is_causal=True`（**不走**块 mask） |
| `bidirectional=false`, `block_size>1` | 块间单向、块内双向 | [`cola_vae/layers.py`](../../models/latent/cola_vae/layers.py) 块 mask |

**谁可用哪种模式：**

| 模型 | `bidirectional=true` | `bidirectional=false` |
|---|---|---|
| `latent_t5` | 允许（**默认**） | 允许（`--set model.bidirectional=false`） |
| `latent_vae` | **禁止**（YAML/`--set` 设 true 会报错） | 固定；配合 `block_size` 选单向或块因果 |

`latent_t5` 的 encoder **始终** `block_size=1`（单向时走 `is_causal` 快路径，不启用块 mask）。
`latent_vae` 无 `bidirectional` 配置项；实现层写死 `bidirectional=false`。

### 读出（瓶颈）

| 模型 | 配置 | z 空间 | 实现类 |
|---|---|---|---|
| `latent_vae` | — | $\mathbb{R}^B$ | `PosteriorBReadout` |
| `latent_t5` | `readout=e` | $\mathbb{R}^E$ | `PosteriorEReadout`（E→B→E） |
| `latent_t5` | `readout=b` | $\mathbb{R}^B$ | `PosteriorBReadout`；decoder cross-attn K/V 从 B 投影 |

### Decoder 差异

| | `latent_t5` | `latent_vae` |
|---|---|---|
| 结构 | 6×（因果 self-attn + cross-attn + FFN） | 6×（与 encoder 同型 self-attn + FFN） |
| 输入 | token embedding（BOS + 右移） | `from_latent(z)`，无 teacher-force |
| 训练移位 | 模型内 `BOS \Vert x_{:-1}`；`full_sequence_training=True` | 无移位 |
| 生成 | AR + 固定 memory `z`；`supports_prefix=True` | 一次前向 `decode_logits` / `reconstruct` |

### 损失（训练态）

两者均为 Cola Stage-1 形式：

$$\mathcal{L} = \mathcal{L}_{\mathrm{recon}} + \beta\,\mathrm{KL}(q\| \mathcal{N}(0,I)) + \lambda\, \mathcal{L}_{\mathrm{aux}}$$

| 项 | `latent_t5` | `latent_vae` |
|---|---|---|
| $\mathcal{L}_{\mathrm{recon}}$ | 教师强制 AR CE（全长） | 并行全 token CE |
| $\mathcal{L}_{\mathrm{aux}}$ | 原地 span CE（[`span.py`](../../models/latent/latent_t5/span.py)） | BERT-mask CE |
| 日志键 | `recon_ce`, `kl`, `mask`（span） | 同左（mask 为 BERT） |

`latent_t5` encoder 词表扩展 **100** 个 sentinel（`vocab_size + [0,100)`），仅供 span 腐蚀；**不进** `lm_head`。

`latent_vae` encoder 词表 **+1** mask 行（`mask_token_id = vocab_size`）。

## 长度课程（Stage-1，非 Cola Stage-2）

`latent_t5` 与 `latent_vae` **共用**下列日程。协议动机与表格见 [`latex/latent-model.tex`](latex/latent-model.tex) 长度课程节。

总预算 **10B 有效 token（非 pad）**。配比按**有效 token 抽样比例**（不是条数、不是 pad 后长度）。**同一阶段不混** `owt-seg512` 与 `owt-bucket`（两种 512 切分不是同一分布）。阶段内计算图钉死为该阶段最大 $L$；更短桶右 pad 到 $L$ 并 mask loss。优化器走一条 **WSD**（不重置 Adam/Muon/EMA）；`beta_kl=0.1`、`lambda=1` 全程不变。

### 四阶段总表

每步约 **262K pad-token** = 图 $L$ × 全局条数（方案 2）。

| 阶段 | 有效 token | 数据集 | 图 $L$ | 全局条数 |
|---|---:|---|---:|---:|
| S1 | **3.0B** | 仅 [`owt-seg512`](../../config/preprocess/owt-seg512.yaml) | 512 | 512 |
| S2 | **1.5B** | 仅 [`owt-bucket`](../../config/preprocess/owt-bucket.yaml) 的 256+512 桶 | 512 | 512 |
| S3 | **2.5B** | 仅 owt-bucket 的 256+512+1024 桶 | 1024 | 256 |
| S4 | **3.0B** | 仅 owt-bucket 四桶 | 2048 | 128 |

### 有效 token 配比

| 来源 | S1 | S2 | S3 | S4 | 合计 |
|---|---:|---:|---:|---:|---:|
| owt-seg512 | 100%（3.00B） | — | — | — | **3.00B** |
| bucket 256 | — | 30%（0.45B） | 10%（0.25B） | 5%（0.15B） | **0.85B** |
| bucket 512 | — | 70%（1.05B） | 20%（0.50B） | 10%（0.30B） | **1.85B** |
| bucket 1024 | — | — | 70%（1.75B） | 20%（0.60B） | **2.35B** |
| bucket 2048 | — | — | — | 65%（1.95B） | **1.95B** |

### 优化器（WSD，一条轨迹）

| 段 | 有效 token | LR |
|---|---:|---|
| warmup（仅 S1 开头一次） | ~**150M** | 线性 0 → **2e-3** |
| stable | ~**9.2B** | 恒定 **2e-3** |
| decay（仅 S4 末） | **0.8B** | cosine 2e-3 → 2e-4 |

Muon + AdamW 同一峰值 LR；`weight_decay=0`；`beta1=0.9` `beta2=0.95`；`grad_clip=1`；`bf16`；`ema_decay=0.9999`。阶段边界**不**改 LR、**不**重置动量/EMA。换 run 目录须整包 `resume`，LR 按总 10B 进度算。

### 阶段闸与观察窗

| 切换 | 条件 |
|---|---|
| S1→S2 | dev `recon_ce` 走平；KL 不贴 0、不发散；`mask_acc` 明显高于随机 |
| S2→S3 | **bucket-512** 重建不差于 S1 结束（勿用 seg512 dev 冒充） |
| S3→S4 | 1024 桶 `recon_ce` 已降；256/512 桶无遗忘；S3 入口观察窗已过 |

S3/S4 入口各留 **0.2B** 观察窗（计入该阶段预算）：允许 CE/KL 尖峰、clip 变多；**禁止**降 LR、重置优化器、改 `beta_kl`。窗后仅长桶崩、短桶还好 → 本阶段加 0.5B（配比不变）；长短一起崩 → 停训查因。S4 cosine **只吃最后 0.8B**（观察窗结束后）。

### 采样与评测

- 同一步只来自**同一数据集、同一桶**（256 与 2048 不拼 batch）。
- 桶间按上表有效 token 配比抽（非语料自然比例）。
- S1：owt-seg512 的 dev；S2 起按 owt-bucket **各桶**报 `recon_ce` / `kl` / `mask_acc`，并保留 bucket-512 遗忘对照。禁止用一条 `eval_loss` 决策。

### 工程备注

- 长度课程：`--preprocess latent-curriculum` + `--config 100m-curriculum`（`scripts/train/latent-*-100m-curriculum.sh`）；sampler `train/latent_curriculum.py`，分桶评测 `train/latent_eval.py`。
- 在线 eval：S1 起写 `seg512_*`；S2 起追加 `b256/512/1024/2048_*`（`eval_log.csv` 宽表）。
- 仓库 `target_tokens` 按含 pad 计；设日程时按该阶段有效 token 比折算。
- 4 卡参考 micro：S1/S2=16；S3=16 或 8；S4=8 或 4（T5 OOM 只减 micro，accum 维持全局条数）。

## 训练与 checkpoint

### 本机 fast 冒烟

```bash
.venv/bin/python train.py \
  --model latent_vae \
  --config 100m-fast \
  --dataset owt \
  --preprocess default \
  --generate eval

.venv/bin/python train.py \
  --model latent_t5 \
  --config 100m-fast \
  --dataset owt \
  --preprocess default \
  --generate eval
```

### 远端 full · 长度课程（10B 有效 token）

```bash
bash scripts/train/latent-vae-100m-curriculum.sh
bash scripts/train/latent-t5-100m-curriculum.sh
```

`--preprocess latent-curriculum`；checkpoint 仍在 `full/latent/...`。

### 远端 full（定长，无课程）

```bash
bash scripts/train/latent-vae-100m-full.sh
bash scripts/train/latent-t5-100m-full.sh
```

经 `slurm/sbatch-train.sh` 或 `scripts/launch-train.sh` 提交；日志 `logs/<服务名>/<时间戳>/`。

### T5 读出消融

```bash
.venv/bin/python train.py \
  --model latent_t5 \
  --config 100m-fast \
  --dataset owt \
  --preprocess default \
  --generate eval \
  --set model.readout=b
```

`readout` 进入配置哈希；`e` 与 `b` 为不同 run 目录。

### T5 encoder 单向消融

```bash
.venv/bin/python train.py \
  --model latent_t5 \
  --config 100m-fast \
  --dataset owt \
  --preprocess default \
  --generate eval \
  --set model.bidirectional=false
```

仅 `latent_t5` 可切换；`latent_vae` **禁止** `bidirectional=true`。

### Checkpoint 布局

```
cache/checkpoints/{fast|full}/latent/{latent_t5|latent_vae}/{config-hash}/
```

解析：

```bash
.venv/bin/python scripts/resolve_checkpoint.py \
  --model latent_vae \
  --config 100m-fast \
  --dataset owt \
  --preprocess default \
  --generate eval
```

### 生成（本机）

```bash
.venv/bin/python generate.py --run fast/latent/latent_vae/<hash>
.venv/bin/python generate.py --run fast/latent/latent_t5/<hash>
```

`latent_t5` 支持 `--prompt` 前缀；无条件时先验形状随 `readout`（`e`→$(L,E)$，`b`→$(L,B)$）。

## 与 `cola_vae` 的关系

| | `cola_vae` | `latent_vae` |
|---|---|---|
| 块算子 | SwiGLU + post-norm + VAE RoPE | GELU + pre-norm + 仓库 RoPE |
| 默认规模 | 384 维 × 4 层 | 512 维 × 6 层 |
| Cola Stage-2 | **加载此族** | 不接 DiT |

勿将二者 checkpoint 混用。

## PDF 文档

完整设计说明（公式、读出图、损失推导、局限）：

```bash
cd docs/latent-model/latex
latexmk -xelatex -interaction=nonstopmode -halt-on-error latent-model.tex
```

产物：[`latex/latent-model.pdf`](latex/latent-model.pdf)。
