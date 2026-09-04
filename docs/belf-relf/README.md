# BELF / RELF（工程说明）

本文档描述两族半自回归连续语言模型在 **bdelf 仓库里如何落地**：与现有模块的关系、目录、加载入口、训练 / 推理接口、checkpoint 与配置指纹。  
**主树已登记 `--model belf` / `--model relf`。** 设计公式、时间场、掩码与消融网格见 [`math.md`](math.md) 与 [`latex/belf-relf.tex`](latex/belf-relf.tex)。

| 文档 | 内容 |
|---|---|
| 本文 | 工程：现状、复用点、包结构、加载、指纹、训练日程、训练热路径 |
| [`math.md`](math.md) | 流几何、梯子、块/窗条件分解、v-MSE、self-left、CFG、半群 |
| [`latex/belf-relf.tex`](latex/belf-relf.tex) | 完整规格（参数表、Clean、掩码图、消融） |

图在 [`assets/`](assets/)（`plot_masks.py` 可重绘）。编译 PDF：

```bash
cd docs/belf-relf/latex
latexmk -xelatex -interaction=nonstopmode -halt-on-error belf-relf.tex
```

## 现状

| 项 | 状态 |
|---|---|
| 规格 | 冻结于本夹 LaTeX；`temp/ideas/belf-relf/proposal/` 不再同步 |
| 主树模型包 | `models/lm/belf/`、`models/lm/relf/`、共享 `models/lm/belf_relf_core/` |
| 配置 | `config/models/lm/{belf,relf}/`、`config/train/model/lm/{belf,relf}/` |
| 启动脚本 | Stage1 `…-full-s1.sh`；Stage2 `…-s2.sh`；fast 冒烟 `…-100m-fast.sh`；中档验证 `…-mid-s1.sh`（仅 s1，10B@128） |
| 入口 VAE | `load_latent_artifact(latent_model, tag)` |
| 近邻实现 | ELF（`models/lm/elf/`）、Cola Stage-2（`models/lm/cola/`）、`latent_vae` |

两族**共享模块代码、分开训练、不共享权重**。生成循环锁死：BELF=`block_generate`，RELF=`rolling_generate`，不设 `gen_mode`。主训与扩展是两次独立作业，不共用 checkpoint 目录。

## 与仓库内其它模型的关系

| 现有 | 复用什么 | 不复用什么 |
|---|---|---|
| [`load_latent_artifact`](../../models/latent/artifact_loader.py) | s2 入口：按 `latent_model`+`tag` 只读加载 `artifacts/latent/` | 不写回 artifacts；不走 Cola 的 `vae_loader`（那是 `artifacts/cola_vae/`） |
| [`latent_vae`](../../models/latent/latent_vae/) | 典型入口：因果 / 块因果 encoder，token 对齐 \(z\)，不压序列长度 | 不是 Stage-2 prior；出口 decoder **不是** VAE-dec |
| [`elf`](../../models/lm/elf/) | \(t=1\) 干净 rectified flow、x-pred→v-MSE、logit-normal 梯子、SC-CFG 的 \(w_{\mathrm{sc}}\) 采样 | in-context 时间 / 模式 token（`num_time_tokens>0`）；`decoder_prob` 双模式 |
| [`cola`](../../models/lm/cola/) | 2L pack `[clean\|noisy]`、块因果注意力、AdaLN-Zero DiT、独立读出 | \(t=0\) 干净；速度靶 \(z_1-z_0\)；压缩码 + `cola_vae` |
| [`bd3lm`](../../models/lm/bd3lm/) | 块间提交、KV 的**离散**对应物 | 吸收态 mask、同一网络出 logits |

入口 encoder 的注意力块长以加载结果为准（例如 `latent_vae` 的 `block_size`），不随 BELF 的 \(W\) 改写。BELF 的 `block_size`（\(W\)）是 \(G\) 的去噪块长：100m 默认 16。受约束的是加载入口块长：只能是 \(1\)（逐 token 因果）或等于本题 BELF \(W\)。RELF **不声明** `block_size`，加载入口块长必须为 \(1\)；窗长 `window_size` 与入口无关。

Cola 的 VAE 加载器与本规格入口不是同一条路：

```python
# BELF / RELF 入口
from models.latent.artifact_loader import load_latent_artifact
loaded = load_latent_artifact("latent_vae", "100m-b32-d1")

# Cola Stage-2 入口（现网，勿混用）
# models/lm/cola/vae_loader.py → artifacts/cola_vae/<tag>/
```

现有选用示例（与本规格无关，仅说明加载器已可用）：

```bash
.venv/bin/python generate.py --latent-model latent_vae --tag 100m-b32-d1
```

## 代码地图

两族各一包，共享一层：

```
models/lm/
├── belf_relf_core/          # 共享：映射、AdaLN-Zero G、出口、梯子 Q、CFG
│   ├── layers.py            # AdaLN-Zero；prefill_left / forward_right（训练共用左 KV）
│   ├── flex_mask.py         # 训练 Flex：2L、左 prefill 组因果、右段 Q=N KV=2N
│   ├── rows.py              # 按 batch 切行（self_left / guided 子批）
│   ├── time.py              # 训练 sample_logit_normal_t；推理 Φ / Q / {L_i}；T≥4
│   ├── flow.py              # z_t 插值、x-pred→v、v*
│   ├── pack.py              # 2L [sg(h_left)|h_t]；组内双向、组间单向
│   ├── cfg.py               # w_sc 采样；ctx drop；v_tgt
│   ├── gen_buf.py           # 生成 KV 扩容
│   ├── latent.py            # load_latent_artifact 包装、s1、入口块长校验
│   └── exit.py              # 出口叠法辅助（规格锁死 VAE-dec，无 linear）
├── belf/
│   ├── config.py            # FL_BelfConfig；KIND 由目录 kind=lm 决定
│   ├── model.py             # forward（跳过多余 G + CFG 共用左 KV）
│   └── generate.py          # block_generate
└── relf/
    ├── config.py            # FL_RelfConfig；无 block_size
    └── model.py             # forward + rolling_generate
```

**不要**把 BELF/RELF 做成 ELF 的 YAML 变体，也不要在 `cola` 上改时间轴冒充。条件通道钉 AdaLN-Zero，`num_time_tokens=0`。Cola [`DiTBlock`](../../models/lm/cola/layers.py) 已是 AdaLN-Zero，可作算子参考，但时间嵌入与 2L 右段 \(t\) 场必须按本规格（ELF \(t=1\)、逐列 \((t,w_{\mathrm{sc}},m)\)），不能直接调用 Cola 的 `t * ode_T`。

网络三段：

```
tokens
  → load_latent_artifact(latent_model, tag)
  → Enc → z ∈ R^{S×X} → 可选逐维 μ 白化（仍在 X；m,s 由 artifact 导出后离线写入）
  → G 茎 Linear(X→D=n_embd)
  → 训练 CFG：prefill_left（茎后 sg，对齐 pack 左半）+ 右段 G
     self_left / 生成完整 2L：pack_2l [sg(h_left)|h_t]
  → 逐列 (t, [w_sc], [m]) → G（AdaLN-Zero → Attn(RoPE, qk-RMSNorm) → SwiGLU）
  → FinalLayer D→X → x̂₀
  → Exit（锁死）：\(\hat x_0\) **unwhiten** 后走加载的 VAE-dec → logits（推理读出；训练主体不算 CE）
```

出口无 `exit` 键、无 `linear` 对照。推理默认每块 / 每次 pop 后 VAE-dec 读词再 encode（`commit_x0hat=false`）；对照 `true` 提交 \(\hat x_0\)、跑满后全文一次 VAE-dec。\(\mathcal{L}_{\mathrm{s1}}\) 重建仍经同一 VAE-dec。

### 训练热路径（不改规格 / 哈希）

`v_{\mathrm{tgt}}`、sc、AdaLN 与 self-left 的**数值语义**仍以 LaTeX 为准；实现做两件等价事（不进指纹、不改 YAML）：

1. **按样本跳过用不上的 \(G\)。** 先抽 `use_self ~ Bern(p_left)` 与 `g ~ Bern(p_g^{sc})`。self-left 只对命中行跑完整 2L `no_grad` \(G\)（\(t=1-\varepsilon\)、sc=0），再写回左段；0 行则整次跳过。CFG teacher 只跑 \(g=1\) 的行；\(g=0\) 时 \(v_{\mathrm{tgt}}=v^\star\)、学生 sc=0。学生始终整批。
2. **CFG 三次共用左 KV。** self-left 之后对最终左段茎一次（茎后 `sg`，对齐 `pack_2l` 左半），`prefill_left` **带梯度**（不用生成路径的 `prefill_left_kv`，也不 `copy_` 扩容）。随后 \(G_u/G_c/\) 学生只跑右段（`cat(K_左,K_右)`）。训练默认 `attn_backend=flex`：右段 `Q=N`（RELF 为 `right_len`）、`KV=left+right`，可见性与完整 2L 右块相同；左 prefill 为组因果（BELF 组长 \(W\)，RELF 组长 1）。sc 只加在右列再 `sc_proj`。self-left 那次 \(G\) 仍是完整 2L，不共用这份 KV。

对照脚本：`scripts/check_belf_relf_flex_mask.py`（右段 mask vs 2L 右块）。生成循环仍 SDPA + 按 hop 的左 KV，未改。

## 配置与指纹

登记后与现网同构：

| 角色 | 路径 |
|---|---|
| 架构 | `config/models/lm/belf/{prototype,100m}.yaml`；`config/models/lm/relf/...` |
| 训练 optimizer/batch | `config/train/model/lm/belf/{fast,mid,full}.yaml`（relf 同） |
| 在线评测 generate | `config/generate/lm/belf/{generate,eval}.yaml`；`config/generate/lm/relf/...` 同构 |
| 启动 | Stage1 `scripts/train/belf-100m-full-s1.sh`（短名 `belf-100m-full.sh`）；Stage2 `…-s2.sh`。RELF 同。远端经 `sbatch-train` / `launch-train` |

参数分三类（进入哪份指纹以规格为准）：

| 类 | 例子 | 指纹 |
|---|---|---|
| 模型 | `latent_model`、`tag`、`kl_entropy`、`n_embd`、`sc_cfg`、`self_left_prob`、`self_left_thaw_tokens`；BELF 另有 `block_size`；RELF 有 `window_size`/`step_size`（\(T=W/S\)） | 模型指纹 |
| 训练 | `latent_tune`、日程、损失系数 | 训练指纹 |
| 推理 | `commit_x0hat`、`sampling_method`、`sde_gamma`、`w_sc`/`w_ctx`；BELF 另有 `num_sampling_steps` | 现网 `build_train_fingerprint` 纳入 `generate_cfg` 与 model YAML 的 `sampling`（改推理默认会换哈希；不改全局管线） |

`latent_tune` 三档是**三个训练配置哈希**（`frozen` / `full` / `mid`），不是同一个 run 里的开关。`sc_cfg` 进模型指纹：主跑 `true`（ScaleEmbedder / teacher）；消融短训再关。

`belf` / `relf` 且 preprocess YAML 带 `curriculum` 指针时，`build_train_fingerprint` 另写入 `curriculum_cfg`（`config/train/curriculum/<名>.yaml` 正文）。改 mix / `graph_l` 会换本族哈希。其它模型不加此键，哈希不变。fast 用 `owt-seg512`（无指针）也不加。

启动校验（实现时必须硬拒绝）：

- `latent_dim` 须等于 artifact 输出维 \(X\)；`n_embd` 是 \(G\) 隐层 \(D\)，二者无关
- `n_embd` 整除 `n_head`
- BELF：加载入口 `encoder.block_size` \(\in\{1,W\}\)；**训练** `forward` 要求序列长度被 \(W\) 整除；`num_sampling_steps` \(T\ge 4\)（generate YAML）
- RELF：无 `block_size`；加载入口块长 \(=1\)；\(W\) 能被 \(S\) 整除且 \(T=W/S\ge 4\)
- 可训档（`full` / `mid`）要求入口 `encoder.block_size==1`；块因果入口只允许 `frozen`
- 出口锁死 VAE-dec，启动必须具备 decoder；可训档（`latent_tune` 的 `full` / `mid`）同样须具备 VAE-dec
- 100m 默认 `tag: 100m-b32-d1`（`kl_entropy` 关）：若 artifact 块长为 32，BELF \(W=16\) / RELF 入口必须 1 **硬拒**，不放宽校验

### 100m 默认（规格）

共用：`n_embd=768`，`max_seq_len=4096`，出口锁死 VAE-dec，主体损失仅 v-MSE（`lambda_mse`），`sc_cfg=true`，`self_left_prob=0.25`（`self_left_thaw_tokens=5B`，进指纹、不进消融），`latent_tune=mid`（解冻点 5B）。full 主训 45B + 扩展 5B。日程档 `mid` 仅 Stage1：10B、全局批 128、`eval_step=1000`。优化器与 ELF 配方相同：AdamW / Muon `learning_rate=0.002`，`dtype=bf16`；full/fast 的 `ema_decay=0.9999`，日程档 `mid` 为 `0.999`。  
BELF：`block_size=16`；训练每样本 \(t\sim\mathrm{Ln}\)（\(\mathrm{logit}(t)\sim\mathcal{N}(-1.5,0.8^2)\)）；推理 `num_sampling_steps=32`（可改）。  
RELF：`window_size=64`，`step_size=2`（推理 \(T=W/S=32\) 锁死）；训练逐档独立 \(t\sim\mathrm{Ln}\)；推理窗内铺 \(L_0,\ldots,L_{T-1}\)，最左档 Euler 到 \(1-\varepsilon\) 后提交 \(\hat x_0\)。

推理 CFG 键为 `w_sc` / `w_ctx`（默认 `w_sc=2.0`、`w_ctx=1.0`；`w_cfg` 为 `w_ctx` 别名）。主跑启用 `w_sc`。生成循环仍接受 ELF 别名 `self_cond_cfg_scale` / `ctx_cfg_scale`，但 YAML 已写 `w_sc` / `w_ctx` 时别名不生效；扫参请改这两键。

完整键与消融对照见 LaTeX 参数表；工程实现不得另发明默认。

## Checkpoint

主训（Stage1，45B@512）与扩展（Stage2，5B 混桶）是**两份独立训练**：不同 preprocess → 不同配置哈希 → 不同保存目录。`world_size` / GPU 型号不进哈希，Stage2 可换更大显存的卡。布局仍是：

```
cache/checkpoints/{fast|mid|full}/lm/{belf|relf}/{config-hash}/
```

Stage2 启动时解析同参 Stage1 哈希目录：须已有 `complete.json`（训练正常结束才写）与 `checkpoint_latest.pt`，否则直接报错退出。新 Stage2 run 从该 latest 的 **EMA** 初始化 live 权重（不恢复优化器 / step / RNG）；本 run 的 `hardware.json` 独立。`latent_tune=mid` 若 Stage1 已过 `latent_thaw_tokens`，扩展启动时立即解冻。日程档 `mid` 无 Stage2。

无论是否训练 latent，该目录的 checkpoint **必须写入完整 latent 参数**（含随 artifact 加载的 VAE-dec）。同一 Stage 的续训只认本 run。新 run 自 step 0 调用 `load_latent_artifact`，再被 Stage1 EMA 覆盖，之后只用本 run 副本。

入口选用权重仍在：

```
cache/checkpoints/artifacts/latent/{latent_model}/{tag}/
```

导出（现网工具，Stage-1 用；默认接着离线写逐维 μ 白化）：

```bash
.venv/bin/python scripts/export_latent_artifact.py --run full/latent/latent_vae/<hash> --tag <tag>
.venv/bin/python scripts/compute_latent_whiten.py --latent-model latent_vae --tag <tag>
```

禁止把 s2 训练目录写进 `artifacts/latent/`。白化统计不进 VAE 训练、不进 G 指纹；旧 artifact 无 `m,s` 时仍为单位仿射。新开 BELF/RELF 才会读到新统计（旧 run 的 `whiten_mean/std` buffer 已冻结）。

## 训练 / 推理接口

登记完成后与现网相同：

```bash
.venv/bin/python train.py \
  --model belf \
  --config 100m-fast \
  --dataset owt \
  --preprocess owt-seg512 \
  --generate eval
```

full 主训 / 扩展分开（可换卡）；`mid` 仅 Stage1（10B、全局批 128、`eval_step=1000`）：

```bash
bash scripts/train/belf-100m-full-s1.sh    # 45B@512
bash scripts/train/belf-100m-full-s2.sh    # 5B 混桶；须同参 s1 已完成
bash scripts/train/belf-100m-mid-s1.sh     # 10B@128，eval 每 1000 优化器步，仅 s1
```

RELF 把脚本名里的 `belf` 换成 `relf`。`--set` 可改训练/推理字段；改模型类字段会换哈希。本机只跑 fast 冒烟；`mid` / full 经 `sbatch-train` / `launch-train`（须用户授权占 GPU）。日程档 `mid` 与 `latent_tune=mid` 不是同一键。

| 族 | `forward(tokens)` | 生成 |
|---|---|---|
| BELF | `train_t_schedule=independent`：每样本连续 \(t\sim\mathrm{Ln}\)，未知槽广播；序列长度须被 \(W\) 整除 | `block_generate`（\(T\) 来自 `num_sampling_steps`） |
| RELF | `train_t_schedule=block`：按 BOS/EOS 切窗，逐档独立 \(t\sim\mathrm{Ln}\) | `rolling_generate`（\(T=W/S\)） |

评测：BELF 走块采样，RELF 走 rolling；主指标沿用仓库 TriFluency / Gen.PPL，不再用出口 CE 充当 eval PPL。长度切分见 LaTeX 附录（full/mid 的 Stage1 对齐 512；full 扩展对齐 2048）。

推理默认同规格：`commit_x0hat=false`（VAE-dec 读词再 encode），`sampling_method=sde`，`w_sc=2.0`，`w_ctx=1.0`。对照 `true` 提交 \(\hat x_0\)、全文一次 VAE-dec。前缀 KV 不随 \(w_{\mathrm{sc}}\) 重算。

## `latent_tune` 与 \(\mathcal{L}_{\mathrm{s1}}\)

| 档 | 梯度到 latent | \(\mathcal{L}_{\mathrm{s1}}\) |
|---|---|---|
| `frozen` | 否 | 不算；完整参数仍写入 ckpt |
| `full` | step 0 起 | 重建 + 后验项 + BERT-mask + ref-KL |
| `mid` | 未到解冻点同 frozen；之后同 full | 解冻前不算；解冻后算。新参数新优化器状态 |

\(\mathcal{L}_{\mathrm{s1}}\) 与流损失 \(\mathcal{L}=\lambda_{\mathrm{mse}}\mathcal{L}_{\mathrm{mse}}\) 分开相加。后验项由 `kl_entropy` 切换（缺键 / `false` 不进指纹、与旧哈希一致；默认 YAML 为关：仅 prior-KL；开：\(\beta\,(\mathrm{KL}+\mathbb{E}[\log q])\)，tag 后缀 `-sigma`）。主体不含出口 CE。VAE-dec 参数是否更新仍只由 `latent_tune` 决定。速度 MSE 仍可反传到 encoder；2L 左段 `sg` 只切断 pack 左半。可训档另要求入口块长为 1（块因果 encoder 禁止与 \(G\) 联合训）。

## 明确不做（工程）

- 不把离散 BD3LM 换皮成连续嵌入
- 不把 Cola 压缩项 \(I_q(X;Z_0)\) 当主声明
- 不做 ELF `decoder_prob`；主体损失无出口 CE，读出仅推理 VAE-dec
- 不设 RELF `p_preroll` / `p_freeroll` / `p_postroll`；截断由 BOS/EOS 位置给出
- 不登记 CPU-only 为 GPU 作业；未实现前禁止占卡「冒烟当可行性」
