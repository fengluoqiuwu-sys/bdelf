# BELF / RELF（工程说明）

本文档描述两族半自回归连续语言模型在 **bdelf 仓库里如何落地**：与现有模块的关系、拟议目录、加载入口、训练 / 推理接口、checkpoint 与配置指纹。  
**主树尚未注册 `--model belf` / `--model relf`**；下列路径是规格对齐后的拟议布局，不是当前可跑的 CLI。设计公式、时间场、掩码与消融网格见 [`math.md`](math.md) 与 [`latex/belf-relf.tex`](latex/belf-relf.tex)。

| 文档 | 内容 |
|---|---|
| 本文 | 工程：现状、复用点、拟议包结构、加载、指纹、训练日程 |
| [`math.md`](math.md) | 流几何、梯子、块/窗条件分解、损失拆分、CFG、半群 |
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
| 主树模型包 | **无** `models/lm/belf/`、`models/lm/relf/` |
| 配置 | **无** `config/models/lm/{belf,relf}/`、`config/train/model/{belf,relf}/` |
| 启动脚本 | **无** `scripts/train/belf-*.sh` / `relf-*.sh` |
| 入口 VAE | 已有：`load_latent_artifact(latent_model, tag)` |
| 近邻实现 | ELF（`models/lm/elf/`）、Cola Stage-2（`models/lm/cola/`）、`latent_vae` |

迁入主树时按仓库惯例：`config/models/lm/<family>/` 出现后，`models.kinds` 才会把该名登记为 `kind=lm`。未登记前 `train.py --model belf` 会失败。

两族**共享模块代码、分开训练、不共享权重**。生成循环锁死：BELF=`block_generate`，RELF=`rolling_generate`，不设 `gen_mode`。

## 与仓库内其它模型的关系

| 现有 | 复用什么 | 不复用什么 |
|---|---|---|
| [`load_latent_artifact`](../../models/latent/artifact_loader.py) | s2 入口：按 `latent_model`+`tag` 只读加载 `artifacts/latent/` | 不写回 artifacts；不走 Cola 的 `vae_loader`（那是 `artifacts/cola_vae/`） |
| [`latent_vae`](../../models/latent/latent_vae/) | 典型入口：因果 / 块因果 encoder，token 对齐 \(z\)，不压序列长度 | 不是 Stage-2 prior；出口 decoder **不是** VAE-dec |
| [`elf`](../../models/lm/elf/) | \(t=1\) 干净 rectified flow、x-pred→v-MSE、logit-normal 梯子、SC-CFG 的 \(w_{\mathrm{sc}}\) 采样 | in-context 时间 / 模式 token（`num_time_tokens>0`）；`decoder_prob` 双模式 |
| [`cola`](../../models/lm/cola/) | 2L pack `[clean\|noisy]`、块因果注意力、AdaLN-Zero DiT、独立读出 | \(t=0\) 干净；速度靶 \(z_1-z_0\)；压缩码 + `cola_vae` |
| [`bd3lm`](../../models/lm/bd3lm/) | 块间提交、KV 的**离散**对应物 | 吸收态 mask、同一网络出 logits |

入口 encoder 的注意力块长以加载结果为准（例如 `latent_vae` 的 `block_size`）。BELF 去噪块长只能是 \(1\) 或该加载块长。RELF **不声明** `block_size`，但要求加载入口块长为 \(1\)。

Cola 的 VAE 加载器与本规格入口不是同一条路：

```python
# BELF / RELF 入口（拟议）
from models.latent.artifact_loader import load_latent_artifact
loaded = load_latent_artifact("latent_vae", "100m-b16-d1")

# Cola Stage-2 入口（现网，勿混用）
# models/lm/cola/vae_loader.py → artifacts/cola_vae/<tag>/
```

现有选用示例（与本规格无关，仅说明加载器已可用）：

```bash
.venv/bin/python generate.py --latent-model latent_vae --tag 100m-b16-d1
```

## 拟议代码地图

迁入后建议两族各一包，共享一层（名称可在实现时微调，职责不可混）：

```
models/lm/
├── belf_relf_core/          # 拟议共享：映射、AdaLN-Zero G、出口、梯子 Q、CFG teacher
│   ├── layers.py            # AdaLN-Zero 块（无仿射 LN、残差 gate、调制零初始化）
│   ├── time.py              # Φ / Q / {L_i}；T≥4
│   ├── pack.py              # 2L [sg(h_left)|h_t]；组内双向、组间单向
│   └── cfg.py               # w_sc ScaleEmbedder；ctx drop；v_tgt
├── belf/
│   ├── config.py            # FL_BelfConfig；KIND 由目录 kind=lm 决定
│   └── model.py             # forward + block_generate
└── relf/
    ├── config.py            # FL_RelfConfig；无 block_size
    └── model.py             # forward + rolling_generate
```

**不要**把 BELF/RELF 做成 ELF 的 YAML 变体，也不要在 `cola` 上改时间轴冒充。条件通道钉 AdaLN-Zero，`num_time_tokens=0`。Cola [`DiTBlock`](../../models/lm/cola/layers.py) 已是 AdaLN-Zero，可作算子参考，但时间嵌入与 2L 右段 \(t\) 场必须按本规格（ELF \(t=1\)、逐列 \((t,w_{\mathrm{sc}},m)\)），不能直接调用 Cola 的 `t * ode_T`。

网络三段：

```
tokens
  → load_latent_artifact(latent_model, tag)
  → Enc → z ∈ R^{S×X} → 可选 whiten → Linear(X→n_embd)
  → pack_2l [sg(h_left)|h_t] + 逐列 (t, [w_sc], [m])
  → G（AdaLN-Zero → Attn(RoPE, qk-RMSNorm) → AdaLN-Zero → SwiGLU）
  → Exit：等宽因果 decoder → Linear(D→V)  或  Linear(D→V)
```

VAE-dec 只服务 \(\mathcal{L}_{\mathrm{s1}}\)，不代替出口读 token。

## 配置与指纹（拟议）

登记后与现网同构：

| 角色 | 路径 |
|---|---|
| 架构 | `config/models/lm/belf/{prototype,100m}.yaml`；`config/models/lm/relf/...` |
| 训练 optimizer/batch | `config/train/model/belf/{fast,full}.yaml`（relf 同） |
| 在线评测 generate | `config/generate/lm/belf/{generate,eval}.yaml` |
| 启动 | `scripts/train/belf-100m-full.sh` 等；远端经 `sbatch-train` / `launch-train` |

参数分三类（进入哪份指纹以规格为准）：

| 类 | 例子 | 指纹 |
|---|---|---|
| 模型 | `latent_model`、`tag`、`n_embd`、`exit`、`sc_cfg`；BELF 另有 `block_size`；RELF 有 `window_size`/`time_step`/`step_size` | 模型指纹 |
| 训练 | `latent_tune`、日程、损失系数 | 训练指纹 |
| 推理 | `commit_x0hat`、`sampling_method`、`sde_gamma`、\(w_{\mathrm{sc}}\)/\(w_{\mathrm{ctx}}\) | 不进训练指纹 |

`latent_tune` 三档是**三个训练配置哈希**（`frozen` / `full` / `mid`），不是同一个 run 里的开关。`sc_cfg` 进模型指纹：为假则不构建 ScaleEmbedder。

启动校验（实现时必须硬拒绝）：

- 加载 \(X\) 与映射一致；`n_embd` 整除 `n_head`
- 两族 `time_step` \(T\ge 4\)
- BELF：`block_size` \(\in\{1,\) 加载入口块长\(\}\)
- RELF：无 `block_size`；加载入口块长 \(=1\)；\(S\cdot T=W\)
- 可训档（`full` / 解冻后 `mid`）须具备 VAE-dec（加载自带或另配）

### 100m 默认（规格）

共用：`n_embd=768`，`max_seq_len=4096`，`exit=decoder`，`latent_tune=mid`（解冻点 15B），主训 45B + 扩展 5B。  
BELF：`block_size=16`，`time_step=16`（主跑 \(W=T\)）。  
RELF：`window_size=32`，`time_step=16`，`step_size=2`。

完整键与消融对照见 LaTeX 参数表；工程实现不得另发明默认。

## Checkpoint

s2 run 仍走全局布局（`world_size` 不进哈希）：

```
cache/checkpoints/{fast|full}/{belf|relf}/{config-hash}/
```

无论是否训练 latent，该目录的 checkpoint **必须写入完整 latent 参数**（含另配的 VAE-dec）。续训只认本 run；`mid` 按本 run `tokens_seen` 是否已过 `latent_thaw_tokens` 决定解冻。新 run 自 step 0 调用 `load_latent_artifact`，之后只用本 s2 副本。

入口选用权重仍在：

```
cache/checkpoints/artifacts/latent/{latent_model}/{tag}/
```

导出（现网工具，Stage-1 用）：

```bash
.venv/bin/python scripts/export_latent_artifact.py --run full/latent/latent_vae/<hash> --tag <tag>
```

禁止把 s2 训练目录写进 `artifacts/latent/`。

## 训练 / 推理接口（拟议）

登记完成后与现网相同：

```bash
.venv/bin/python train.py \
  --model belf \
  --config 100m-fast \
  --dataset owt \
  --preprocess default \
  --generate eval
```

RELF 把 `--model` 换成 `relf`。`--set` 可改训练/推理字段；改模型类字段会换哈希。本机只跑 fast 冒烟；full 经 `sbatch-train` / `launch-train`（须用户授权占 GPU）。

| 族 | `forward(tokens)` | 生成 |
|---|---|---|
| BELF | `train_t_schedule=block`：抽一跳 \(i\)，未知槽广播 \(t=L_i\) | `block_generate` |
| RELF | `train_t_schedule=mixed`：按 BOS/EOS 截整窗 \(F\) | `rolling_generate` |

评测：BELF 走块采样，RELF 走 rolling；主指标沿用仓库 TriFluency / Gen.PPL。长度切分见 LaTeX 附录（主训对齐 512，扩展对齐 2048）。

推理默认同规格：`commit_x0hat=true`，`sampling_method=sde`。前缀 KV 不随 \(w_{\mathrm{sc}}\) 重算。

## `latent_tune` 与 \(\mathcal{L}_{\mathrm{s1}}\)

| 档 | 梯度到 latent | \(\mathcal{L}_{\mathrm{s1}}\) |
|---|---|---|
| `frozen` | 否 | 不算；完整参数仍写入 ckpt |
| `full` | step 0 起 | 重建 + \(\mathrm{KL}(q\|N(0,I))\) + BERT-mask + ref-KL |
| `mid` | 未到解冻点同 frozen；之后同 full | 解冻前不算；解冻后算。新参数新优化器状态 |

\(\mathcal{L}_{\mathrm{s1}}\) 与流/出口损失 \(\mathcal{L}\) 分开相加。VAE-dec 仅由 s1 提供监督。速度 MSE 仍可反传到 encoder；2L 左段 `sg` 只切断 pack 左半。

## 明确不做（工程）

- 不把离散 BD3LM 换皮成连续嵌入
- 不把 Cola 压缩项 \(I_q(X;Z_0)\) 当主声明
- 不做 ELF `decoder_prob`；CE 只在出口读出位
- 不设 RELF `p_preroll` / `p_freeroll` / `p_postroll`；截断由 BOS/EOS 位置给出
- 不登记 CPU-only 为 GPU 作业；未实现前禁止占卡「冒烟当可行性」
