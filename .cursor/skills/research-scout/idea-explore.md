# idea-explore

`research-scout` 的配套流程（本文件不是独立 skill）：对一条已轻量查重的假设做深化，写入 scout 指定的 `ideas/I-{n}/`。  
**不是**给用户指定 idea 的入口；**不**改代码、不占 GPU、不提交远端作业。

配合 skill `paper-ingest`（读全文/编 INDEX）与本目录 [critic.md](critic.md)（数学他评）。

**顺序强制：先新颖性，通过后再可行性（现实性 → 数学 → critic → 机制 → 里程碑）。**

## 硬边界

- 只写父代理给定的 `…/ideas/I-{n}/`，以及 ingest 的 `temp/papers/<slug>/`。
- 禁止：改仓库代码、训练/占 GPU、远端作业、写 `temp/ideas/`、写独立的 `temp/idea-explore/`、改 scout run 里其它文件（含把本夹改名为 `D-{n}/`——那是 scout 的事）。
- 主循环**禁止**精读全文；只读 INDEX / 线索 / 检索摘要。
- **钉住本题**：变体只进本夹 `backlog.md`；不自行改题、不另开号。
- **文稿禁止出现 `I-{n}` / `D-{n}` / 条目编号**（夹名可能被外部重排）。标题只用短标题。
- 本流程面向 **subagent**：父代理须按 rule「subagent 模型」选 `model`（`inherit` / `auto` / `composer-2.5`，禁 `*-fast`）。
- 对照 scout 传入的 run `README.md`：**非目标 / Kill 条件** 命中 → **fail**；**算力上限** 里程碑合计超标 → **fail**。

## 落盘

父代理传入绝对路径 `temp/research-scout/<run-slug>/ideas/I-{n}/`（夹名仅路径用，正文不用）：

```text
ideas/I-{n}/
  idea.md         # 短条目；失败时在此标注
  related.md      # 相关论文
  base.md         # base 做了什么 + 相对更改
  novelty.md      # ★ 新颖性（先于可行性）
  reality.md      # 数据 / 指标 / 可复现（新颖性通过后）
  math.md         # 底层数学
  critic.md       # 数学他评（由 critic 子代理写）
  SPEC.md         # 机制 + 成功判据
  milestones.md   # 里程碑；M0 = 最便宜证伪
  backlog.md
  log.md
```

`n` 由 scout 分配；本流程 **不**改夹名、**不**自增号。

## 预算（默认）

| 参数 | 默认 | 含义 |
|------|------|------|
| **N** | 6 | 本条 **新** ingest 上限；不得超过 scout 传入的 `N_left`。本地已有合格 INDEX **不计入**。 |

## 主循环（顺序强制）

```
scout 传入：目标夹路径、假设陈述、轻量查重摘要、N_left、run README 路径（范围/非目标/kill/算力）
  → 建夹；idea.md 先钉住假设（状态: 探索中）
       违反 README 非目标或 Kill → idea.md 标 失败；回报 fail；结束
  → 1. 查相关论文（ingest 仅新计 N）→ related.md
       总结 base 做了什么、本题在 base 上改了什么 → base.md
  → 2. 写 novelty.md：最近邻、residual delta、PASS/FAIL
       FAIL → idea.md 标 失败（新颖性）；回报 fail；结束
  → 3. 写 reality.md：数据从哪来、主指标能否支撑 claim、能否公开复现
       FAIL → idea.md 标 失败（现实性）；回报 fail；结束
  → 4. 写 math.md：先 base 逻辑，再本题逻辑（细）
  → 5. Task(critic) 写 critic.md；FAIL → idea.md 标 失败（数学）；回报 fail；结束
       禁止自己给 math 判通过
  → 6. 细化机制 + 成功判据 → SPEC.md
  → 7. 写 milestones.md（M0 = 最便宜证伪；对照 README 算力；超标 fail）
  → 8. idea.md 标 可行；回报 keep；结束探索
```

需要机制细节才 Task(paper-ingest)；**仅新 ingest 计 N**；缓存命中不计。

### 开 paper-ingest subagent（强制）

用 Task，`subagent_type: generalPurpose`，**`model` 见 rule「subagent 模型」**（默认 `auto` / `composer-2.5`；主模型为 DeepSeek/Qwen 等时优先 `inherit`；禁 `*-fast`）。  
`run_in_background: false`（除非并行多篇且能合并结果）。

Prompt 须包含：

- 读并遵循 `.cursor/skills/paper-ingest/SKILL.md`
- 目标 arXiv/URL/slug
- 只写 `temp/papers/<slug>/`
- 回报：INDEX 路径 + `new` 或 `cache` + 可跟线索 + related 种子（勿贴全文）

### 开 critic subagent（强制，数学写完后）

用 Task，`subagent_type: generalPurpose`，**`model` 见 rule「subagent 模型」**（`inherit` / `auto` / `composer-2.5`，禁 `*-fast`）。

Prompt 须包含：

- 读并遵循 `.cursor/skills/research-scout/critic.md`
- 夹内绝对路径；只写 `critic.md`
- 回报：`PASS` 或 `FAIL` + 一句原因

### `idea.md` 格式

```markdown
# <短标题>
- 状态: 探索中 / 可行 / 失败
- 失败原因: （仅失败；非目标 / kill / 新颖性 / 现实性 / 数学 / 超预算 / 查重）
- 新颖性: PASS / FAIL
- 陈述: …
- 为何可能好: …
- 查重: 搜过什么 → 未见 / 有近邻（链接） / 已有强重叠
- 依据: INDEX 锚点 或 纯假设+检索
- 粗成本: 小 / 中 / 大
- 研究潜力: A+ / A / A- / B+ / B / B- / C
- 成功可能性: 0～1（本题能做成的确信程度；一位小数，如 0.6；失败可接近 0）
```

失败时保留已写出的 `related.md` / `base.md` / `novelty.md` / `reality.md` / `math.md` / `critic.md`；不写或删未完成的 `SPEC.md` / `milestones.md`。

### `base.md` 格式

```markdown
# Base 与更改
## Base 做了什么
…
## 在 Base 上改了什么
…
```

### `novelty.md` 格式

```markdown
# 新颖性
## 最近邻
…（1～3 篇，链到 related.md）
## 差在哪
…（residual delta；禁止只写「我们更高效/更通用」）
## 判定
PASS / FAIL：…
```

**FAIL** 若：强重叠；或实质是把已有方法 A 换到问题 B、没有机制增量。  
新颖性未过，**不要**写现实性、数学、卡时。

### `reality.md` 格式（仅新颖性 PASS 后）

```markdown
# 现实性
## 数据
从哪来、是否公开、有无许可/爬取/隐私障碍
## 指标
主指标是什么、能否直接支撑 claim（禁止用无关 proxy 凑数）
## 可复现
别人按 SPEC 能否在公开设定下复现（权重/数据/协议）
## 判定
PASS / FAIL：…
```

**FAIL** 若：关键数据拿不到或不能公开讨论；主指标量不到声称的东西；复现依赖不可分享的私有资产且 README 未允许。

### `math.md` 格式（写细；仅现实性 PASS 后）

```markdown
# 底层数学逻辑
## Base 的逻辑
…（符号、目标、推导、关键假设）
## 本题的逻辑
…（相对 base 改了哪一步；自洽性、可行性、合理性）
## 自检（非正式）
…（供 critic 攻击；本文件不得写「判定通过」）
```

新逻辑必须**逻辑自洽**且**数学上可行、合理**。是否通过以 `critic.md` 为准。

### `SPEC.md` 格式（仅 critic PASS 后）

```markdown
# <短标题>

- 粗成本: 小 / 中 / 大
- 研究潜力: A+ / A / A- / B+ / B / B- / C（本条复评）
- 成功可能性: 0～1（确信程度）

## 钉住的假设
…

## 机制
…

## 成功判据
- 基准: （对照谁，1～3 个）
- 数据: …
- 主指标: …
- 怎样算做成: （相对基准的可检验提升或定性门槛；禁止空泛「更好」）

## 查重
搜过什么 → 未见 / 有近邻（链到 related.md） / 已有强重叠

## 失败模式
…
```

### `milestones.md` 格式（仅 critic PASS 后）

- **M0（强制）**：最便宜的证伪实验——怎样用最少卡时证明这题不成立。不要一上来完整训练。
- 其后 M1… 才是逐步做正的路径。
- **小模型 / 单卡吃得下** → **5090 卡时**；**大模型** → **A100 卡时**；写选因。
- 主路径（含 M0）卡时加总，对照 run `README.md` 上限；超出 → 本条 **fail**（可行性），`idea.md` 写超预算。

```markdown
# 里程碑
## M0: 最便宜证伪
- 完成: …
- 做法: …
- 卡: 5090 或 A100（选因）
- 估算: <数> 卡时
## M1: <名>
- 完成: …
- 做法: …
- 卡: 5090 或 A100（选因）
- 估算: <数> 卡时
```

估算不含「无目的扫参」。

研究潜力对标（必填一档，勿用中间值）：

| 档 | 对标 |
|---|---|
| A+ / A / A- | A 档文章的上 / 中 / 下 |
| B+ / B / B- | B 档文章的上 / 中 / 下 |
| C | 普通论文（够写成一篇，够不上 A/B 档） |

成功可能性（必填）：`[0,1]`，表示**本题能做成的确信程度**（与研究潜力档独立；失败条也可填，通常接近 0）。critic FAIL 后须下调，不得维持高分。

`related.md` 每条近邻一行：论文 + **与本题差在哪**。列出检索词。

回报 scout（短）：`keep` 或 `fail`、短标题、失败门（非目标/kill/新颖性/现实性/数学/超预算/查重，若有）、研究潜力、成功可能性 0～1、本条**新** ingest 数。勿贴全文、勿回报编号标签。
