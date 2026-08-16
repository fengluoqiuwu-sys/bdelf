---
name: research-scout
description: >-
  Explore a research scope or seed papers, recursively gather related work via
  paper-ingest subagents, and keep feasible ideas under
  temp/research-scout/<run>/ideas/I-{n}/ (failed explores renamed to D-{n}/).
  Free exploration (not tied to a specific codebase). Writes only markdown
  under temp/. Use when the user wants research scouting, idea mining,
  literature-gap hunting, or research-scout.
---

# research-scout

自由探索：从用户给的**范围或种子论文**出发，**尽量自主假设**，再查重留下**还不错、显得没做过/做得少**的 idea，写给人筛选。  
**不是**写论文综述；**不**交接 auto-train；**不**改代码、不占 GPU、不提交远端作业。

配合 skill `paper-ingest`（读全文/编 INDEX）与本目录 [brainstorm.md](brainstorm.md)、[idea-explore.md](idea-explore.md)、[critic.md](critic.md)；均须 Task subagent。  
**主 agent 不独自想 idea**：候选由 brainstorm 出；主循环只筛选、多样性、查重、送审。explore 先新颖性后可行性；数学由 critic 他评。

## 硬边界

- 只写 `temp/`：本 run 目录 + 通过 ingest 写的 `temp/papers/<slug>/`。
- 禁止：改仓库代码/`config/`、训练/占 GPU、远端作业、往 `temp/ideas/` 写（正式开题由人确认后走 `idea-kickoff`）。亦禁止写 `temp/idea/`、`temp/auto-research/`。
- 主循环**禁止**精读全文；只读 INDEX / 线索 / 检索摘要。
- 与当前仓库实现**解耦**：不要求 repo-novel；自由探索即可。

## 落盘

```text
temp/research-scout/<run-slug>/
  README.md       # 范围、非目标、kill、算力上限、N/K、停止原因
  backlog.md      # 论文与查询队列
  brainstorm/R-{r}.md  # 各轮候选假设（尚未送审）
  ideas.md        # 索引：可行区按 潜力×确信 降序；文末 Deprecated
  ideas/I-{n}/    # 新颖性+可行性通过的 idea（计入 K）；由 idea-explore 写入
  ideas/D-{n}/    # 探索失败（新颖性/现实性/数学/超预算/kill 等）；不计入 K
  log.md          # 短决策迹
```

`<run-slug>`：`YYYY-MM-DD-<短主题>`（如 `2026-08-03-diff-lm`）。

## 启动约束（缺则先问）

进入 brainstorm **之前**，`README.md` 必须有下列项。用户拉起时没给的，**停下来问用户**，写入后再开循环；禁止默填、禁止用实验室常识代替。

| 项 | 问什么 |
|---|---|
| 范围 | 主题 / 种子论文 / 不许碰的方向 |
| 非目标 | 明确不做（如不写综述、不训基础模型） |
| Kill 条件 | 什么情况下整条 idea 直接丢（如必须冻结某 backbone） |
| 算力上限 | 5090 卡时、A100 卡时（可只给一种；explore 超标则 fail） |

```markdown
# Scout run
- 范围: …
- 非目标: …
- Kill 条件: …
- 算力上限: 5090 … 卡时 / A100 … 卡时（未使用的写「无」）
- N / K: …
- 停止原因: （结束时再填）
```

## 预算（默认）

| 参数 | 默认 | 含义 |
|------|------|------|
| **N** | 48 | 本 run **新** ingest 上限。本地已有合格 INDEX 的论文（`temp/papers/<slug>/paper/INDEX.md` 缓存）**不计入**，含 idea-explore 只读缓存。 |
| **K** | 8 | 新开送审的目标上限（**不含** `D-*`）。并行在飞时最终可行 `I-*` **可以超过 K**；够了只等在飞跑完，**不再新开**。不够 K 且已无合格新 idea 也停新开，**禁止为凑数降质量**。 |

停止新开探索（任一即停新开；在飞的 `idea-explore` **跑完再结束 run**，不要杀掉；`D-*` 不计入）：
- `可行 I-* + 在飞送审 ≥ K`：条数已够（并行超 K 允许），只等剩下的跑完；或
- 可行条数 **< K** 但连续 2～3 轮加不出非重复、非变弱的新 idea（**到此停新开，不要硬凑低质量 idea**）；或
- backlog 空、brainstorm 连续 2～3 轮 0 合格候选、且无新假设；或
- 用户喊停；或
- **新 ingest** 用尽 N 且无纯检索/缓存可推进。

## 主循环

```
用户范围/种子
  → 约束不全 → **询问用户**，写入 README 后再继续
  → 建 run 目录，写 README / 空 ideas.md / 空 ideas/ / 空 brainstorm/ / backlog
  → loop:
       候选池空且 `可行 + 在飞 < K` → Task(brainstorm) 写 brainstorm/R-{r}.md（可并行多角度）
       主 agent **只筛选**：
         丢掉连 B 档都难的、与已有 I/D 重复的；
         **多样性**：按角度聚类（机制 / 目标 / 表征 / 数据 / 评测…），每簇最多送审 1 条；同质的记 log 不送
         轻量检索查重
       需要机制细节才 → Task(paper-ingest)（**仅新 ingest 计 N**；缓存命中不计）
       筛过值得送审 **且** `可行 I-* + 在飞 < K` → 分配 n，Task(idea-explore) 写入 ideas/I-{n}/（传入 README 约束；其**新** ingest 计入 N）
       接到 explore 返回：
         fail → 将 ideas/I-{n}/ 重命名为 ideas/D-{n}/；ideas.md 文末 Deprecated 收录并写明原因；不占用 K
         keep → 保留 ideas/I-{n}/；写入 ideas.md 可行区；占用 K（并行下最终可 > K）
       停新开条件满足 → 不再新开 brainstorm/explore，等在飞 explore 跑完（无合格新 idea 勿为凑 K 降质量）
  → 重排 ideas.md 可行区；README 写停止原因；把 ideas.md 指给用户，请人挑选后走 `idea-kickoff` 拷入 `temp/ideas/<name>/`
```

### 开 brainstorm subagent（强制，产候选）

候选假设**必须**由 brainstorm 出，scout 主 agent **禁止**自己编一批再送审。  
用 Task，`subagent_type: generalPurpose`，**`model` 只能是 `auto`**。  
`run_in_background: false`（除非并行多路角度且能合并）。

Prompt 须包含：

- 读并遵循 `.cursor/skills/research-scout/brainstorm.md`
- run 目录绝对路径、本轮 `r`、用户范围/种子
- 已有可行与 Deprecated 的短标题（避免重复）
- 可选本轮角度（如机制 / 目标 / 表征，便于并行发散）
- 只写 `brainstorm/R-{r}.md`；默认不 ingest
- 回报：候选条数、各条短标题 + 角度 + 预估档 + 成功可能性

候选入池后由**主 agent**做轻量查重、滤档与**多样性挑选**，通过的才送 `idea-explore`。同簇不连送。brainstorm 回报 0 条算一轮「加不出新 idea」。

### 开 idea-explore subagent（强制，送审时）

轻量查重且多样性挑选后决定送审的假设**必须**交给 `idea-explore`，scout **不**自己写夹内长文。  
用 Task，`subagent_type: generalPurpose`，**`model` 只能是 `auto`**。  
`run_in_background: false`（除非并行多条且能合并结果）。

Prompt 须包含：

- 读并遵循 `.cursor/skills/research-scout/idea-explore.md`
- 目标目录：`temp/research-scout/<run-slug>/ideas/I-{n}/`（绝对路径）
- 假设陈述 + 已做轻量查重摘要
- run `README.md` 绝对路径（范围 / 非目标 / kill / 算力上限）
- 本条 **新** ingest 上限与当前 `N_left`（本条默认上限 6，且不得超过 `N_left`；缓存命中不占）
- 只写该夹与 `temp/papers/`；文稿不要写 `I-{n}` 字样
- 回报：`keep` 或 `fail`、短标题、失败门、研究潜力、成功可能性 0～1、本条**新** ingest 数

接到返回后：

1. **`fail`**：`ideas/I-{n}/` **整夹重命名**为 `ideas/D-{n}/`。不占 idea 计数（K 只统计可行的 `I-*`）。写入 `ideas.md` 文末 **Deprecated**（链到该夹 `idea.md`，**写明失败原因**）。`n` 不复用。
2. **`keep`**：夹名保持 `ideas/I-{n}/`，占用 idea 计数，写入 `ideas.md` **可行区**（按 潜力×确信 插入排序）。若 `可行 + 在飞 < K` 才继续新开；否则只等在飞。

可并行多条，但剩余配额 `N_left` 约束同时条数。并行时仍先写入各自 `I-{n}/`，返回后再按上式改名或入索引。  
`可行 + 在飞 ≥ K` 后**禁止再开新的** idea-explore；已在飞的跑完即可。并行导致最终 `I-*` 超过 K **允许**，不必砍掉多出来的可行条。

### 开 paper-ingest subagent（强制）

用 Task，`subagent_type: generalPurpose`，**`model` 只能是 `auto`**（禁止 composer 或其它显式模型）。  
`run_in_background: false`（除非并行多篇且你能合并结果）。

Prompt 须包含：

- 读并遵循 `.cursor/skills/paper-ingest/SKILL.md`
- 目标 arXiv/URL/slug
- 只写 `temp/papers/<slug>/`
- 回报：INDEX 路径 + `new` 或 `cache` + 可跟线索 + related 种子（勿贴全文）

可并行多篇，但剩余配额 `N_left` 只约束**新** ingest；缓存命中不占并发额度。

### Idea 来源

1. **brainstorm（默认）**：凭空或从范围推；可参考 future work。预估**连 B 档都难**的不准进 `R-{r}.md`。
2. 主 agent **只筛选、查重、按角度去同质**；不要跳过 brainstorm 直接写 idea。每簇最多送审 1 条。
3. 自主假设枯竭、brainstorm 连续空轮 → 停新开，不要降到 C 去凑。

### `ideas/I-{n}/` 与 `D-{n}/`

`n` 从 1 起单调递增（每次送审占一个号，**不复用**）。探索过程写入 `ideas/I-{n}/`；返回后失败则改名为 `ideas/D-{n}/`。  
**K 只计仍叫 `I-*` 的夹**（未触 kill/非目标，且新颖性、现实性、critic 通过，未超算力）。并行收尾时 `I-*` 可以多于 K。  
`D-*` 仍进 `ideas.md`，但放在文末 **Deprecated**，并写明失败原因；不计入 K。

夹内文稿由 `idea-explore` 写，**正文不得含 `I-{n}` / `D-{n}`**（可能被外部重排）。

`ideas.md` 结构。可行区按 **潜力×成功可能性** 降序（映射：A+=7 … A=6、A-=5、B+=4、B=3、B-=2、C=1，再乘 0～1）。Deprecated 不参与排序权重。文首写一句请人筛选。

```markdown
# Ideas
请从「可行」挑选后走 skill `idea-kickoff` 拷入 `temp/ideas/<name>/`；本索引不是正式规格。

## 可行
- [短标题](ideas/I-{n}/idea.md) — 潜力 A- · 成功可能性 0.6 · 分 3.0
## Deprecated
- [短标题](ideas/D-{n}/idea.md) — 原因：…
```

每条 keep/fail 返回后更新索引；run 结束再按分数重排可行区。

研究潜力对标（可行区抄 `idea.md` 的档；必填一档，勿用中间值）。成功可能性为 0～1 的确信程度（抄 `idea.md`）：

| 档 | 对标 |
|---|---|
| A+ / A / A- | A 档文章的上 / 中 / 下 |
| B+ / B / B- | B 档文章的上 / 中 / 下 |
| C | 普通论文（够写成一篇，够不上 A/B 档） |

scout 主循环里轻量查重已撞车、尚未送审的假设：不必建夹，`log.md` 记一行即可。  
送审后失败：只改名为 `D-{n}/`，不要删夹。

## 触发

用户提到 research-scout、扫论文找 idea、文献缺口、从某篇/某主题发散探索时启用。  
约束不全先问再跑。轻量「只问一句有没有类似工作」不必建满 run 目录；一旦进入多轮发散再落盘。  
送审的每条由 scout **自行**先 `brainstorm` 再多样性筛选再 `idea-explore`。不要等用户指定某条 idea、也不要在主循环里独自编完假设再写长文。
