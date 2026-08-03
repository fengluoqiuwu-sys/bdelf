---
name: research-scout
description: >-
  Explore a research scope or seed papers, recursively gather related work via
  paper-ingest subagents, and list promising underexplored ideas under
  temp/research-scout/<run>/. Free exploration (not tied to this repo). Writes
  only markdown under temp/. Use when the user wants research scouting, idea
  mining, literature-gap hunting, or research-scout.
---

# research-scout

自由探索：从用户给的**范围或种子论文**出发，发散找**还不错、查重后显得没做过/做得少**的 idea，写给人筛选。  
**不是**写论文综述；**不**交接 auto-train；**不**改代码。

配合 skill `paper-ingest`（读全文/编 INDEX 必须丢给 subagent，避免占满主上下文）。

## 硬边界

- 只写 `temp/`：本 run 目录 + 通过 ingest 写的 `temp/papers/<slug>/`。
- 禁止：改仓库代码/`config/`、训练、远端、往 `temp/idea/` 写（正式 idea 由人搬入）。
- 主循环**禁止**精读全文；只读 INDEX / 线索 / 检索摘要。
- 与本仓库实现**解耦**：不要求 repo-novel；自由探索即可。

## 落盘

```text
temp/research-scout/<run-slug>/
  README.md    # 范围、种子、N/K、停止原因
  backlog.md   # 论文与查询队列
  ideas.md     # ★ 交付物
  log.md       # 短决策迹
```

`<run-slug>`：`YYYY-MM-DD-<短主题>`（如 `2026-08-03-diff-lm`）。

## 预算（默认）

| 参数 | 默认 | 含义 |
|------|------|------|
| **N** | 30 | 本 run **新** ingest 的论文上限（已有合格 INDEX **不计入**） |
| **K** | 12 | `ideas.md` 中保留条数达到后，若边际变弱则停 |

停止：`ideas ≥ K` 且连续 2～3 轮加不出非重复、非变弱的新 idea；或 backlog 空且无新假设；或用户喊停；或 N 用尽且无纯检索可推进。

## 主循环

```
用户范围/种子
  → 建 run 目录，写 README / 空 ideas / backlog
  → loop:
       提出或细化假设（可来自论文线索，也可凭空）
       轻量检索：这事有没有人做过？（WebSearch / arXiv）
       需要机制细节 → Task(paper-ingest)（计 N）
       读 INDEX 线索 → 更新 ideas.md / backlog / log
       评估边际与预算 → 停或继续
  → README 写停止原因；把 ideas.md 指给用户
```

### 开 paper-ingest subagent（强制）

用 Task，`subagent_type: generalPurpose`，**`model` 只能是 `auto` 或 `composer-2.5-fast`**（同族 composer 亦可）。  
`run_in_background: false`（除非并行多篇且你能合并结果）。

Prompt 须包含：

- 读并遵循 `.cursor/skills/paper-ingest/SKILL.md`
- 目标 arXiv/URL/slug
- 只写 `temp/papers/<slug>/`
- 回报：INDEX 路径 + 可跟线索 + related 种子（勿贴全文）

可并行多篇，但剩余配额 `N_left` 约束同时 ingest 数。

### Idea 来源

1. 论文局限 / 未做 / future work / related 空白  
2. **自主假设** → 再搜是否做过（不必每条都 ingest；只有要读机制时才占 N）

### `ideas.md` 条目格式

```markdown
### <短标题>
- 陈述: …
- 为何可能好: …
- 查重: 搜过什么 → 未见 / 有近邻（链接） / 已有强重叠
- 依据: INDEX 锚点 或 纯假设+检索
- 粗成本: 小 / 中 / 大
```

丢弃明显重复或查重已撞车的条目（可在 `log.md` 记一行原因，不必堆在 ideas 里）。

## 触发

用户提到 research-scout、扫论文找 idea、文献缺口、从某篇/某主题发散探索时启用。  
轻量「只问一句有没有类似工作」不必建满 run 目录；一旦进入多轮发散再落盘。
