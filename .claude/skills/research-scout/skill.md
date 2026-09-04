---
name: research-scout
description: 自由探索找 idea；Claude 可只读已有 run，不负责创建新 run
---

# research-scout

自由探索：从用户给的**范围或种子论文**出发，查重留下**还不错、显得没做过/做得少**的 idea，写给人筛选。  
**不**改代码、不占 GPU、不提交远端作业。

**Claude 范围**

- Claude 可**只读**已有 `temp/research-scout/<run>/` 目录（`README.md`、`ideas.md`、`ideas/I-{n}/`、`ideas/D-{n}/`）查看探索结果。
- Claude **不**负责创建新 run、不开 brainstorm/explore subagent、不写 scout 目录（由 Cursor 侧 skill 完成）。
- 用户要求探索新主题时：建议用户切到 Cursor 调用此 skill。

## 硬边界（Cursor）

- 只写 `temp/`：本 run 目录 + 通过 ingest 写的 `temp/papers/<slug>/`。
- 禁止：改仓库代码/`config/`、训练/占 GPU、远端作业、往 `temp/ideas/` 写（正式开题由人确认后走 `idea-kickoff`）。
- 主循环**禁止**精读全文；只读 INDEX / 线索 / 检索摘要。
- 与当前仓库实现**解耦**：不要求 repo-novel；自由探索即可。

## 落盘（Cursor）

```text
temp/research-scout/<run-slug>/
  README.md       # 范围、非目标、kill、算力上限、subagent 模型、N/K、停止原因
  backlog.md      # 论文与查询队列
  brainstorm/R-{r}.md  # 各轮候选假设（尚未送审）
  ideas.md        # 索引：可行区按 潜力×确信 降序；文末 Deprecated
  ideas/I-{n}/    # 新颖性+可行性通过的 idea（计入 K）
  ideas/D-{n}/    # 探索失败（新颖性/现实性/数学/超预算/kill 等）；不计入 K
  log.md          # 短决策迹
```

`<run-slug>`：`YYYY-MM-DD-<短主题>`（如 `2026-08-03-diff-lm`）。

## 预算（Cursor 默认）

| 参数 | 默认 | 含义 |
|------|------|------|
| **N** | 48 | 本 run **新** ingest 上限。缓存命中**不计入**。 |
| **K** | 8 | 新开送审的目标上限（**不含** `D-*`）。 |

## ideas.md 结构

可行区按 **潜力×成功可能性** 降序。文首写一句请人筛选。

```markdown
# Ideas
请从「可行」挑选后走 skill `idea-kickoff` 拷入 `temp/ideas/<name>/`；本索引不是正式规格。

## 可行
- [短标题](ideas/I-{n}/idea.md) — 潜力 A- · 成功可能性 0.6 · 分 3.0
## Deprecated
- [短标题](ideas/D-{n}/idea.md) — 原因：…
```

研究潜力对标：

| 档 | 对标 |
|---|---|
| A+ / A / A- | **A刊 / A会** 的上 / 中 / 下 |
| B+ / B / B- | **B刊 / B会** 的上 / 中 / 下 |
| C | 普通论文（够写成一篇，够不上 A/B 刊会） |

## 触发

用户提到 research-scout、扫论文找 idea、文献缺口、从某篇/某主题发散探索时，建议用户用 Cursor 调用此 skill。
