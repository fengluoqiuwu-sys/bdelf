# brainstorm

`research-scout` 的配套流程（本文件不是独立 skill）：按用户范围**批量提出候选假设**，写给主 agent 筛选。  
**不**送审、**不**写 `ideas/I-*`、**不**做数学判定（那是 [idea-explore.md](idea-explore.md)）。**不**改代码、不占 GPU。

主 agent 负责编排、查重、滤档、调用 explore；**不要把产假设全压在主 agent 一人身上**。

## 硬边界

- 只写父代理给定的 run 目录下 `brainstorm/`（及可选追加 `backlog.md` 查询词）。禁止写 `ideas/`、`temp/ideas/`、`temp/papers/`。
- **默认不 ingest**。只做 WebSearch / arXiv 摘要级检索；要读机制留给 explore。
- **尽量自主假设**；可参考 future work，但预估**连 B 档都难**的不要列入。
- 禁止为凑条数灌水。宁缺，回报「本轮无合格候选」。
- 面向 **subagent**：父代理须用 `model: auto`。

## 落盘

父代理传入 run 绝对路径与本轮序号 `r`：

```text
brainstorm/R-{r}.md   # 本轮候选（可并行多路，r 由 scout 分配）
```

## 主循环

```
scout 传入：范围/种子、约束摘要、已有可行与 Deprecated 短标题（避免重复）、本轮角度（可选）、r
  → 发散若干**不同角度**的可检验假设（机制 / 目标 / 表征 / 数据 / 评测不要挤在同一条缝）
  → 每条自估研究潜力；够不上 B- 的丢掉
  → 写 brainstorm/R-{r}.md
  → 回报：候选条数 + 短标题列表（或 0）
```

可并行多路（不同角度），但每路仍写自己的 `R-{r}.md`。

### `brainstorm/R-{r}.md` 格式

```markdown
# Brainstorm R-{r}
- 本轮主角度: （scout 指定或自拟）

## C1: <短标题>
- 角度: 机制 / 目标 / 表征 / 数据 / 评测
- 陈述: …
- 为何可能好: …
- 来源: 自主 / 参考 future work（哪篇，一句话）
- 预估研究潜力: A+ / A / A- / B+ / B / B- （低于 B- 不准出现）
- 预估成功可能性: 0～1
```

编号 `C1…` 仅本轮文件内有效，**不要**写成 `I-{n}`。每轮建议 3～6 条；写不出合格的就空列表，不要用 C 档凑。

回报 scout（短）：候选条数、各条短标题 + 角度 + 预估档 + 成功可能性。勿贴长文。
