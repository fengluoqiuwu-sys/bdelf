---
name: research-scout
description: >-
  从用户给定范围或种子论文自由探索，递归读相关工作，列出有潜力且查重后
  显得未充分做过的 idea，写入 temp/research-scout/<run>/。只写 temp 下 markdown；
  读论文须开 subagent 走 paper-ingest。用户要找研究缺口/发散 idea 时使用。
---

# research-scout

自由探索找 idea，写给人筛选。不是综述，不交接训练，不改代码。

读全文 / 编 INDEX：开 **subagent**，令其遵循 skill `paper-ingest`（见 `.claude/skills/paper-ingest/SKILL.md`），主会话只收路径与线索。

**硬边界**：只写 `temp/`；不写 `temp/idea/`（人筛后自搬）；禁止训练与远端（`no-train` / `no-remote`）。与本仓库实现解耦。

## 落盘

```text
temp/research-scout/<run-slug>/
  README.md    # 范围、种子、N/K、停止原因
  backlog.md
  ideas.md     # ★ 交付
  log.md
```

`<run-slug>`：`YYYY-MM-DD-<短主题>`。

## 预算

| 参数 | 默认 | 含义 |
|------|------|------|
| **N** | 30 | 本 run **新** ingest 上限（已有合格 INDEX 不计） |
| **K** | 12 | ideas 保留条数达阈值后，边际变弱则停 |

停止：`ideas ≥ K` 且连续 2～3 轮无增量；或队列空；或用户停；或 N 尽且无纯检索可推。

## 主循环

1. 建 run 目录与 README / backlog / ideas / log。  
2. 提出假设（论文线索或自主想法）→ 轻量检索是否做过。  
3. 需机制细节 → **subagent + paper-ingest**（计 N）。主会话禁止贴全文。  
4. 更新 `ideas.md`；扩 backlog（related / 为验证假设而找的论文）。  
5. 评估边际与预算 → 停则写停止原因。

### `ideas.md` 条目

```markdown
### <短标题>
- 陈述: …
- 为何可能好: …
- 查重: 搜过什么 → 未见 / 有近邻（链接） / 已有强重叠
- 依据: INDEX 锚点 或 纯假设+检索
- 粗成本: 小 / 中 / 大
```

## 触发

research-scout、扫论文找 idea、文献缺口、从主题/种子发散时启用。单次查重问答不必建满目录。
