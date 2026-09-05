---
name: paper-ingest
description: 下载论文到 temp/papers/<slug>/，编 AI 用 INDEX.md；Claude 可只读，不用 subagent
---

# paper-ingest

把**一篇**论文变成主循环可检索的本地资产。Cursor 侧由 `research-scout` / `idea-explore` / `idea-kickoff` 用 Task subagent 调用。

**Claude 范围**

- Claude 可**只读**已有 `temp/papers/<slug>/paper/INDEX.md` 查阅论文信息。
- Claude **不**负责下载论文、不开 subagent、不写 INDEX（由 Cursor 侧 skill 完成）。
- 用户明确要求下载并编索引时：建议用户切到 Cursor 调用此 skill。

## 硬边界（Cursor）

- 只写 `temp/papers/<slug>/`（及其中 `paper/`）。
- 禁止改代码、开训/占 GPU、远端作业、clone GitHub/`sources/`（除非用户明示）。
- **不产出完整 research idea**（那是 scout / idea-explore 的事）；INDEX 里只给「可跟线索」。
- 不写 `temp/ideas/`、`temp/idea/`、`temp/research-scout/`、`temp/auto-research/`。

## 输入（Cursor）

提供其一即可：

- arXiv id（如 `2605.10938`）或 abs/pdf/html/e-print URL
- 已有本地目录（补 INDEX）
- 标题 + 足够唯一的检索词（先解析到 arXiv 再下）

可选：`slug`（短横线小写）；未给则从标题/id 生成。

## INDEX 结构（Cursor 写）

```markdown
# <短名> 论文本地索引

> **用途**：agent 检索；细节回源文件。
> **论文**：<全标题>
> **arXiv**：… · HTML：…
> **本地**：`temp/papers/<slug>/paper/`

## 0. 一句话中心论点
…

## 1. 本地文件地图
| 路径 | 内容 |
|------|------|
| … | … |

## 2. 核心机制（短）
…（公式/流程只保留 scout 决策所需）

## 3. 局限 / 未做 / future work
…（主矿区，尽量锚到章节）

## 4. 相关工作种子
- arXiv:… — 为何可能相关

## 5. 可跟线索（scout / idea-explore）
- （5～10 条子弹级提示，不是完整 idea）
```
