---
name: paper-ingest
description: >-
  将一篇论文下载到 temp/papers/<slug>/，编写供 agent 检索的 INDEX.md（文件地图、
  机制、局限、related 种子、scout 线索）并短回报。在 research-scout 开 subagent
  读论文或用户要求下载编索引时使用。禁止写 idea、改代码、训练与远端。
---

# paper-ingest

把**一篇**论文变成主循环可检索的本地资产。由 `research-scout` 用 subagent 调用；也可在用户明确要求时单独用。

**硬边界**

- 只写 `temp/papers/<slug>/`（及其中 `paper/`）。
- 禁止改代码、训练、远端、clone GitHub/`sources/`（除非用户明示）。
- **不产出完整 research idea**；INDEX 只给「可跟线索」。
- 遵守 `.claude/rules`：`no-train`、`no-remote`。

## 输入

- arXiv id / abs/pdf/html/e-print URL
- 已有本地目录（补 INDEX）
- 标题 + 检索词（先定位再下）

可选 `slug`（短横线小写）。

## 步骤

1. 解析 title、arXiv id、目标 `temp/papers/<slug>/`。
2. 若 `paper/INDEX.md` 已合格（含局限/未做、可跟线索、文件地图）→ 不重下，直接回报（**不计** scout 的 N）。
3. 下载：优先 HTML/txt；有则解 e-print tex；PDF 仅补充。
4. 写 `paper/INDEX.md`（结构同下）。
5. 短回报：slug、INDEX 路径、线索、related 种子。勿贴全文。

### INDEX 结构

```markdown
# <短名> 论文本地索引

> **用途**：agent 检索；细节回源文件。
> **论文**：<全标题>
> **arXiv**：… · HTML：…
> **本地**：`temp/papers/<slug>/paper/`

## 0. 一句话中心论点
## 1. 本地文件地图
## 2. 核心机制（短）
## 3. 局限 / 未做 / future work
## 4. 相关工作种子
## 5. 给 scout 的可跟线索
```

## 禁止

- 写 `temp/idea/`、`temp/research-scout/`、`temp/auto-research/`
- 用长综述代替上述结构
