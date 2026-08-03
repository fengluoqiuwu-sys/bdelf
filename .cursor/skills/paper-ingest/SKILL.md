---
name: paper-ingest
description: >-
  Download a paper into temp/papers/<slug>/, build an AI-oriented INDEX.md
  (file map, mechanism sketch, limitations, related seeds, scout cues), and
  return paths plus short cues. Use when research-scout (or the user) needs a
  local retrievable paper asset. Cursor-only; intended for Task subagents on
  auto or composer models. Does not invent research ideas.
---

# paper-ingest

把**一篇**论文变成主循环可检索的本地资产。由 `research-scout` 用 Task subagent 调用；也可在用户明确要求「下载并编索引」时单独用。

**硬边界**

- 只写 `temp/papers/<slug>/`（及其中 `paper/`）。
- 禁止改代码、开训、远端作业、clone GitHub/`sources/`（除非用户明示）。
- **不产出完整 research idea**（那是 scout 的事）；INDEX 里只给「可跟线索」。
- 本 skill 面向 **subagent**：父代理须用 `model: auto` 或 `composer-2.5-fast`（及同族 composer）启动；不要用更重的主模型跑 ingest。

## 输入

父代理 / 用户提供其一即可：

- arXiv id（如 `2605.10938`）或 abs/pdf/html/e-print URL
- 已有本地目录（补 INDEX）
- 标题 + 足够唯一的检索词（先解析到 arXiv 再下）

可选：`slug`（短横线小写）；未给则从标题/id 生成。

## 步骤

1. **解析身份**：确定 title、arXiv id（若有）、目标 `temp/papers/<slug>/`。
2. **跳过条件**：若 `paper/INDEX.md` 已存在且含「局限/未做」「可跟线索」「文件地图」等必备块 → **不重下**，直接回报路径 + 从 INDEX 抽出的线索（**不计** scout 的 N）。
3. **下载**（优先可检索文本）：
   - 优先 arXiv HTML → 另存可读 `*.html` / 抽 `*.txt`
   - 有 source：可下 `e-print`（tar）并解开 tex；保留 `source.tar.gz` 可选
   - PDF 可作补充，但 INDEX 检索应以 txt/html/tex 为主
   - 工具：`curl`/`WebFetch`；大文件用 shell 下载到目标目录
4. **写 `paper/INDEX.md`**（中文为主，专有名词可英文），结构固定：

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

## 5. 给 scout 的可跟线索
- （5～10 条子弹级提示，不是完整 idea）
```

5. **回报父代理**（短）：`slug`、INDEX 路径、线索列表、related 种子 id。勿贴全文。

## 禁止

- 写 `temp/idea/`、`temp/research-scout/`、`temp/auto-research/`
- 长篇「论文笔记」代替 INDEX 结构
- 把 PDF 二进制内容读进对话（用 txt/html/tex）
