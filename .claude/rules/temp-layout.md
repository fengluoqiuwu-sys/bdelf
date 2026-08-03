---
description: 本地 temp/ 目录布局（auto-research / idea / papers）
---

# 本地 temp/ 布局

`temp/` 在 `.gitignore` 中。Claude **不得**使用远端（见 rule「禁止使用远端」）；本规则只约束本地 `temp/`。

## 目录约定

| 路径 | 用途 |
|------|------|
| `temp/auto-research/<idea>/` | AI **自动训练与优化**的记录与产物（实验笔记、监控、冒烟日志、生成样本等） |
| `temp/idea/<idea>/` | AI **读论文后提出的想法**（规格/思路文稿；对应 skill 稍后补充，先按此落盘） |
| `temp/papers/<name>/` | 下载的论文与相关代码 |

### `temp/papers/<name>/`

- `paper/`：论文正文（HTML，或已解压的 LaTeX 源）。过长时在同目录编制快速索引（如 `INDEX.md`），需要细节再回源文件。
- `sources/`：相关代码仓库克隆。

不要把大 checkpoint 放进 `temp/`（权重留存路径另定，勿再用 `cache/temp/`）。

## 命名

- `<idea>` / `<name>`：短横线小写 slug（如 `elf-cfg`、`cola-dlm`、`ar2`）。
- 每个 `auto-research` / `idea` 目录建议有 `README.md` 作为入口。
