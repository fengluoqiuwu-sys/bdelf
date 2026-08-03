---
description: 本地 temp/ 三分法（Claude 只读/写笔记，不开训）
---

# temp/ 布局

`temp/` 在 `.gitignore` 中。Claude **不得**使用远端 `temp/`（见 rule「禁止使用远端」）。

| 路径 | 用途 |
|------|------|
| `temp/auto-research/<idea>/` | 实验记录（Claude 可只读；不开训） |
| `temp/idea/<idea>/` | 人工认可后的想法/规格 |
| `temp/papers/<name>/` | 论文（`paper/` + `INDEX.md`）与可选 `sources/` |
| `temp/research-scout/<run>/` | 自由探索找 idea（见 skill `research-scout`）；交付 `ideas.md` |

- slug：短横线小写。建议有 `README.md`。
- scout 不写 `temp/idea/`；人筛选后再搬入。
- 勿把大 checkpoint 放进 `temp/`。
