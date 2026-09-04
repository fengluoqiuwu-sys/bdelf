# temp/ 布局

`temp/` 在 `.gitignore` 中。

| 路径 | 用途 |
|------|------|
| `temp/auto-research/<idea>/` | 实验记录（Claude 可只读；不开训） |
| `temp/idea/<idea>/` | 人工认可后的想法/规格 |
| `temp/papers/<name>/` | 论文（`paper/` + `INDEX.md`）与可选 `sources/` |
| `temp/research-scout/<run>/` | 自由探索找 idea（见 skill `research-scout`）；交付 `ideas.md` |
| `temp/agent/` | 作业登记（本机 `scheduler:local`；远端 `slurm`/`common`） |

- slug：短横线小写。建议有 `README.md`。
- scout 不写 `temp/idea/`；人筛选后再搬入。
- 勿把大 checkpoint 放进 `temp/`。
- Claude 远端操作需用户明确授权。
