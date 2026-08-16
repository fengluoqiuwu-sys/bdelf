---
name: git-commit
description: >-
  Create git commits only when the user asks. Conventional type plus Chinese
  subject; propose a split and wait if changes are unrelated. Use when the
  user asks to 提交, commit, or git commit. Does not push, amend, force, or
  skip hooks. Never commit secrets or gitignored paths.
---

# git-commit

人明确说**提交**之后，按本仓库格式写入 git。硬闸见 rule「Git 提交约束」。  
**不是** `push`；**不是** `template-update`（实例仓库用 `instance-git.sh`）。不改 `git config`，不用 `git -i`。

## 硬边界

- 没说提交 → 停。
- staged / 拟提交路径命中机密或 gitignore → 停并列出路径：`.env*`、`scripts/servers.csv`、`instances.csv`、`temp/`、`logs/`、`.venv/`、`cache/`、权重/checkpoint。
- 多段无关改动 → 先提案，**等同意**再提；不要一胡脸 `add`。
- 禁止 `--no-verify`、未要求的 `--amend` / force / `reset --hard`、`git add -A` / `git add .`。
- hook 失败：修好后 **新 commit**，不要 amend。

## 主流程

```
人说提交
  → git status / git diff（含 staged）/ git log（对齐语气）
  → 扫机密与 gitignore
  → 多逻辑 → 列出拟拆（标题 + 文件）并等待
  → 按次 git add <点名文件>，HEREDOC commit
  → git status 回报；默认不 push
```

并行先跑：

```bash
git status
git diff
git diff --staged
git log -8 --format='%s%n%b---'
```

无变更 → 不建空提交。只暂存本次该进这条 commit 的文件。

```bash
git commit -m "$(cat <<'EOF'
type(scope): 中文主语

为什么改。不要罗列文件。

EOF
)"
```

## 说明格式

| type | 何时 |
|------|------|
| `feat` | 新能力 |
| `fix` | 修缺陷 |
| `docs` | 只改文档 |
| `refactor` | 行为不变的结构调整 |
| `chore` | 脚手架、工具、杂项 |

有明确模块再写 `scope`（如 `skill`、`sync`、`slurm`）；否则 `feat: …`。对照最近 `git log`。主语祈使、中文；主题行尽量短。正文写 **why**。

```
feat(skill): 增加 git 提交规范

人说才提交；多逻辑先提案再拆；禁止把 servers.csv 等机密打进历史。
```

## 触发

用户说提交、commit、`git commit`、或要写 commit message 时启用。不要在改完代码后自行提交。
