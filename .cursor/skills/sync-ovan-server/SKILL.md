---
name: sync-ovan-server
description: >-
  Sync bdelf with ovan-server via scripts/sync-ovan-server.sh (push code/cache,
  pull checkpoints). Use when the user asks to push/pull to ovan-server, sync
  checkpoints, or fetch a specific checkpoint step file.
---

# sync-ovan-server

仓库根执行：`bash scripts/sync-ovan-server.sh`。远端：`ovan-server:~/source/bdelf`。

## 禁止（AI 必须遵守）

- **禁止使用 `--mode full`**。即使脚本支持、即使用户口语说「全部拉取」，也不要主动执行 `full`。
- 若用户**明确要求** `full`，先拒绝并说明体积风险，改用 `common` 或 `pull-file`；仅当用户再次明确坚持时，才可执行，并先确认目标 `NAME`。

## 命令

在仓库根执行（见 rule「辅助脚本」）。

### push

```bash
bash scripts/sync-ovan-server.sh push
```

- 代码镜像推送（`--delete`）；排除 `.venv` / `cache` / `temp/` / `.git` / `.cursor/` / `.claude/` 等
- `cache/` 增量推送；排除 `preprocessed_datasets/`、`checkpoints/`、`compile*` 等
- **`temp/` 在 push 与 pull 中均屏蔽**（远端 agent 任务登记，互不同步）
- `cache/checkpoints/hash_guide.csv` 仅本地哈希指引：整目录本就不 push；pull 时也排除该文件

### pull

```bash
bash scripts/sync-ovan-server.sh pull [--mode fast|common] [NAME]
```

| mode | 行为 |
|------|------|
| `fast`（默认） | 排除 `*.pt`，只拉元数据等小文件 |
| `common` | 同 fast，另同步 `checkpoint_latest.pt` |
| `full` | 全部 `.pt` — **AI 禁止使用** |

- `NAME`：可选，限定 `cache/checkpoints/{fast|full}/{model}/{hash}/`（用 `scripts/resolve_checkpoint.py` 解析）

- 默认拉全部 run 目录的范围内文件；有 `NAME` 时只同步该目录

### pull-file（单文件 / 指定步数）

```bash
bash scripts/sync-ovan-server.sh pull-file NAME FILE
```

例：

```bash
bash scripts/sync-ovan-server.sh pull-file ar2-300m-full-muon checkpoint_step_0100000.pt
```

## 选用规则

1. 只要列表/配置/目录结构 → `pull` 或 `pull --mode fast [NAME]`
2. **测试训练效果**：先 `pull --mode fast [NAME]`，再 `pull-file` 拉所需文件，在本机跑（不要在远端测）
3. 需要最新权重继续训/生成 → `pull --mode common [NAME]`
4. 需要某一历史步数 → `pull-file NAME checkpoint_step_XXXXXXX.pt`
5. 不要用 `full` 代替多次 `pull-file` 或 `common`

## 注意事项

- Checkpoint 布局：`cache/checkpoints/{fast|full}/{model}/{config-hash}/`（无别名；见 rule「Checkpoint 路径与配置哈希」）

- `NAME` 必须与远端目录名一致；不存在会 rsync 报 `No such file or directory`
- 不确定目录名时，先 `pull --mode fast`（或 ssh 列远端 `cache/checkpoints/`）再针对性拉取
