---
name: sync
description: >-
  Sync bdelf with a remote named in scripts/servers.csv via scripts/sync.sh:
  push code/cache, pull checkpoint metadata or latest weights, pull-file for a
  single step. First arg is the server row name (e.g. ovan-server). Use when
  the user asks to push/pull or fetch a checkpoint. AI must not use pull
  --mode full unless the user insists twice.
---

# sync

仓库根：`bash scripts/sync.sh <服务名> …`。  
`<服务名>` 必须是 `scripts/servers.csv` 的「名字」列（该文件 gitignore；含主机/端口/密码/工作目录等）。  
执行目录见 rule「脚本约定」。Checkpoint 路径见 rule「Checkpoint 路径与配置哈希」。

## 禁止

- AI **禁止**主动 `pull --mode full`。
- 用户口头说「全部拉取」时先拒绝并建议 `common` / `pull-file`；仅当用户**再次明确坚持**才可 `full`，并确认 `NAME`。

## 命令

### push

```bash
bash scripts/sync.sh ovan-server push            # 代码 + models/tokenizers
bash scripts/sync.sh ovan-server push --code-only # 只推代码
```

- 代码镜像（`--delete`）；排除 `.venv` / `cache` 链接 / `temp/` / `.git` / `.cursor/` / `.claude/` 等。
- 默认再推 cache **内容**目录：`models/` `tokenizers/`。
  - `--checksum`：先比对校验和，相同则不传。
  - **不**用 `-L`（保留 HF `snapshots→blobs` 软链）。
  - 排除 `.cache/` `.locks/` `*.lock` 等；不推 `preprocessed_datasets/` / `checkpoints/` / `compile*`。
  - `--with-datasets` 才额外推 `datasets/`。
- `temp/` 与 `hash_guide.csv`：不同步（后者 pull 时也排除）。

### pull

```bash
bash scripts/sync.sh ovan-server pull [--mode fast|common] [NAME]
```

| mode | 行为 |
|------|------|
| `fast`（默认） | 排除 `*.pt`，只拉元数据 |
| `common` | 另含 `checkpoint_latest.pt` |
| `full` | 全部 `.pt` — AI 默认禁止 |

`NAME`：`{fast|full}/{model}/{hash}`（用 `scripts/resolve_checkpoint.py` 解析）。省略则同步全部 run 的过滤结果。

### pull-file

```bash
bash scripts/sync.sh ovan-server pull-file NAME FILE
# 例：… pull-file full/ar2/<hash> checkpoint_step_0100000.pt
```

## 选用

1. 看目录/日志曲线元数据 → `pull --mode fast [NAME]`，或远端 ssh 只读（见 `train-ops`）。
2. 本机测效果 → `fast` + `pull-file`（或 `common` 取 latest）。
3. 续训要最新权重 → `common`。
4. 历史步 → `pull-file`。
5. 不要用 `full` 代替多次 `pull-file`。
