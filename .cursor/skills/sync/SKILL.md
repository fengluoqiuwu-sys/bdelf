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
登录/执行：系统 `ssh <服务名> 'cd ~/source/bdelf && …'`（「名字」即可 SSH）。  
`<服务名>` 必须是 `scripts/servers.csv` 的「名字」列（该文件 gitignore；含调度类型/工作目录/显卡额度等）。  
执行目录见 rule「脚本约定」。Checkpoint 路径见 rule「Checkpoint 路径与配置哈希」。

## 禁止

- AI **禁止**主动 `pull --mode full`。
- 用户口头说「全部拉取」时先拒绝并建议 `common` / `pull-file`；仅当用户**再次明确坚持**才可 `full`，并确认 `NAME`。

## 命令

### push

```bash
bash scripts/sync.sh ovan-server push            # 代码 + models/tokenizers
bash scripts/sync.sh ovan-server push --code-only # 只推代码
bash scripts/sync.sh ovan-server push --checkpoints full/odar/<hash> checkpoint_latest.pt
bash scripts/sync.sh ovan-server push \
  --checkpoints full/odar/<hash> checkpoint_step_0400000.pt \
  --checkpoints full/odar/<hash> checkpoint_latest.pt
```

- 代码镜像（`--delete`）；排除 `.venv` / `cache` 链接 / `temp/` / `.git` / `.cursor/` / `.claude/` 等。
- `logs/`：gitignore；**push 不传、且 `--delete` 不删远端**；由 **pull** 增量拉取。
- 默认再推 cache **内容**目录：`models/` `tokenizers/`。
  - `--checksum`：先比对校验和，相同则不传。
  - **不**用 `-L`（保留 HF `snapshots→blobs` 软链）。
  - 排除 `.cache/` `.locks/` `*.lock` 等；不推 `preprocessed_datasets/` / `compile*` / `eval/`。
  - `--with-datasets` 才额外推 `datasets/`。
- **checkpoints 默认不推**；`--checkpoints NAME FILE`（可重复）只推 `cache/checkpoints/NAME/FILE`（通常为某个 `.pt`；不做 `--delete`），并**同时增量推**对应 `cache/eval/<model>/<hash>/`（若本地有；供远端 eval 跳过已跑组）。`NAME`=`{fast|full}/{model}/{hash}`。与 `pull-file` 对称。`--code-only` 仍可配合。评测流程见 skill `eval`。
- `temp/` 与 `hash_guide.csv`：不同步（后者 push/pull 均排除）。

### pull

```bash
bash scripts/sync.sh ovan-server pull [--mode fast|common] [NAME]
```

| mode | 行为 |
|------|------|
| `fast`（默认） | 排除 `*.pt`，只拉元数据 |
| `common` | 另含 `checkpoint_latest.pt` |
| `full` | 全部 `.pt` — AI 默认禁止 |

另：每次 `pull` 都会增量同步远端 `logs/` → 本地（作业 `.out` / `.err` / `gpu.log`），以及 `cache/eval/` → 本地（评测产物；体量小）。

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
