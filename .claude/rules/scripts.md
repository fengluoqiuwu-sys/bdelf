# 脚本约定

## 放置（摘要）

| 位置 | 内容 |
|------|------|
| 仓库根 | `train.py`、`generate.py`（Claude **不跑** `train.py`） |
| `scripts/` | 辅助脚本；`resolve_checkpoint.py`、`download_dataset.py` 等 |
| `scripts/train/` | 训练启动 sh（Claude **不执行**） |
| `slurm/` | 远端提交用（Claude **不使用**） |

## 执行目录

- 工作目录一律是**仓库根**。
- 写成：`.venv/bin/python scripts/foo.py`、`.venv/bin/python generate.py`。
- 禁止 `cd scripts && python foo.py`。

Claude 常用：`generate.py`、`scripts/resolve_checkpoint.py`。其余训练/同步脚本留给 Cursor。
