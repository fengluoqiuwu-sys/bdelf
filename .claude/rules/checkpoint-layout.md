---
description: checkpoint 路径为 {fast|full}/{model}/{hash}；禁止别名
---

# Checkpoint 路径与配置哈希

## 布局

```
cache/checkpoints/{fast|full}/{model}/{config-hash}/
```

- 权威相对路径：`{variant}/{model}/{config-hash}`。
- **禁止**软链、人类别名、旧扁平 run 名。
- `generate.py --run` 填 `fast|full/<model>/<hash>`。
- 定位（本机已有权重时）：

```bash
.venv/bin/python scripts/resolve_checkpoint.py \
  --model ar --config 100m-full \
  --dataset owt --preprocess default --generate eval
```

- 本地 `hash_guide.csv` 可只读查阅；Claude 不负责 sync。
- 硬件锁定 / 训练哈希细节以仓库 Cursor rule 为准；Claude 侧只需能正确定位本机 checkpoint。
