---
name: generate
description: >-
  本机用 generate.py 从已有 checkpoint 生成文本。Claude 只允许推理、禁止训练与远端。
  在用户要生成/续写、列本机 checkpoint、或测试输出时使用。
---

# Generate

**硬约束**：只本机推理；不跑 `train.py`；不用 ssh / sync / Slurm（见 `.claude/rules/no-train.md`、`no-remote.md`）。

权重必须已在本机 `cache/checkpoints/`。若缺失：停止并请用户用 Cursor 的 `sync` 拉取后再继续。

## 何时用

- 从 checkpoint 采样 / 续写
- 列出本机 run 下的 `checkpoint_*.pt`
- 需要 resolve 本机已有路径时用 `scripts/resolve_checkpoint.py`

## 命令

工作目录：仓库根。Python：`.venv/bin/python`。

```bash
.venv/bin/python generate.py \
  --run {fast|full}/<model>/<config-hash> \
  --checkpoint latest \
  --generate-config generate \
  --prompt "Once upon a time" \
  --max-new 64 \
  --seed 0
```

| 参数 | 说明 |
|------|------|
| `--run` | `fast\|full/<model>/<hash>`，对应 `cache/checkpoints/<run>/` |
| `--checkpoint` | `latest` 或步数（如 `1000`） |
| `--generate-config` | `config/generate/<model>/` 下名字（默认 `generate`；评测常用 `eval`） |
| `--prompt` | 可选续写前缀 |
| `--max-new` | 覆盖配置中的生成长度 |
| `--seed` | 随机种子 |

定位本机已有 run：

```bash
.venv/bin/python scripts/resolve_checkpoint.py \
  --model ar --config 100m-full \
  --dataset owt --preprocess default --generate eval
```

## 配置

- 路径：`config/generate/<model>/{generate,eval,...}.yaml`
- 字段随模型而异；同模型内三文件同构（见 rule「config YAML 注释与结构」）

## 注意

- GPU 互斥：同时只跑一个占卡进程（见 rule「本机计算约束」）。
- 不要把大段生成结果写进 git；需要落盘可用 `temp/`。
