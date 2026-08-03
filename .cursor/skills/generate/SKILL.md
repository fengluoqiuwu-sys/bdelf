---
name: generate
description: >-
  Run bdelf generate.py on a local checkpoint to sample or continue text. Use
  when the user wants generation, completion, listing runs, or testing model
  output. Always on the local RTX 5080; never on the remote cluster.
---

# generate

本机推理入口（rule「本机计算约束」：不在远端跑；注意 GPU 互斥）。  
权重不足时先 skill `sync-ovan-server` 拉取，再生成。

## 选 checkpoint

| 参数 | 含义 |
|------|------|
| `--run <rel>` | `cache/checkpoints/<rel>/checkpoint_latest.pt`，`<rel>`=`{fast\|full}/{model}/{hash}` |
| `--checkpoint <path>` | 任意 `.pt` |
| （省略） | 扫全部 `checkpoint_latest.pt` 取 mtime 最新 |

```bash
.venv/bin/python generate.py --list-runs
.venv/bin/python scripts/resolve_checkpoint.py --model ar --config 100m-full \
  --dataset owt --preprocess default --generate eval   # 不知 hash 时
```

## 用法

```bash
.venv/bin/python generate.py
.venv/bin/python generate.py --run full/elf/19de90b094488c46
.venv/bin/python generate.py --checkpoint cache/checkpoints/full/ar/<hash>/checkpoint_step_0100000.pt
.venv/bin/python generate.py --generate generate          # 默认正式生成配置
.venv/bin/python generate.py --generate eval              # 与训练在线评测同配置
.venv/bin/python generate.py --num-samples 3 --prompt "Once upon a time"
.venv/bin/python generate.py --prompt-file prompt.txt --run full/ar/<hash>
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--generate` | `generate` | `config/generate/<model>/{generate,eval}.yaml` |
| `--num-tokens` | 1024 | 总序列长（含 prompt） |
| `--num-samples` | 1 | 独立样本数 |
| `--temperature` / `--top-k` | 配置文件 | 仅显式传入时覆盖 |
| `--seed` | 42 | |
| `--device` | cuda | |

## 注意

- 权重旁需有 `model_meta` / `config.json`。
- ELF 生成无条件：传 `--prompt` 会报错。
- 给 ELF 显式 `--temperature` 会从 argmax 切到 multinomial。
- 跨实验对比请显式 `--run`，勿依赖「最新 mtime」。
- **AI**：跑前确认处于本任务分支；结束后 `git switch master`。generate **不**占 `temp/local-workspace.lock/`（见 skill `train-ops` / `auto-train`）。
