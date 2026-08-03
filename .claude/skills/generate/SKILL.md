---
name: generate
description: >-
  Generate text from a bdelf training checkpoint via generate.py. Load a model
  from cache/checkpoints/<run>/checkpoint_latest.pt (or a specific checkpoint),
  optionally continue from a prompt, and produce samples. Use when the user wants
  to generate/completion text, list run checkpoints, or test a model's output.
---

# generate

推理/生成入口：本机用 `.venv/bin/python generate.py`（见 rule「Python 虚拟环境」与「本机计算约束」；
注意本机 GPU 互斥）。Claude **禁止**操作远端（见 rule「禁止使用远端」）；checkpoint 须已在本机
`cache/checkpoints/` 下。

## 选 checkpoint

三选一，默认取最新的 `checkpoint_latest.pt`：

| 参数 | 含义 |
|------|------|
| `--run <rel>` | `cache/checkpoints/{fast\|full}/{model}/{hash}/checkpoint_latest.pt` |
| `--checkpoint <path>` | 显式指定任意 `.pt`（含历史快照） |
| （省略） | 扫 `cache/checkpoints/{fast,full}/*/*/checkpoint_latest.pt` 取 mtime 最新者 |

不确定有哪些 run：`.venv/bin/python generate.py --list-runs`。  
不知道 hash：用与 train 相同入参跑 `resolve_checkpoint.py`。

## 基本用法

```bash
.venv/bin/python generate.py                                    # 最新 checkpoint，1024 token 无条件生成
.venv/bin/python generate.py --run elf-100m-full                # 指定 run 的最新权重
.venv/bin/python generate.py --checkpoint cache/checkpoints/<run>/checkpoint_step_0100000.pt  # 指定步数
.venv/bin/python generate.py --generate generate                # 正式生成配置（默认）
.venv/bin/python generate.py --generate eval                    # 与训练在线评测同一套采样
.venv/bin/python generate.py --num-samples 3                    # 多个独立样本
.venv/bin/python generate.py --prompt "Once upon a time"        # 续写（自动前置 BOS）
.venv/bin/python generate.py --prompt-file prompt.txt --run ar-100m-full-muon
.venv/bin/python generate.py --device cuda:0                    # 默认 cuda（无卡则 cpu）
```

## 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--generate` | `generate` | `config/generate/<model>/{generate,eval}.yaml` |
| `--num-tokens` | 1024 | 总序列长（含 prompt 前缀） |
| `--num-samples` | 1 | 独立样本数 |
| `--prompt` / `--prompt-file` | 无 | 续写前缀；二者互斥，`--prompt-file` 读 UTF-8 文件 |
| `--temperature` | 配置文件 | 覆盖 generate 配置中的温度 |
| `--top-k` | 配置文件 | 覆盖 generate 配置中的 top-k |
| `--seed` | 42 | 采样种子 |
| `--device` | cuda | torch 设备 |
| `--list-runs` | - | 列出含 `checkpoint_latest.pt` 的 run 并退出 |

## 约束与注意事项

- 模型/权重由 checkpoint 自带 `model_meta`（或其同目录 `config.json`）恢复；缺任一报 `ValueError`。
- `--prompt` 只在模型 `generate` 支持 `prefix_tokens` 时可用；**ELF 生成无条件**，传 `--prompt` 会报错。
- `--temperature`/`--top-k` 仅在显式传入时覆盖 YAML；给 ELF 传 `--temperature 1.0` 会从 argmax 切到 multinomial，测试需留意采样差异。
- prefix（含 BOS）编码后必须**短于** `--num-tokens`，否则报错。
- 未显式选 run 时会挑 mtime 最新的 run；跨实验对比请用 `--run` 显式指定。