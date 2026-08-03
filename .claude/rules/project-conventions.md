---
description: bdelf 目录结构、配置模式与编码原则
---

# bdelf 项目规范

## 目录（摘要）

```
config/{datasets,tokenizers,preprocess,models,train,generate}/
dataset/  tokenizer/  models/
scripts/  scripts/train/
slurm/
train.py  generate.py  hf_config.py
cache/
```

Claude：**只推理、不训练、不用远端**（见 rule「禁止训练」「禁止使用远端」）。

## 命名与配置

- 基类/配置类：`FL_` 前缀。
- 配置文件名与 `name` 一致；`prototype.yaml` 不实例化。
- Dataset：注册表；Tokenizer：`get_tokenizer`；Generate：`config/generate/<model>/{generate,eval}.yaml`。
- 非属性键进 `extra`；缺必需字段报错。
- `config/**/*.yaml` 行内中文注释与同构：见 rule「config YAML 注释与结构」。

## HuggingFace / 依赖

- 先 `import hf_config`，再导入 HF 相关包。
- 本机 `.venv`（见 rule「Python 虚拟环境」）。

## 注释与编码

- 注释主体中文；最小改动；不主动加未要求的测试/文档。

## Claude 可用 skill

| Skill | 用途 |
|-------|------|
| `generate` | 本机 `generate.py` 推理 |
| `research-scout` | 自由探索找 idea → `temp/research-scout/` |
| `paper-ingest` | 下载论文编 INDEX（供 scout subagent） |

训练 / 远端 / 同步属于 **Cursor**（`train`、`train-ops`、`sync-ovan-server`、`auto-train`、`vram-probe`），Claude 不要调用。
