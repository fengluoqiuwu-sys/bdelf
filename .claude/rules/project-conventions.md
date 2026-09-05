# bdelf 项目规范

## 目录（摘要）

```
config/{datasets,tokenizers,preprocess,models,train,generate}/
dataset/  tokenizer/  models/
scripts/  scripts/train/   # 辅助与训练启动 sh；common 用 launch-train.sh
scripts/eval/              # 离线评测启动 sh；launch-eval
slurm/                     # 遗留 gpu/tail；Slurm 作业走 sar（sbatch-*.sh 仅人用）
logs/<服务名>/<时间戳>/     # 作业 .out/.err/gpu.log（gitignore；pull 同步，push 不删远端）
train.py  generate.py  hf_config.py
cache/                     # gitignore
```

脚本与包装器细则见 rule「脚本约定」；checkpoint 路径见 rule「Checkpoint 路径与配置哈希」。  
Claude：**只推理、禁止训练与自动训练**；远端须用户明确授权（见 rule「禁止训练」「禁止自动训练」）。

## 命名与配置

- 基类/配置类：`FL_` 前缀（`FL_Dataset`、`FL_Tokenizer`…）。
- 配置文件名与 `name` 字段一致；`prototype.yaml` 仅模板、不实例化。
- Dataset：YAML + 实现类 + `register_dataset` → `get_dataset(name)`。
- Tokenizer：YAML → `get_tokenizer(name)`（无需注册）。
- Generate：`config/generate/<model>/{generate,eval}.yaml` → `get_generate(model, name)`；训练用 `eval`，`generate.py` 默认 `generate`。
- 非属性键（如 `_doc`）进 `extra`；缺 `_YAML_REQUIRED` 字段则报错退出。
- `config/**/*.yaml` 行内中文注释与同构：见 rule「config YAML 注释与结构」。

## HuggingFace / 依赖

- 先 `import hf_config`，再导入 `huggingface_hub` / `transformers` / `datasets`。
- 本机依赖：仓库 `.venv`（见 rule「Python 虚拟环境」）；`requirements.txt` 含 cu130 torch 源。

## 注释与编码

- 注释主体用**中文**（专有名词可英文）；改到的英文注释顺手改中文。
- 最小改动；复用 registry/config 加载；不主动加未要求的测试/文档。
- AI 改代码默认在 **`master`**（不抢锁）；须向前兼容、不影响其他模型；破坏性改动二次确认。
- Claude **不开** Task / subagent。

## Claude 可用 skill

| Skill | 用途 |
|-------|------|
| `generate` | 本机 `generate.py` 推理 |
| `research-scout` | 只读已有 `temp/research-scout/` |
| `paper-ingest` | 只读已有论文 INDEX |

训练 / 远端提交 / 同步 / 研究探索属于 **Cursor**（`train`、`train-ops`、`slurm-auto-run`、`sync`、`auto-train`、`vram-probe`、`idea-kickoff`）。未授权远端请求引导到 Cursor。
