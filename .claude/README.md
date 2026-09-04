# bdelf - Claude Code 配置

本目录为 Claude Code 配置。Cursor 配置在 `.cursor/`。

## 目录结构

```
.claude/
  rules/        # 规则文件（.md 格式）
  skills/       # 技能文件（.md 格式）
```

## 角色分工

### Claude Code（本配置）

- **本机推理**：`generate.py` 采样/续写
- **只读查看**：checkpoint、日志、配置、已有 temp/ 产物
- **代码阅读**：项目结构、模型实现
- **远端操作**：需用户明确授权（同步、查看日志、手动提交作业）
- **禁止**：训练、自动训练闭环、开 subagent

### Cursor（`.cursor/` 配置）

- **自动训练闭环**：auto-train（Claude 禁用）
- **研究探索**：research-scout / paper-ingest / idea-kickoff（subagent 创建）
- **计算管理**：vram-probe / compute-ops

## 规则（rules/）

| 文件 | 内容 |
|------|------|
| `python-venv.md` | 本机一律用 `.venv` |
| `no-train.md` | Claude 禁止训练 |
| `no-auto-train.md` | Claude 禁止自动训练闭环 |
| `compute-local.md` | 本机 RTX 5080 只推理 |
| `compute-remote-common.md` | 远端 common 主机计算约束 |
| `compute-remote-slurm.md` | 远端 Slurm 计算约束 |
| `checkpoint-layout.md` | {fast\|full}/{model}/{hash} 路径 |
| `config-yaml-comments.md` | YAML 行内中文注释与同构 |
| `project-conventions.md` | 目录结构与命名约定 |
| `scripts.md` | 脚本放置与执行目录 |
| `temp-layout.md` | temp/ 布局 |

## 技能（skills/）

| 技能 | 说明 |
|------|------|
| `generate` | 本机 `generate.py` 推理 |
| `paper-ingest` | 只读已有论文 INDEX |
| `research-scout` | 只读已有探索结果 |

## 用法

启动 Claude Code 后，规则和技能自动生效。Claude 会：

1. 遵守所有 rules/ 中的约束
2. 使用 skills/ 中定义的工作流
3. 将训练/远端相关请求引导到 Cursor

## 更新

从 Cursor 配置同步时：

```bash
# 这是示意，实际由用户或脚本执行
# Claude 不执行此操作
cp .cursor/rules/*.mdc .claude/rules/  # 转为 .md 并适配
cp .cursor/skills/*/SKILL.md .claude/skills/  # 适配 Claude 范围
```
