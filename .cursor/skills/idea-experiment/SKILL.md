---
name: idea-experiment
description: >-
  Experiment-stage layout for an idea already under temp/ideas/<name>/.
  Idea-specific code (e.g. probes) goes to ideas/{name}/source; data and
  results go to result. Shared training and checkpoints stay global. Use
  when the user starts 实验, experiment, probes, or after 开题确认. Does
  not require the AI to run experiments, occupy GPU, or train.
---

# idea-experiment

开题之后的**实验阶段规范**。同一夹 `temp/ideas/<name>/`，当前阶段写 **`stage.md`**。  
**实验不要求 AI 做**：不写训练闭环、不占 GPU、不提交远端作业。人要跑时另走 `compute-ops`。

## 硬边界

- idea 专门的代码（例如探针）写至 `ideas/{name}/source`，然后数据和结果放到 `result`。全局共用的训练之类的还有 checkpoint 这种还是全局。
- 落盘前缀与开题相同：`temp/ideas/<name>/source`、`temp/ideas/<name>/result`。
- 禁止把全局训练脚本、共享数据、checkpoint / 权重拷进本夹 `source/` 或 `result/`。`result` 里最多写全局 ckpt 的路径或哈希。
- 作业日志仍走全局 `logs/<服务名>/<时间戳>/`（见 `compute-ops`），不要改放到 `result/`。
- **阶段只写 `stage.md`**：禁止把实验闸 / 产物 / 下一步写进共用的 `README.md`。
- 不改 scout 源夹；不把开题文稿重写成实验报告。
- 未点名 idea、或夹不存在、或 `stage.md` 为开题失败 → 停下来问 / 拒绝。开题尚未到「待人确认」且用户未明确开始实验 → 先问。

## 落盘

```text
temp/ideas/<name>/
  README.md       # 跨阶段身份；本 skill 不改（除非尚无且必须补身份）
  stage.md        # 当前=实验
  …               # 开题产物保留
  experiment.md   # 本题实验规范（本阶段文稿，不是 README）
  source/         # 本题专用代码（探针、局部改动、评测脚本）
  result/         # 本题数据与结果（表、曲线、导出、局部产物）
```

全局（**不要**放进本夹）：仓库既有训练入口、共享数据、checkpoint。本模板不规定全局 ckpt 的具体目录（各项目自补）。

## 主流程

```
用户点名 idea / 说开始实验
  → 确认 temp/ideas/<name>/ 存在且开题非失败
  → 建 source/ 与 result/（已有则保留，不覆盖里面的代码或结果）
  → 写 experiment.md（规范实例化；已有则只补缺，不把结果写进 README）
  → 更新 stage.md：当前=实验；产物列 experiment.md、source/、result/
  → 回报路径；停止。除非用户另行明确要求，否则不写探针实现、不跑实验
```

### `experiment.md`

```markdown
# 实验规范
- 专用代码: `source/`（探针等；不要把全局训练拷进来）
- 数据与结果: `result/`
- 全局训练 / checkpoint: 仍用仓库全局位置；本夹只可引用路径或哈希
- 作业日志: `logs/<服务>/…`（compute-ops）
- 执行: 不要求 AI 跑；人要跑再走 compute-ops
```

### `stage.md`（本阶段）

只改阶段字段与本阶段产物，保留开题历史。

```markdown
- 当前: 实验
- 步骤: 规范
- 状态: 进行中
```

## 若用户后来要改代码 / 落结果

仍**不**自动开训。若人明确要写本题探针：只动 `source/`。本题产出的数据或表：只动 `result/`。改全局训练或 ckpt 布局须用户明确，且须在 **`master`** 上改，不放进 idea 夹。

## 明确不做

- 不跑实验、不占 GPU、不 sbatch / launch。
- 不把全局 trainer / checkpoint 复制进 `source/` 或 `result/`。
- 不把大权重塞进 `temp/`（全局 ckpt 保持全局）。
- 不写论文、不重做开题。

## 触发

用户说某 idea **开始实验**、要按实验规范落代码/结果、或提到本题探针 / `source/` / `result/` 时启用。  
只问「实验怎么排」而夹还不存在 → 先走 `idea-kickoff`，不要在这里开训。
