---
name: idea-kickoff
description: >-
  After the user says a scout idea is feasible, copy it to temp/ideas/<name>/
  (shared folder for later stages; current stage lives in stage.md, not
  README.md) and run 开题 as three sequential subagent steps: literature
  survey, spec freeze, then 开题报告 (LaTeX PDF or markdown fallback). Use
  when the user approves an idea, says 开题, kickoff, or to move a feasible
  I-* into temp/ideas/. Does not train, occupy GPU, write a paper, or
  modify the scout source folder.
---

# idea-kickoff

人说某条 idea **可行**之后：把 scout 产物**拷贝**到 `temp/ideas/<name>/`（后续阶段仍用这一夹），再按固定顺序开题。当前阶段写 **`stage.md`**。  
**不是** scout / idea-explore；**不**改代码、不占 GPU、不写论文、不跑实验。

源夹通常是 `temp/research-scout/<run>/ideas/I-{n}/`（状态须为**可行**）。配合 skill `paper-ingest` 与本目录 [survey.md](survey.md)、[spec.md](spec.md)、[proposal.md](proposal.md)；均须 Task subagent。  
**主 agent 不独自写综述、规格或开题报告**：只拷贝、过闸、派 Task、更新 `stage.md`、回报。

## 硬边界

- 只写 `temp/ideas/<name>/`；补文献时可通过 ingest 写 `temp/papers/<slug>/`。
- **拷贝、不搬移**：禁止改、删、重命名 scout 的 `I-*` / `D-*`。
- 禁止：改仓库代码、训练/占 GPU、远端作业、写稿/审稿模拟、把工程冒烟当可行性通过。
- 源夹必须是**可行**（`idea.md` 状态为可行，且新颖性 / 现实性 / critic 均为 PASS，且有 `potential.md` 档）。`D-*` 或失败条 → **拒绝开题**。
- 开题文稿**禁止**写 `I-{n}` / `D-{n}`。
- **阶段只写 `stage.md`**：禁止把某一阶段的闸 / 产物 / 下一步写进 `README.md`。
- **禁止把开题捏成一次做完**：综述 → 规格 → 报告，顺序强制；前一步 FAIL 则停，不写后面的。
- 开任何 research / research-high / ingest Task **必须**把三类模型块写入 prompt 并逐层原样传递（见 rule「subagent 模型」；**含禁止 fast**）；禁 fast、禁默填。
- 最终判断（是否真开实验、改 claim、停题）仍归用户。

## 落盘

```text
temp/ideas/<name>/
  README.md       # 跨阶段身份（标题 / 陈述 / 来源指针）；不含当前阶段内容
  stage.md        # ★ 当前阶段 + 开题内部步骤
  SOURCE.md       # 拷贝来源；不写 I-{n} 进其它文稿
  idea.md …       # 自源夹原样拷贝（含 milestones.md = 实验路线权威）
  survey.md       # 综述（survey subagent）
  scope.md        # 规格（spec subagent）
  claims.md
  grounding.md
  risk.md
  protocol.md     # 实验协议（对齐 milestones；不授权执行）
  proposal/ 或 proposal.md   # 开题报告（report subagent）
```

`<name>`：用户给的短横线小写 slug；未给则从短标题生成。

## 主流程（编排）

```
用户点名某条可行 idea
  → 缺路径或不是「可行」→ 停下来问 / 拒绝
  → 解析 <name>；目标已存在 → 停下来问，禁止覆盖
  → cp -a 源夹 → temp/ideas/<name>/
  → 主 agent 只写 SOURCE.md + 薄 README.md + stage.md（当前=开题，步骤=综述）
  → 三类 subagent 模型：抄 scout README，缺则先问，写入 stage.md
  → 确认拷贝的 novelty / reality / critic 均为 PASS，且 `potential.md` 有档；缺一则停
  → Task(survey) 写 survey.md
       FAIL → stage.md 标失败；停；不写规格/报告
  → Task(spec) 写 scope / claims / grounding / risk / protocol
       FAIL → stage.md 标失败；停；不写报告
  → Task(report) 写开题报告并编译（或 md fallback）
  → 更新 stage.md（步骤=待人确认）
  → 回报：目录、当前步骤、报告路径；请人确认后再谈实验
```

拷贝用仓库根：

```bash
mkdir -p temp/ideas
cp -a <源夹绝对路径> temp/ideas/<name>
```

### 主 agent 可写的身份文件

`SOURCE.md`：拷贝自哪、时间、scout run README 路径（范围 / 非目标 / kill / **算力上限** / **三类 subagent 模型**）。源夹未改。三类模型若 scout README 没有 → **先问用户**，写入 `stage.md` 后再开 survey。

`README.md`（仅当尚无此文件）：标题、陈述、来源指针、「当前阶段见 stage.md」。禁止写闸、产物、下一步。已有则不要覆盖。

`stage.md`（每步结束后由主 agent 更新）：

```markdown
# 阶段
- 当前: 开题
- 步骤: 综述 / 规格 / 报告 / 待人确认 / 失败
- 状态: 进行中 / 待人确认 / 失败
- 更新: YYYY-MM-DD
- subagent 模型: research … / research-high … / ingest … / 禁止 fast: 一律禁止 `*-fast`（抄 scout 或人指定；向内传须带上本行）
## 本阶段产物
（只列已完成步骤的文件；综述未完不要预列报告）
## 闸 / 禁止
- 人确认后才可谈实验；禁止占 GPU、改 scout 源夹
## 下一步
- owner / 验收 / 成本上限 / 停止条件
## 历史
- YYYY-MM-DD 进入开题 / 综述 PASS / …
```

### 开 survey subagent（强制，综述）

综述**必须**交给 survey，主 agent **禁止**自己写 `survey.md` 或把综述揉进开题报告。  
用 Task，`subagent_type: generalPurpose`，**`model` 用已指定的 research**（见 rule「subagent 模型」；禁 `*-fast`）。

Prompt 须包含：

- 读并遵循 `.cursor/skills/idea-kickoff/survey.md`
- **三类 subagent 模型块**（本 Task 类型=`research` + 三值原文 + **禁止 fast**；见 rule「subagent 模型」；内层 ingest 必须再写入 prompt）
- 目标夹绝对路径；`SOURCE.md` 里 scout README 路径
- 只写该夹 `survey.md` 与 `temp/papers/`（ingest）
- 回报：`PASS` 或 `FAIL`、一句原因、新 ingest 数

接到 `FAIL`：更新 `stage.md`（步骤=失败），结束开题。

### 开 spec subagent（强制，仅综述 PASS 后）

规格与实验协议**必须**交给 spec，主 agent **禁止**自己写 `scope.md` / `claims.md` / `protocol.md` 等。  
用 Task，`subagent_type: generalPurpose`，**`model` 用已指定的 research**（见 rule「subagent 模型」；禁 `*-fast`）。

Prompt 须包含：

- 读并遵循 `.cursor/skills/idea-kickoff/spec.md`
- **三类 subagent 模型块**（本 Task 类型=`research` + 三值原文 + **禁止 fast**；见 rule「subagent 模型」）
- 目标夹绝对路径；scout README 路径（算力上限必须抄进本夹）
- 只写 scope / claims / grounding / risk / protocol
- 回报：`PASS` 或 `FAIL`、一句原因

接到 `FAIL`：更新 `stage.md`（步骤=失败），结束开题。

### 开 report subagent（强制，仅规格 PASS 后）

开题报告**必须**交给 report，主 agent **禁止**自己写 `proposal.tex` / `proposal.md`。  
用 Task，`subagent_type: generalPurpose`，**`model` 用已指定的 research**（见 rule「subagent 模型」；禁 `*-fast`）。

Prompt 须包含：

- 读并遵循 `.cursor/skills/idea-kickoff/proposal.md`
- **三类 subagent 模型块**（本 Task 类型=`research` + 三值原文 + **禁止 fast**；见 rule「subagent 模型」）
- 目标夹绝对路径
- 只写 `proposal/` 或 fallback 的 `proposal.md`
- 回报：`pdf` 或 `md`、路径、若 fallback 写明原因

### 开 paper-ingest（仅综述需要时，由 survey 去开）

主 agent **不要**为开题自己 ingest。综述缺口由 survey 按 [survey.md](survey.md) 开 Task。

## 明确不做

- 不重新跑 idea-explore；不把综述、规格、报告并成一次长文。
- 不提交 / 启动 GPU 作业（须用户另走 `compute-ops`）。
- 开题报告是短稿，**不是**论文；不审稿模拟、不把开题当「可以开训」。
- 实验阶段（`source/` / `result/`）见 skill `idea-experiment`；开题结束只把 `stage.md` 放到待人确认，不要在本 skill 里建实验目录或跑实验。

## 触发

用户说某 idea **可行**、要 **开题** / kickoff、或把 scout 的 `I-*` 做成正式规格时启用。  
未点名哪一条 → 先问。只讨论「这条看起来怎样」不必建 `temp/ideas/`。
