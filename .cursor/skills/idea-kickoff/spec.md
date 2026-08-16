# spec（规格冻结）

`idea-kickoff` 的配套流程（本文件不是独立 skill）：综述 **PASS** 之后，冻结范围、claim、实验协议。  
**不是**综述（那是 [survey.md](survey.md)）；**不是**开题报告（那是 [proposal.md](proposal.md)）。**不**改代码、不占 GPU。

## 硬边界

- 只写父代理给定夹内：`scope.md` / `claims.md` / `grounding.md` / `risk.md` / `protocol.md`。
- 禁止写 `survey.md` / `proposal*` / `stage.md` / `README.md`；禁止改 `milestones.md`（路线权威是拷贝来的该文件，本步只对齐、不另写一套里程碑）。
- 文稿禁止出现 `I-{n}` / `D-{n}`。
- 面向 **subagent**：父代理须把 **三类模型块**写入 prompt（本 Task=`research`；见 rule「subagent 模型」）。
- 不 ingest（文献缺口应已在综述解决；仍缺则 **FAIL** 并说明，让父代理决定是否退回综述）。

## 主循环

```
父代理传入：目标夹、scout README 路径（须含算力上限）、**三类 subagent 模型块**
  → 读 survey.md（须已 PASS）+ idea.md / SPEC.md / milestones.md / novelty / reality / critic
  → 写 grounding.md（从综述抽出最近邻 + residual，短）
  → 写 scope.md（问题、贡献 ≤3、非目标、kill；**把 scout 卡时上限抄进本文件**）
  → 写 claims.md（主 claim 为 hypothesis；做成/证伪对齐 SPEC 与 M0）
  → 写 risk.md
  → 写 protocol.md（对齐 milestones.md 与 SPEC.md；授权=无）
  → 里程碑合计卡时 > 已抄上限 → FAIL
  → 回报 PASS / FAIL
```

### `scope.md`

```markdown
# 范围
## 问题
…
## 贡献（≤3）
- …
## 非目标
…
## Kill
…
## 算力上限（冻结）
- 5090: … 卡时
- A100: … 卡时
（从 scout README 抄数字，禁止只写「见 scout README」）
## 听众 / 预期档
研究潜力抄 idea.md
```

### `claims.md`

```markdown
# 冻结的 claim
## 主 claim
…（可证伪；状态: hypothesis）
## 测量
主指标、数据、对照（对齐 SPEC）
## 关键假设
…
## 怎样算做成 / 证伪
做成: （SPEC 成功判据）
证伪: （对齐 milestones.md 的 M0；命中则 drop）
## 次要 claim
…（没有则「无」）
```

### `grounding.md`

从 `survey.md` 抽出，短；不要再写一篇综述。

```markdown
# 文献锚定
## 最近邻（1～3）
- … — 差在哪（链 survey.md）
## Residual
…
## 基准
…（与 SPEC 一致）
```

### `risk.md`

```markdown
# 风险
## 技术
… → 预案
## 数据 / 复现
… → 预案
## 新颖性被近邻吃掉
… → 预案（停或收窄）
## 算力
对照 scope.md 已冻上限；超则停
```

### `protocol.md`

**实验路线权威是同夹 `milestones.md`（explore 拷贝，本步不改）。** 本文件是开题体协议：对照、指标、授权、如何执行那些里程碑。

```markdown
# 实验协议（未授权执行）
- 授权: 无（须用户另走 compute-ops）
- 路线: 见 milestones.md（M0 最便宜证伪，其后 M1…；本文件不重复发明一套里程碑）
- 主对照: …
- 数据: …
- 主指标: …
- 卡时合计: … ≤ scope.md 算力上限
- 产物: 日志 / 表；sync pull；禁止远端交互式占卡
```

回报父代理（短）：`PASS` 或 `FAIL` + 一句原因。勿贴全文。
