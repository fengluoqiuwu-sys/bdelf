# survey（综述）

`idea-kickoff` 的配套流程（本文件不是独立 skill）：对已拷贝的可行 idea 写**文献综述**，写入目标夹 `survey.md`。  
**不是**规格冻结（那是 [spec.md](spec.md)）；**不是**开题报告（那是 [proposal.md](proposal.md)）。**不**改代码、不占 GPU。

配合 skill `paper-ingest`（读全文 / 编 INDEX 必须丢给 subagent）。

## 硬边界

- 只写父代理给定夹内的 `survey.md`，以及 ingest 的 `temp/papers/<slug>/`。
- 禁止写 `scope.md` / `claims.md` / `protocol.md` / `proposal*` / `stage.md` / `README.md`；禁止改拷贝来的 explore 文稿（含 `related.md` / `milestones.md`）。
- 禁止把综述写成开题报告或论文 related work 章。
- 文稿禁止出现 `I-{n}` / `D-{n}`。
- 面向 **subagent**：父代理须按 rule「subagent 模型」选 `model`（`inherit` / `auto` / `composer-2.5`，禁 `*-fast`）。
- 对照 scout README：**非目标 / Kill** 命中 → **FAIL**。

## 预算（默认）

| 参数 | 默认 | 含义 |
|------|------|------|
| **N** | 8 | 本步 **新** ingest 上限。本地已有合格 INDEX **不计入**。 |

需要机制细节才 Task(paper-ingest)；只读 INDEX / 检索摘要，禁止把 PDF 二进制读进对话。

## 主循环

```
父代理传入：目标夹、SOURCE / scout README 路径
  → 读 idea.md / related.md / base.md / novelty.md（以及已有 INDEX）
  → 脉络不够再 ingest（计 N）
  → 写 survey.md
  → 近邻已强重叠或只是方法 A 换问题 B → 判定 FAIL；否则 PASS
  → 回报父代理
```

### 开 paper-ingest subagent（强制，需全文时）

用 Task，`subagent_type: generalPurpose`，**`model` 见 rule「subagent 模型」**（默认 `auto` / `composer-2.5`；主模型为 DeepSeek/Qwen 等时优先 `inherit`；禁 `*-fast`）。只写 `temp/papers/<slug>/`。

### `survey.md` 格式

```markdown
# 综述
## 问题脉络
…（本题所在问题怎么演到现在；不是领域百科）
## 最近邻
…（1～5 篇，机制级；链 temp/papers/.../INDEX.md 或 related.md）
## 差分
…（residual；禁止只写「我们更高效/更通用」）
## 缺口
…（综述之后仍没被盖住的缝；本题打算站哪）
## 判定
PASS / FAIL：…
```

**FAIL** 若：强重叠；或实质是方法 A 套问题 B；或关键近邻读完后 novelty 不再成立。  
FAIL 时仍留下已写的综述，供人看原因。

回报父代理（短）：`PASS` 或 `FAIL`、一句原因、本步**新** ingest 数。勿贴全文。
