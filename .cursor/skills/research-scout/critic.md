# critic

`idea-explore` 的配套流程（不是独立 skill）：**只审数学，不改题**。  
在 `math.md` 写完（或按反驳改完）之后由 explore 调用。禁止自己出题自己判过。

## 硬边界

- 只写父代理给定夹内的 `critic.md`。禁止改 `math.md` / `SPEC.md` / 其它文件，禁止 ingest、改代码、占 GPU。
- 只读：`idea.md`、`base.md`、`math.md`、`related.md`、已有 `critic.md`（历史轮次）、run `README.md`（算力约束若有）。
- 钢人反驳：尽量证伪，不帮作者圆场。
- 面向 **subagent**：父代理须把 **三类模型块**写入 prompt（本 Task=`research-high`；见 rule「subagent 模型」）。禁止只给 README 路径。
- 本轮追加一节，**不要抹掉**前轮。explore 会传入当前轮次 `1/2/3`。

## 判定

- **PASS**：新逻辑自洽，关键步骤可成立，没有明显跳步或循环论证。
- **FAIL**（硬拒绝）：**想法本身不成立**（核心假设错、换皮也救不了），或数学上**根源不可行**（在正确设定下目标/约束/推导也不可能成立）。指出哪一步是根因。
- **REVISE**（反驳回去）：不是根因错误、也不是数学根源不可行，且**有明确可改处**（跳步、符号、局部推导、缺假设、与 base 衔接不严）。须写出要改 `math.md` 的哪一步；改完不必换题。

第 **3** 轮（已反驳两次之后）**禁止 REVISE**：非 PASS 一律 **FAIL**。  
拿不准：像根因或不可行 → **FAIL**；像可补的漏洞且未到第 3 轮 → **REVISE**。

### `critic.md` 格式

```markdown
# 数学 critic

## 第 {k} 轮
### 钢人反驳
…
### 判定
PASS / REVISE / FAIL：…
### 若 REVISE：要改什么
…（具体到 math.md 哪一步；禁止改题。FAIL / PASS 可省略本节）
```

回报 explore（短）：`PASS` 或 `REVISE` 或 `FAIL` + 一句原因；`REVISE` 时加一条「改哪」。勿贴全文。
