# potential

`idea-explore` 的配套流程（不是独立 skill）：**只评研究潜力档，不改题、不审数学**。  
在 **critic 最后一轮 PASS** 之后、写 SPEC 之前，由 explore 调用。禁止 explore 自己给档。

目的：数学通过后重评；独立 agent，防止 explore 对自己的文稿过于乐观。

## 硬边界

- 只写父代理给定夹内的 `potential.md`。禁止改 `idea.md` / `math.md` / `critic.md` / `SPEC.md` / 其它文件，禁止 ingest、改代码、占 GPU。
- 只读：`idea.md`、`base.md`、`novelty.md`、`reality.md`、`math.md`、`critic.md`（以**最后一轮 PASS** 为准）、`related.md`。不要读尚未存在的 SPEC / 里程碑。
- **忽略** idea.md / brainstorm 里任何「预估潜力」；当作没看见。
- 钢人压档：先找「为什么不够上一档」，不帮作者圆到 A。
- 面向 **subagent**：父代理须把 **三类模型块**写入 prompt（本 Task=`research`；见 rule「subagent 模型」）。禁止只给 README 路径。
- 数学未 PASS 时 **不准**调用本流程。

## 判定

必填一档，勿用中间值：

| 档 | 对标 |
|---|---|
| A+ / A / A- | **A刊 / A会** 的上 / 中 / 下 |
| B+ / B / B- | **B刊 / B会** 的上 / 中 / 下 |
| C | 普通论文（够写成一篇，够不上 A/B 刊会） |

critic PASS **不等于**潜力高。看 residual 相对最近邻是否撑得起该档（A = 能投 A刊/A会，B = 能投 B刊/B会）。拿不准 → 取较低档。

### `potential.md` 格式

```markdown
# 研究潜力（独立重评）
## 钢人（为何可能更低）
…
## 判定
档: A+ / A / A- / B+ / B / B- / C
理由: …
```

回报 explore（短）：`档` + 一句理由。勿贴全文。explore **原样抄**进 `idea.md` / `SPEC.md`，禁止改档。
