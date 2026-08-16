# critic

`idea-explore` 的配套流程（不是独立 skill）：**只审数学，不改题**。  
在 `math.md` 写完之后由 explore 调用。禁止自己出题自己判过。

## 硬边界

- 只写父代理给定夹内的 `critic.md`。禁止改 `math.md` / `SPEC.md` / 其它文件，禁止 ingest、改代码、占 GPU。
- 只读：`idea.md`、`base.md`、`math.md`、`related.md`、run `README.md`（算力约束若有）。
- 钢人反驳：尽量证伪，不帮作者圆场。
- 面向 **subagent**：父代理须用 `model: auto`。

## 判定

- **PASS**：新逻辑自洽，关键步骤可成立，没有明显跳步或循环论证。
- **FAIL**：指出**哪一步**不成立（公式、假设、从 base 到新逻辑的跳跃）。一句原因即可。

拿不准则 **FAIL**（宁可误杀）。

### `critic.md` 格式

```markdown
# 数学 critic
## 钢人反驳
…
## 判定
PASS / FAIL：…
```

回报 explore（短）：`PASS` 或 `FAIL` + 一句原因。勿贴全文。
