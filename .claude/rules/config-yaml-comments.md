---
description: config/**/*.yaml 行内中文注释与同功能文件结构一致
---

# config YAML 注释与结构

适用于 `config/` 下全部 YAML（datasets / tokenizers / preprocess / models / train / generate）。

## 行内注释（强制）

- **每个键值行**必须带行内单行注释：写在该行**右侧**，形式为 `key: value  # 中文说明`。
- 注释语言：**中文**（专有名词 / 缩写可保留英文）。
- 禁止把说明写在键上方的独立 `#` 行（分区横幅除外）。
- 禁止用 YAML 多行字符串（`>` / `|`）写说明。

分区可用 `# ===...===` 横幅（中文）；同功能家族内横幅文字必须一致。

## 同功能文件必须同构

同组文件除**参数取值**外必须完全一致：键集合/顺序、右侧中文注释、横幅与空行布局。

新增变体：先复制同组已有文件，只改 `name` 与参数值。

```yaml
name: full                 # 配置名，非 prototype 时与文件名一致
variant: full              # 训练变体：fast | full
target_tokens: 50000000000 # 数据 token 预算，用于推导 max_steps
compile: true              # 是否启用 torch.compile(Inductor)
```
