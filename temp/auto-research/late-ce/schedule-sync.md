# LateCE 同步 ELF 论文日程（2026-08-06）

## 同步内容（与 `elf-cfg-100m-full.sh` 一致）

| 项 | 旧 late_ce（14350 / `c81ffbd037f6cd5d`） | 新 |
|----|------------------------------------------|----|
| warmup_ratio | 0.005（默认 full） | **0.1** |
| min_lr_ratio | 0.1（cosine 衰减） | **1.0**（warmup 后 constant LR） |
| target_tokens | 50B | **45.2B** |
| gen_eval_samples | 256（默认） | **32** |
| run | `full/late_ce/c81ffbd037f6cd5d` | `full/late_ce/87e6aac8af3ccd2e` |

优化器 / batch / 模型算法旋钮未改（仍 uniform t、无 decode、δ=0.2 hard）。

## 旧跑对照结论（是否还需进一步改算法）

同进度 vs ELF `57ef50375e85d826`（旧 ELF 日程，非新 paper schedule）@~440k：

| | eval_ppl | gen_ppl | uniq |
|---|----------|---------|------|
| ELF | ~1.4 | ~104 | ~353 |
| LateCE 旧 | ~20.5 | ~8.4 | ~45 |

- **日程不同不是主因**：旧 LateCE 与该 ELF 的 warmup/min_lr/target 在 config 里实际一致；落差来自算法配方。
- **本次先只对齐论文日程**，便于与新 ELF `4ab96e311b796009` 公平对照。
- **仍建议后续（若新日程下 @~200k+ 仍 uniq≪ELF 且 gen≪REF≈16.9）**再开变体：
  1. 降 `late_ce_weight`（如 0.1–0.5）或扫 δ∈{0.05,0.1,0.2}
  2. 考虑保留少量 ELF 式 decode 分支（变体 B）而非完全删除
  3. 拉 EMA 样本肉眼确认是否仍偏塌缩

本轮：**不**在未看新日程曲线前改 δ/weight/decode。
