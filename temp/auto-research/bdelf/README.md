# auto-research: bdelf

## 目标
新对齐 BDELF（ELF 式 in-context + SC-CFG，无 AdaLN）100m-full 基线。

## 口径（重启 2026-08-03）
- 沿用用户先前确认：对照 ELF；明显问题停修续训；batch=16；2 GPU；允许并行
- **token 预算**：跟当前 `master` schedule **50B**（用户底层已改，不再用旧 20B）
- 模型：`bdelf` / `100m-full`；`owt` + `default` + `eval`
- run：`full/bdelf/940234d42df7ce7b`
- holder：`auto-train:bdelf`
- 分支：`bdelf` ← `master` @ 当前 tip

## 状态
- 重启中：冒烟 → push → sbatch
