# auto-research：late-ce

> 规格：`temp/idea/late-ce/README.md`  
> 实现：`temp/idea/late-ce/IMPLEMENTATION.md`  
> 分支：`late-ce` · 模型：`late_ce`

## 本轮目标

远端 full 训练 LateCE-δ=0.2（均匀 \(t\)、无 decode 分支），对照 ELF 基线看 Gen.PPL–entropy。

## 开训配置

| 项 | 值 |
|----|-----|
| script | `scripts/train/late_ce-100m-full.sh` |
| run | `full/late_ce/c81ffbd037f6cd5d` |
| GPUs | 2（prototype 默认） |
| batch_size | 16（alloc：4090 budget≈21.52；global_bs=512；ws=2） |
| holder | `auto-train:late-ce` |

## 实验记录

（唤醒循环中追加）

## 状态
- 2026-08-04：冒烟通过，准备远端 full

## 日程同步（2026-08-06）
- 对齐 `elf-cfg-100m-full`：warmup=0.1 / constant LR / 45.2B / gen_eval=32
- 新 run：`full/late_ce/87e6aac8af3ccd2e`
- 旧 run 保留对照：`full/late_ce/c81ffbd037f6cd5d`（job 14350 将停）
- 分析见 `schedule-sync.md`
