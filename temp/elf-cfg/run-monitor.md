# elf-cfg 复现 run 监控记录

## 远端 run
- run name `elf-cfg-100m-full-muon`，脚本 `slurm/full/elf-100m-full.slurm`
- 分支 `elf-cfg`；2026-08-02 起带推理/评测修复后续训
- 4× RTX 4090，全程 305176 步（20B token）

## SC-CFG 实现（对照官方 Algorithm 3/4）
- **训练**：两遍 stop-grad 前向外推 `v_target = v + (1−1/w)(v_cond−v_uncond)`，w 作 in-context 前缀
- **推理**：有 scale token → **单前向 + w**（勿做双前向外推）
- 详见 `infer-fix-2026-08-02.md`

## 在线 eval 口径（修复后）
- **32 步 SDE + SC-CFG=3 + logit_normal + argmax**
- 额外字段：`gen_uniq_mean`（token 唯一数均值）、`gen_nonempty_frac`
- **判据**：低 gen_ppl 且 `gen_uniq_mean` 极低 → 仍是 collapse（假好）；需 uniq 上升且样例可读

## 基线对照（elf-100m-full-muon，无 CFG）
- ~13.6 万步 gen_ppl 见底 ~24 → 末期崩到 300–560；早期假低来自 `/` 重复

## 当前轨迹
- job **13566** 自 104000 续训 → **FAILED**（NCCL watchdog 10min timeout @~118146）；磁盘 ckpt 仍停在 **112000**
- 已重提 **13572** 从 `checkpoint_latest` 再续；115999 评测数字已记在下方（resume 可能截断 log）
- **107999（修复后首条）**：`eval_ppl=5.80`，`gen_ppl=48.95`，`gen_uniq_mean=113.46`，`gen_nonempty_frac=1.0` → 非 collapse（旧 103999 的 gen_ppl≈6 作废）
- **111999**：`eval_ppl=4.65`，`gen_ppl=55.81`，`uniq=132.92`，`nonempty=1.0`（uniq 继续升；gen_ppl 略升仍正常）
- **115999**（13572 重跑）：`eval_ppl=3.88`，`gen_ppl=70.55`，`uniq=153.54`，`nonempty=1.0`
- **119999**：`eval_ppl=3.33`，`gen_ppl=78.52`，`uniq=169.23`，`nonempty=1.0`（已过 118k NCCL 点；`checkpoint_step_0120000.pt` 已落盘）
- **123999**：`eval_ppl=2.92`，`gen_ppl=86.57`，`uniq=183.67`，`nonempty=1.0`
- **127999**：`eval_ppl=2.60`，`gen_ppl=99.17`，`uniq=201.19`，`nonempty=1.0`
- **131999**：`eval_ppl=2.39`，`gen_ppl=115.05`，`uniq=216.81`，`nonempty=1.0`
- **135999（对照点）**：`eval_ppl=2.22`，`gen_ppl=114.78`，`uniq=226.89`，`nonempty=1.0` → 曾微降；无 CFG 基线同点≈24 **不可比**
- **139999**：`eval_ppl=2.09`，`gen_ppl=119.24`，`uniq=240.19`，`nonempty=1.0`
- **143999**：`eval_ppl=1.99`，`gen_ppl=117.46`，`uniq=247.84`，`nonempty=1.0`
- **147999**：`eval_ppl=1.90`，`gen_ppl=119.29`，`uniq=260.01`，`nonempty=1.0`
- **151999**：`eval_ppl=1.83`，`gen_ppl=119.46`，`uniq=268.49`，`nonempty=1.0` → gen_ppl 平台持续；uniq 仍升；下一里程碑 ~160k ckpt
- 本机 generate 需 fp32（整模 `.to(bf16)` 会炸 ELF embed 路径）
- 120k 抽检：破碎英文但非 collapse

## 决策
- **CONTINUE**：136k 未崩；看 140–160k 是否出现 gen_ppl 真正下行
- 早停条件：`uniq` 骤降 + 样例退化，或 gen_ppl 连续多点失控飙升且可读性变差
- 2026-08-02 ~16:40：用户临时 checkout 他分支 → **下一次唤醒改为 1h**，之后恢复 15min
