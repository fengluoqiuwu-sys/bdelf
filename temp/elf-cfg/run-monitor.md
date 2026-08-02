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

## 当前轨迹（修复前日志，仅供参考）
- ~108k：decode 在降；日志 gen_ppl 仍偏低（旧 ODE/双重 CFG），肉眼 96k 刚现破碎英文

## 下一步
- 续训 + 每 15min 唤醒：ssh 探查 → pull fast → 本机正确口径 generate
- 穿过 136k：看 uniq / 样例是否稳住，再决定早停或继续
