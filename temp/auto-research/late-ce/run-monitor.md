# run-monitor：late-ce

## 口径
- 分支：`late-ce`
- 模型：`late_ce` δ=0.2 hard / late / uniform t；变体 B（decode 0.2 + late CE weight 0.1）
- script：`scripts/train/late_ce-100m-full.sh`
- run：`full/late_ce/c81ffbd037f6cd5d`
- batch=16；2 GPU；holder=`auto-train:late-ce`
- 主指标：Gen.PPL（不以 decode/late_ce 单独早停）

## 2026-08-04 启动
- 本机冒烟 OK：`mse`/`late_ce` 正常打印；无 decode `ce`
- 下一步：push → remote_status → sbatch → 登记 → 唤醒

## sbatch 14340
- time: 2026-08-04T19:00:47+08:00
- name: late-ce-100m-full
- run: full/late_ce/c81ffbd037f6cd5d
- state: SUBMITTED
- holder: auto-train:late-ce

## wakeup-1 job 14340 FAILED
- 原因：`AutoConfig.from_pretrained(t5-small)` 未传 `cache_dir`；离线找不到 `cache/huggingface/hub`
- 修复：`tokenizer.py` / `train/eval.py` 传入 `cache/tokenizers/...`
- 已清 active/14340；准备重提

## sbatch 14350（重提）
- time: 2026-08-04T19:24:08+08:00
- fix: tokenizer/eval cache_dir offline
- run: full/late_ce/c81ffbd037f6cd5d

## 2026-08-06 同步 ELF 论文日程并重提
- 改 `scripts/train/late_ce-100m-full.sh`：与 elf-cfg 相同 --set
- 新 hash：`87e6aac8af3ccd2e`；旧 `c81ffbd037f6cd5d` 保留不删
- 分析：旧曲线相对 ELF 的落差主因是算法配方而非日程；本轮先对齐日程，算法旋钮暂不动（见 schedule-sync.md）
- 动作：scancel 14350 → 新 sbatch

## sbatch 14834（论文日程）
- time: 2026-08-06T09:04:09+08:00
- run: full/late_ce/87e6aac8af3ccd2e
- baseline kept: full/late_ce/c81ffbd037f6cd5d

## 2026-08-06 用户暂停 + 实现复盘 → 变体 B
- 用户要求暂停：scancel 14834，清 agent 登记，停唤醒
- 实现复盘发现 4 个问题（详见 idea/late-ce/IMPLEMENTATION.md）：
  1. mode 错位：LateCE CE 在 denoise mode 算、推理用 decode mode；mode_tokens 零梯度
  2. eval 假指标：forward 忽略 branch="decode"，旧 run eval_ppl≈20.5 是混合损失、与 ELF 不可比
  3. 无 decode corruption 训练
  4. weight=1.0 CE 压过 MSE → 塌缩（gen_ppl≈8.4 / uniq≈45 是假象）
- 旧 run `c81ffbd037f6cd5d` / `87e6aac8af3ccd2e` 标记为「变体 A + weight1.0 + eval 失真」历史留档
- 代码改为变体 B：decoder_prob=0.2 + late_ce_weight=0.1 + eval decode 分支修复
- YAML 变更 → 新 config-hash；重提与否待用户确认
