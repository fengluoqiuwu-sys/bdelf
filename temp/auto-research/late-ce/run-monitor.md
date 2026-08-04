# run-monitor：late-ce

## 口径
- 分支：`late-ce`
- 模型：`late_ce` δ=0.2 hard / late / uniform t；无 decode 分支
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
