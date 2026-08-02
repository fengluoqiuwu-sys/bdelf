# elf-cfg：ELF 复现 — 补回 CFG / SC-CFG

## 背景 / 诊断

基线：`master` 上提交 `cc150cf`「Align ELF training with official PyTorch (no CFG/SC-CFG)」
实现的 ELF-B 无 CFG 版。远端 run `elf-100m-full-muon` 训满 305176 步，
**生成质量先升后崩**（decode eval CE 收敛到 0.32，但 gen_ppl 在 ~13.6 万步见底 24.1
后爆涨到 300–560，最终 ~278）。这是典型的「解码 collapse」信号——模型退化到输出
局部最优/重复文本；decode CE 低（重建 ok）但生成样本多样性差。

论文 arXiv:2605.10938 中 ELF-B 达 Gen.PPL≈24 @ **32 步 SDE + SC-CFG=3**。无 CFG 恰恰
拿掉了论文达成该指标最核心的成分（Fig.4：CFG 是把 PPL 从高位压到 24 的关键；
Fig.5c：SDE 采样少步数下显著优于 ODE）。

**根因两个**：
1. 缺 CFG/SC-CFG（论文主贡献，图 4）。
2. 训练步数过冲到最优 gen_ppl 点之后（~13.6 万步为最佳，30 万步已过拟合退化）。

## 修复口径（本分支任务）

在现有 `models/elf/` 实现上**补回 SC-CFG**，严格对齐官方 Algorithm 3 / 4 与采样公式：

- **训练时 SC-CFG（denoise 分支）**：对每个 sample 采样 self-cond-cfg scale `w`，
  v-target 外推为 `v_cfg = v + (1 − 1/w)·(v_sc − v_no_sc)`，其中 v_sc 用 stop-grad 的自条件
  预测作条件、v_no_sc 用零自条件（对应 JAX `train_step.get_sc_guided_v`）。
- **网络输入**：`net(z, t, c, w, mode)` 额外接受 self-cond-cfg scale `w`，作为 in-context
  前缀 token（`num_self_cond_cfg_tokens`，对应官方 `build_context` 的 SC-CFG embedder）。
- **推理时 SC-CFG**（`num_self_cond_cfg_tokens > 0`）：**单前向**，把 w 作为 in-context
  前缀喂入网络（training-time CFG；对应官方 `_forward_sample_self_cond` 首支）。
  仅当无 scale token 时才用 inference-time `v_uncond + w·(v_cond − v_uncond)`。

## 目标

ELF-B (105M) 32 步 SDE + SC-CFG 达 gen_ppl ≈ 24，复现论文。

## 复现信息

- 论文全文已本地化：`temp/thirdparty/ELF_paper.txt`（115KB, 1948 行）
- 官方 JAX 实现：`temp/thirdparty/ELF/src/train_step.py`（Alg.3/4）与
  `.../utils/sampling_utils.py`（采样 CFG、sample_cfg_scale）
- 官方 PyTorch 分支参考：`https://github.com/lillian039/ELF/tree/pytorch_elf`（待 clone）