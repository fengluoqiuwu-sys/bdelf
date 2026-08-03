---
name: auto-train
description: >-
  Automate the full loop of training and tuning a model end to end: fork experiment
  branches, record under temp/auto-research/<idea>/, run fast local smoke-tests, push to
  ovan-server, submit full training via Slurm, then wake on a schedule to pull-fresh
  results, evaluate, decide whether to keep training, adjust on the same branch or
  pivot to a new direction, and finally summarize. Trigger only when the user
  explicitly says "自动执行" / "auto" AND explicitly authorizes execution. Not for
  one-off manual runs; those belong to train-ops / sync-ovan-server.
---

# auto-train

自动训练 + 自动优化的完整闭环 skill。仅在用户**明确说"请自动执行"并授权**后启动。

配合 rule「本机/远端计算」「temp/ 布局」、skill `train-ops` / `sync-ovan-server` / `train` / `generate` / `vram-probe`。  
本 skill 只编排闭环与决策；具体命令去读对应 skill。

## 远端只读探查（优先于 pull）

看目录、config、登记、日志末尾时：**优先 ssh 只读**（命令见 skill `train-ops`），不要为看一眼就 pull。  
勿 ssh 读大 `.pt`；需要权重用 `pull-file` / `pull --mode common`。

## 触发与授权

1. 用户描述任务（可能还提到某个架构/思路），并说"请自动执行"。
2. **必须**在下一条回复里先问"是否执行"，得到明确确认后才动。
3. 确认执行时，把后续**所有可能运行的一次性指令**（建分支、打 tag、起后台调度、首次 push、
   sbatch 等）合并成**一个待批准块**一次给出，让用户一次性授权，防止其离开后逐条授权卡住。
   - 构建鉴权：提交命令尽量合并为一条 `&&` / 一个脚本；初始化、登记等放同一块。
   - 若某条被拒，停下向用户说明，不要擅自更换等价命令绕过。
4. **一次性授权即覆盖整个自动训练循环，不设边界**：得到"批准"后，整个循环（ssh 只读探查、
   push/sbatch、开新分支、删本 agent 登记 run 的 checkpoint、换向 fork、sleep 唤醒、连不上重试等）
   均视为已被授权，自动执行，不再逐条打断用户。
   - 唯一**绝对底线**（不因授权而放宽，见「边界」）：不删/不动**非本 agent 登记范围**他人的
     job 与 checkpoint；不执行 `pull --mode full`；不 push 非 full 脚本。这几条即便已"全权"
     也不做，如确有必要先停下来向用户确认。

## 分支与目录约定

本地 `temp/` 三分法见 rule「本地 temp/ 布局」：`auto-research/`（本 skill 实验记录）、
`idea/`（读论文提出的想法规格）、`papers/`（论文与克隆代码）。远端 `temp/agent/` 仅 job 登记。

- **基线分支：`master`**（仓库当前只有 master，无 main）。"fork" = `git branch`/`git switch` 切分支，不是 GitHub fork。
- 每个任务从 `master` fork 出**想法根分支** `<idea>`（如 `ar2`），再按需派生子分支。
- **继承链原则（关键）**：派生子分支 / 换向分支时，一律从**当前携带相关代码实现的分支** fork，
  靠 git 祖先链继承已写好的代码，**绝不回基线从零恢复**——避免重写代码浪费时间、引入错误。
  只对真正全新的、与现有实现无关的想法才直接从 `master` fork。
- **开新分支前重读（关键）**：每产生一个新分支（fork 根分支、架构调整 `-<variant>`、换向）之前，
  **必须重新通读一遍相关思路与实现**，再做决定：
  - 读 `temp/idea/<idea>/`（若有）与 `temp/auto-research/<idea>/` 下的记录（`README.md`、各实验/调整 `.md`、放弃原因）——现在在哪一步、
    父节点是谁、之前试过什么、为何放弃/调整；相关论文见 `temp/papers/<name>/`；
  - 读当前分支的实际实现（模型/配置/脚本）：`git log --graph --oneline` 确认祖先链与当前 head；
  - **确认已完全理解**思路与现状后再 fork 新分支，不要把"调参/修 bug"误判成"该开新分支"、
    也不要在没读懂现状的情况下派生新方向。
  - 目的：保证每个新分支是在正确理解下产生的，避免在错误理解上层层叠加、浪费训练与时间。
- **分支粒度（hybrid），分两类新分支**：
  - **架构变体**（如改 embedding 格式、改注意力结构等，属同一思路的演进）→ 从**当前分支** fork
    新分支 `<idea>-<variant>`（继承现有实现，在此基础上改），记录在**原 `temp/auto-research/<idea>/`** 下
    （新增一个 `<variant>.md`）。
  - **全新想法**（与现有实现无关、独立方向）→ 从 `master` fork 新根分支，并**新建独立**
    `temp/auto-research/<newidea>/`（放弃原因写回原 `temp/auto-research/<idea>/`）。不在原思路目录里混入新想法记录。
  - 同思路内纯参数微调 → **留在当前分支**，用 `temp/auto-research/<idea>/<实验>.md` + 带命名 run/config 区分，
    不为此开分支（避免分支爆炸）。
- 维护分支拓扑可追溯：`git log --graph --oneline` 看祖先链；每个分支在 `temp/auto-research/<idea>/` 下记录
  父分支名、改动内容与原因，保证任何时刻能从某分支恢复现场继续。
- 每个自动训练思路在 **本地 `temp/auto-research/<idea>/`** 下建目录记录：
  - `README.md`：本轮自动研究目标/口径（可链到 `temp/idea/<idea>/README.md` 规格，如 `temp/idea/ar2/README.md`）。
  - 每个调整/子实验一个 `.md`：写明改动内容、原因、父节点、结果数据。
  - `SUMMARY.md`（只在完成后写）：整个思路的结论与最终建议。
- **换向/停止训练时的权重留存**（见第 10 步）：需保留一份最新权重时，**不放进 `temp/`**，
  **也不再使用 `cache/temp/`**；checkpoint 在 `cache/checkpoints/{fast|full}/{model}/{hash}/`（见 rule「Checkpoint 路径与配置哈希」），并在
  `temp/auto-research/<idea>/` 笔记里写明 run 名与用途。
- 远端 `temp/` 与本地 `temp/` **互不同步**：远端 `~/source/bdelf/temp/agent/` 只做 AI job 登记
  （见 train-ops）；自动训练记录放本地 `temp/auto-research/<idea>/`。

## 主流程

```
0. 授权确认（见上）
1. fork 想法根分支（或从当前分支 fork 新变体，继承已有实现）
2. 在 temp/auto-research/<idea>/README.md 记录口径
3. 实现思路
4. 本地验证：fast 冒烟（起训练看到首批 loss 正常、2–3 分钟后停）+ generate/ppl 跑通
4.6 显存探针（强制，见「VRAM 探针」）：改动影响显存时 → push → vram-probe → 填 alloc.md
5. push 到 ovan-server（sync 脚本）→ 按表+global_bs 选型 → train-ops 登记互斥 → sbatch full
   → 起「5 分钟首次唤醒」后台调度（确认拉起）
6. 唤醒循环：5m → 15m → 30m → 此后每 30m（见「唤醒调度」）
   ├ 7a 决定继续 → 回 6
   ├ 7b 需调整 → 8
   ├ 7c 已完成 → 11
   └ 7d 同 bug 卡住（见「卡住即停」）→ 11，勿空烧 token
8. 停止训练：本分支可修 → 9；需换架构 → 10；修不动 →「卡住即停」→ 11
9. 本分支修改 → commit →（若影响显存再跑 4.6）→ 删远端该 run checkpoint → push 重跑 → 回 4/5
10. 架构调整 / 换向 → 从当前分支 fork 新分支继承代码 → 删远端旧 run checkpoint
    → temp 记放弃原因/调整 → 回 3（实现调整；再训前必过 4.6）
11. 完成 / 放弃 → 写 temp/auto-research/<idea>/SUMMARY.md → 结束并总结
```

### 步骤详述

**1. fork 想法根分支**

开分支前，先按「开新分支前重读」约定重新通读思路记录与当前实现（见「分支与目录约定」）。

```bash
git switch master && git switch -c <idea>
```

换向 / 架构调整时，**不要回 master**，从当前携带实现的分支 fork 新分支继承代码（见第 10 步）。

**3. 实现思路**

按项目规范复写模型/配置；复用 registry / config 加载逻辑，不重复造轮子。改动尽量最小。

**4. 本地验证（5080 / fast）—— 只做冒烟，不跑满**

- 本机 Python 用 `.venv/bin/python`（见 rule「Python 虚拟环境」、skill train / generate）。
- `fast` **仅用于验证改动能跑通**，不是正式训练：起训练后观察到**首批训练步 loss 正常打印、
  无报错 / 崩溃 / segfault** 即可，通常 **2–3 分钟**内确认后就**主动停掉该进程**（kill），
  不要让 fast 跑满整个 token 预算（正式训练只在远端 full 跑）。
- 本机 GPU 互斥，一次一个进程（见 train-ops）。
- 验证通过后**清理本机中间产物**（快照/调试文件/临时 run 的 checkpoint 不提交），只保留有意义的改动。

**4.5 强制提交（推送到远端前必做）**

推送 `scripts/sync-ovan-server.sh push` 之前，**必须先提交到 git，不允许有未保留的内容**：

```bash
git status
git add <相关文件>      # 或 git add -A（用前用 git status 确认无夹带）
git commit -m "<语义化描述>"
```

- `git push` 推送的是 git **commit**；`scripts/sync-ovan-server.sh push` 推送的是**工作区文件**。若工作区有
  未提交的改动，远端拿到的是无法从 git 恢复的环境——这是不允许的。
- 本地验证通过后、确认改动可用即提交；提交信息写清改动内容与目的（如 `ar2: change anchor embedding format`）。
- 提交后 `git status` 必须**干净**（无未跟踪/未提交变体）再进入第 5 步。
- 思路/实验记录（`temp/auto-research/<idea>/*.md`）属本地记录、不同步；`temp/` 在 `.gitignore` 中，版本化时用
  `git add -f temp/auto-research/<idea>/...`。是否提交不影响远端 push。

**4.6 VRAM 探针（显存相关改动后强制）**

凡改动会影响显存占用（模型结构、精度、序列长 / chunk、EMA、优化器状态规模等），在提交 full **之前**必须：

1. 工作区已提交且干净（4.5）；`bash scripts/sync-ovan-server.sh push`
2. 按 skill **`vram-probe`** + **`train-ops`**：`remote_status` → 登记互斥 → `bash slurm/sbatch-vram-probe.sh …`
3. 读日志；把各档 **`alloc_peak_GiB`**（及 `oom`）填入 **`temp/vram-probe/alloc.md`** 对应行
4. **不要**把探针结论写回 recipe 默认 YAML；可在 `temp/auto-research/<idea>/` 记一笔链接到该表
5. **禁止**未填/过期表直接 `sbatch-train`

仅改 loss 日志文案等明显不影响显存的改动可跳过本步；有疑虑时宁可跑探针。

**开训选型**（步骤 5）：查 `alloc.md` 该模型各列，结合**当前目标卡** `total−2` 与本次 `global_batch_size` / `world_size`，取最大安全且整除的 micro-batch（公式见 skill `vram-probe`）。默认 YAML 已是该值则不必 `--set`；否则专用 `scripts/train/<name>.sh` 加 `--set batch.batch_size=…`。

**5. 推送与远端训练**

```text
- [前置] 工作区已干净（第 4.5 步已提交，git status 无未提交改动）
- [前置] 若本轮改动影响显存：第 4.6 步已完成，alloc.md 对应行已更新
- [前置] 已按表 + 当前卡 + global_bs 定好 batch_size（必要时 scripts/train 含 --set）
- bash scripts/sync-ovan-server.sh push
- 确认 `scripts/train/<name>.sh` 为 full 配置（禁止 preprocess）
- bash scripts/remote_status.sh   # 强制：GPU / 队列 / agent 登记
- 确认无未结束 AI job（或已 scancel 自己的）且有足够 AVAIL
- ssh 后 `bash slurm/sbatch-train.sh <name>`（可选 `--name JOB_NAME`）
- 写 current.json + launched/<job_id>.json
- 启动「5 分钟后首次唤醒」后台调度（见「唤醒调度」）
```

**6-7. 唤醒循环与判据**

唤醒节奏见下节「唤醒调度」（每次 sbatch / 续训重提后重新计数）。
每次唤醒：

1. **优先只读探查**：`bash scripts/remote_status.sh`（队列/登记/GPU），再按需 `tail_remote_logs`、
   列 `checkpoints/<NAME>/`、读 `config.json`——确认拉起状态与进度，**不必先 pull**。
   若本轮要 `scancel` / 重提 sbatch，同样先跑 `remote_status.sh`。
2. 需要本地对照元数据时再 `pull --mode fast [NAME]`（禁 full）。
3. 看训练数据：loss/step、gen_ppl 等；需要权重时 `pull-file` 拉某个 checkpoint。
4. 在本机跑 generate / eval（**不要在远端测**，见 train-ops）。
5. 四选一：
   - 曲线健康、还值得训 → 继续循环（6）。
   - 需要调整 → 8。
   - 已收敛/无需再调 → 11。
   - **卡死在同一 bug**（见「卡住即停」）→ 11，勿再修。

**8-9. 同分支调整**

- 可溯源的小问题（代码 bug、超参），在当前分支直接改。
- 改完先 **commit**（遵循 4.5 的"推前必须提交、工作区干净"约定），再继续。
- 若改动影响显存：必须再跑 **4.6**（探针 → 填 `alloc.md` → 查表选型），**禁止**缺测直接重提 full。
- 处理旧 checkpoint：
  - **保留一份最佳/最近的基准 checkpoint**：勿用 `cache/temp/`；暂留在原
    `cache/checkpoints/<run>/`，并在 `temp/auto-research/<idea>/` 注明对照 run；
  - 其余旧 run 的 checkpoint 删除（仅限 AI 登记任务范围 —— 这是对"远端只读、仅 temp 可写"
    rule 的显式例外；不要动他人 run）。
- 在 `temp/auto-research/<idea>/` 记本次调整原因与基准位置。
- 重新 `push` 后回 4/5。

**10. 架构调整 / 换向**

当前分支方向无望（指标天花板、结构缺陷）或需调整 → 产生**新分支**。分两类：

- **架构变体（同一思路的演进）**：
  - **派生新分支前**：按「开新分支前重读」重新通读该思路记录与当前实现，确认理解后决定改点。
  - 从**当前分支** fork（`git switch -c <idea>-<variant>`），继承已实现代码，在此之上修改，
    绝不回 master 从零恢复。
  - 记录在**原 `temp/auto-research/<idea>/`**（新增 `<variant>.md`，注明父分支、改动、放弃/调整原因）。
  - 处理旧 run checkpoint：留存一份最新权重于原 `cache/checkpoints/<run>/`（勿用 `cache/temp/`），
    其余删除；在笔记中写明保留的 run。
- **全新想法（切换方向）**：
  - 从 `master` fork 新根分支，新建独立 `temp/auto-research/<newidea>/`。
  - 旧想法的放弃原因写回原 `temp/auto-research/<idea>/`；如需保留权重，同样暂留原
    `cache/checkpoints/<run>/` 并在笔记标明（勿用 `cache/temp/`）。
- 两种都回步骤 3（实现/调整），而非步骤 1。

> 判别：改动是**同一思路的演进** → 架构变体；是**无关新方向** → 全新想法。

**11. 完成与总结**

- 判定标准：当前方向已收敛、继续调优收益趋近于零、资源/时间到限、或触发「卡住即停」（久修不果的 bug）。
- 写 `temp/auto-research/<idea>/SUMMARY.md`：最终架构、最佳配置、数据、优缺点、后续建议；若因 bug 放弃，写清症状、已试修复与放弃原因。
- 结束循环，向用户给出**完整总结**（用了哪些分支、跑了哪些实验、结论、checkpoint 位置）。

## 唤醒调度（重要）

Cursor agent 无自主闹钟；用后台 `sleep` + 特征串输出，在 turn 结束后由系统通知唤醒。  
间隔按**本轮作业已成功唤醒次数**递进（每次 sbatch / 续训重提后重新计数）：

| 次序 | 间隔 | sleep |
|------|------|-------|
| 第 1 次 | 5 分钟 | `300` |
| 第 2 次 | 15 分钟 | `900` |
| 第 3 次起 | 每 30 分钟 | `1800` |

```bash
sleep 300 && echo "AUTO-TRAIN-WAKEUP-1"    # 第 1 次
sleep 900 && echo "AUTO-TRAIN-WAKEUP-2"    # 第 2 次
sleep 1800 && echo "AUTO-TRAIN-WAKEUP"     # 第 3 次及以后
```

- 唤醒消息须带**明确下一步**（读日志 / pull fast / 决定继续或调整）。
- **依赖对话保持开启**；会话中断后，下次从 ssh 读 job/日志恢复，勿从头重来。
- ssh 连不上时的重试仍睡 5 分钟（与上表无关）。
- 收敛退出：多次不可救报错、换向、判定完成、或「卡住即停」。

## 卡住即停（防空烧 token）

同一阻塞问题（编译/运行崩溃、loss NaN、数据管线挂死、反复相同 traceback 等）**不要无限修—重跑**。满足任一即结束本轮 auto-train（进 11，写 SUMMARY，向用户说明），**禁止**为「再试一次」继续烧对话 / 训练 token：

- **同因修复 ≥3 次**仍不过本地冒烟，或远端仍以同类错误失败；
- **连续 ≥2 次换向/大改**仍卡在同一根因（说明方向或问题理解不对，继续叠改无意义）；
- 已清楚根因但修复依赖外部条件（缺数据、集群策略、非本仓库权限等），短期无法推进。

未达阈值前：每次失败在 `temp/auto-research/<idea>/` 记清报错摘要与已试手段，避免重复无效路径。到阈值后**立刻停**，不要再开一轮 sbatch「碰运气」。

## 异常 / 失败处理

- **ssh 连不上**：睡 5 分钟重试（可能是网络波动）。
- **连续 3 次连不上**：退出任务，向用户总结已经做到哪一步、远端状态如何、如何恢复。
- **远端 job 报错 / 崩溃**：看日志定位；可救则同分支修（9），不可救则换向（10）；反复同类错误按「卡住即停」。
- **本地训练/推理卡死**：结束本次占用进程重试；不随意杀用户自己启动的进程；同因多次卡死同样适用「卡住即停」。

## 边界（不要自动做）

- 没有明确"请自动执行"→ 只讨论/给方案，不实际开训。
- 不用 `pull --mode full`（体积风险，见 sync skill 硬性禁令）。
- 不 push 非 full 的 slurm 脚本；不用 preprocess 作业。
- 不删非本 agent 登记范围内他人的 job / checkpoint。