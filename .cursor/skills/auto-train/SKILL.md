---
name: auto-train
description: >-
  Automate the full loop of training and tuning a model end to end: record under
  temp/auto-research/<idea>/, edit only on master (with workspace lock), run fast
  local smoke-tests, push to a user-confirmed servers.csv host, submit full training
  (Slurm or common per that host), then wake on a schedule to pull-fresh results,
  evaluate, decide whether to keep training, adjust forward-compatibly or pivot,
  and finally summarize. Trigger only when the user explicitly says "自动执行" /
  "auto" AND confirms the training server name AND authorizes execution. Not for
  one-off manual runs; those belong to train-ops / sync.
---

# auto-train

自动训练 + 自动优化的完整闭环 skill。仅在用户**明确说"请自动执行"并授权**后启动。

配合 rule「本机计算约束」「远端 Slurm 计算约束」「远端 common 计算约束」「temp/ 布局」、skill `train-ops` / `sync` / `train` / `generate` / `vram-probe`。
本 skill 只编排闭环与决策；具体命令去读对应 skill。

## 远端只读探查（优先于 pull）

看目录、config、登记、日志末尾时：**优先 ssh 只读**（命令见 skill `train-ops`），不要为看一眼就 pull。  
勿 ssh 读大 `.pt`；需要权重用 `pull-file` / `pull --mode common`。

## 触发与授权

1. 用户描述任务（可能还提到某个架构/思路），并说"请自动执行"。
2. **必须**在下一条回复里先确认两件事，都得到明确答复后才动：
   - **训练服务名**：`scripts/servers.csv` 的「名字」列（如 `ovan-server` / `upload-server`）；
     未指定则列出可用名字请用户选。**禁止**默认任一主机。
   - **是否执行**：得到明确确认后才进入下一步。
3. 将确认的服务名记入 `temp/auto-research/<idea>/README.md`（字段建议：`server: <名字>`、
   `scheduler: slurm|common`）；本轮全程 sync / ssh / 提交 / 登记**只使用该名字**。
   中途换机须重新向用户确认并改写 README。
4. 确认执行时，把后续**所有可能运行的一次性指令**（打 tag、起后台调度、首次 push、
   sbatch / common 启训等）合并成**一个待批准块**一次给出，让用户一次性授权，防止其离开后逐条授权卡住。
   - 构建鉴权：提交命令尽量合并为一条 `&&` / 一个脚本；初始化、登记等放同一块。
   - 若某条被拒，停下向用户说明，不要擅自更换等价命令绕过。
5. **一次性授权即覆盖整个自动训练循环，不设边界**：得到"批准"后，整个循环（ssh 只读探查、
   push/提交、删本 agent 登记 run 的 checkpoint、换向记笔记、sleep 唤醒、
   资源等待（排队后 60m 再看 / AI 额度满 60m / 抢锁 30m）、连不上重试等）
   均视为已被授权，自动执行，不再逐条打断用户。
   - **绝对底线**（不因授权而放宽，见「边界」）：不删/不动**非本 holder 登记范围**他人的
     job 与 checkpoint；不执行 `pull --mode full`；不 push 非 full 脚本；
     **未确认服务名不得开训**；
     **非向前兼容 / 可能影响其他模型训练或推理的改动须向用户二次确认**（见下「改动兼容性」）。
     这几条即便已"全权"也不做，如确有必要先停下来向用户确认。

调度类型按该服务在 `servers.csv` 的「调度类型」列走：`slurm` → `remote_status` + `sbatch-train`（见
train-ops / 远端 Slurm 规则）；`common` → `scripts/launch-train.sh --server <服务名> --gpus …`（自动写
`temp/agent` 与 `logs/<服务名>/<时间戳>/`；见远端 common 规则）。
下文示例里的 `<服务名>` 均指本轮已确认的名字。

## 代码与记录约定（一律在 master）

本地 `temp/` 三分法见 rule「temp/ 布局」：`auto-research/`（本 skill 实验记录）、
`idea/`（读论文提出的想法规格）、`papers/`（论文与克隆代码）。远端 `temp/agent/` 仅 job 登记。

- **唯一工作分支：`master`**。自动训练的代码/配置改动**全部在 `master` 上完成**；**禁止**再为 idea
  fork 任务分支或改完后 `git switch` 到别的实现分支。思路隔离靠 `temp/auto-research/<idea>/`，不靠 git 分支。
- **改代码前须抢工作区锁**（见下节与 skill `train-ops`）；**generate 不占锁**，但应确认当前为 `master`
  （若被切走则 `git switch master`）。
- **开改前重读**：动手实现 / 换向前，通读 `temp/idea/<idea>/`（若有）、`temp/auto-research/<idea>/`
  与相关代码，确认理解后再改，避免叠错方向。
- **记录粒度**：
  - 架构变体 / 调参 / 修 bug → 仍用同一 `temp/auto-research/<idea>/`，新增 `.md` 记改动与原因。
  - 全新想法 → **新建** `temp/auto-research/<newidea>/`；放弃原因写回旧 idea 目录。
- 每个自动训练思路在 **本地 `temp/auto-research/<idea>/`** 下建目录记录：
  - `README.md`：本轮自动研究目标/口径（可链到 `temp/idea/<idea>/README.md`）。
  - 每个调整/子实验一个 `.md`：写明改动内容、原因、父节点、结果数据。
  - `SUMMARY.md`（只在完成后写）：整个思路的结论与最终建议。
- **换向/停止时的权重留存**：checkpoint 留在 `cache/checkpoints/{fast|full}/{model}/{hash}/`，
  在笔记里写明 run；**不**放进 `temp/` / `cache/temp/`。
- 远端 `temp/` 与本地 `temp/` **互不同步**（登记规则见 train-ops）。

## 改动兼容性（强制）

因改动落在共享的 `master` 上，**默认只做向前兼容、且不影响其他模型训练/推理**的改动：

- 优先：新增模型/配置文件、新增可选字段并给**保持旧行为**的默认值、仅本 idea 专用脚本/YAML。
- 改共享代码（`train/`、`models/` 公共基类、tokenizer、preprocess、registry 等）时：旧模型与旧
  checkpoint 的训练/续训/generate **行为应不变**；见 rule「Checkpoint 路径与配置哈希」配置演进。
- **下列属不向前兼容（或可能波及其他模型）→ 必须停下来向用户二次确认**，不得因「已自动执行授权」自行落地：
  - 修复「实现错误」但会改变已有模型数值结果 / 旧 checkpoint 可续训性；
  - 删除、改名或改变必填配置语义；改共享 API/张量布局/loss 定义导致其他模型训练或推理异常；
  - 任何你无法确信「只影响本 idea、旧 run 仍可按原语义用」的改动。
- 二次确认时说明：影响面、为何不能只用新增默认值规避、是否需作废/迁移哪些旧 checkpoint。

## 本机工作区锁（非 temp 改动）

凡 AI 改动本机**非 `temp/`** 文件（实现、配置、commit、fast 冒烟相关改动等）前，须抢占
工作区锁（见 skill `train-ops`）：

```bash
WHO="auto-train:<idea>"
.venv/bin/python scripts/workspace_lock.py acquire --holder "$WHO" --purpose "<简述>"
# …在 master 上改 / commit…
.venv/bin/python scripts/workspace_lock.py release --holder "$WHO"
```

**generate 不占锁**。

- 抢到 → 确认在 **`master`** → 操作 → commit（如需）→ 释锁（仍停在 master）。
- 抢不到（`acquire` exit 1）→ 睡 **30 分钟**再试（见「资源等待」）。

## 主流程

```
0. 授权确认 + 确认训练服务名 <服务名>（见上；写入 README）
1. 在 temp/auto-research/<idea>/README.md 记录口径与 server/scheduler（不 fork git 分支）
2. 实现思路（抢锁 → master 上改 → 兼容性自检 → commit → 释锁）
3. 本地验证：抢锁 → master 上 fast 冒烟；generate 不占锁 → 释锁
3.6 显存探针（强制，见「VRAM 探针」）：改动影响显存时 → push → vram-probe → 填 alloc.md
4. push 到 <服务名> → 按表+global_bs 选型 → 按调度类型提交 full
   （slurm：remote_status，AI 合计 GPU+2≤4；额度满睡 60m；勿因 AVAIL=0 空等 → sbatch 排队）
   （common：选卡 → launch-train --server --gpus，遵守 csv 额度；自动登记）
   → 写 active/（slurm 手写；common 由 launch-train 写）→ 起唤醒调度
5. 唤醒循环：5m → 15m → 30m → 此后每 60m（见「唤醒调度」）
   ├ 继续 → 回 5
   ├ 需调整 → 6
   ├ 已完成 → 9
   └ 同 bug 卡住（见「卡住即停」）→ 9，勿空烧 token
6. 停止训练：可向前兼容修 → 7；需换架构/新 idea → 8；修不动 →「卡住即停」→ 9
7. 抢锁 → master 修改（兼容）→ commit → 释锁 →（若影响显存再跑 3.6）
   → 删远端该 run checkpoint → push 重跑 → 回 3/4
8. 架构调整 / 换向 → 抢锁 → master 上按兼容性规则改（或新配新模型文件）
   → 笔记记放弃/调整 → 释锁 → 回 2；**破坏性改动先二次确认**
9. 完成 / 放弃 → 写 temp/auto-research/<idea>/SUMMARY.md → 结束并总结
```

### 步骤详述

**1. 记录口径（不建 git 分支）**

开改前通读 `temp/idea/<idea>/`（若有）与既有 `temp/auto-research/<idea>/`，写好 `README.md` 口径。

**2. 实现思路**

按项目规范实现；复用 registry / config 加载；改动尽量最小且**向前兼容**（见「改动兼容性」）。  
动手前抢工作区锁并确认 `git branch --show-current` 为 **`master`**；改完按 3.5 提交后释锁。

**3. 本地验证（5080 / fast）—— 只做冒烟，不跑满**

- 本机 Python 用 `.venv/bin/python`（见 rule「Python 虚拟环境」、skill train / generate）。
- 改代码/冒烟前抢锁并确认在 **`master`**；generate 段可不占锁，但仍须在 `master`。
- `fast` **仅用于验证改动能跑通**，不是正式训练：起训练后观察到**首批训练步 loss 正常打印、
  无报错 / 崩溃 / segfault** 即可，通常 **2–3 分钟**内确认后就**主动停掉该进程**（kill），
  不要让 fast 跑满整个 token 预算（正式训练只在远端 full 跑）。
- 本机 GPU 互斥，一次一个进程；冒烟起训后写 `temp/agent/active/pid<PID>.json`（`scheduler:local`），停掉后删 active（见 train-ops）。
- 验证通过后**清理本机中间产物**（快照/调试文件/临时 run 的 checkpoint 不提交），只保留有意义的改动。

**3.5 强制提交（推送到远端前必做）**

推送 `scripts/sync.sh <服务名> push` 之前，**必须先提交到 git，不允许有未保留的内容**
（持有工作区锁、在 **`master`** 上）：

```bash
git branch --show-current   # 须为 master
git status
git add <相关文件>      # 或 git add -A（用前用 git status 确认无夹带）
git commit -m "<语义化描述>"
.venv/bin/python scripts/workspace_lock.py release --holder "$WHO"
# 保持在 master
```

- `git push` 推送的是 git **commit**；`scripts/sync.sh <服务名> push` 推送的是**工作区文件**。若工作区有
  未提交的改动，远端拿到的是无法从 git 恢复的环境——这是不允许的。
- 本地验证通过后、确认改动可用即提交；提交信息写清改动内容与目的。
- 提交后 `git status` 必须**干净**，再进入第 4 步。
- 思路/实验记录（`temp/auto-research/<idea>/*.md`）属**仅本地**记录、不同步；`temp/` 在 `.gitignore` 中，
  **禁止** `git add -f` 或其它方式纳入版本库。

**3.6 VRAM 探针（显存相关改动后强制）**

凡改动会影响显存占用（模型结构、精度、序列长 / chunk、EMA、优化器状态规模等），在提交 full **之前**必须：

1. 工作区已提交且干净（3.5）；`bash scripts/sync.sh <服务名> push`
2. 按 skill **`vram-probe`** + **`train-ops`**（slurm 主机：`remote_status` → 若合计 GPU+1>4 则睡 60m 再看；
   **AVAIL 不足仍先 sbatch 排队**；写 `active/` → `bash slurm/sbatch-vram-probe.sh …`。
   common 主机：按该机额度选空闲 `--gpus` 后跑探针，勿超 csv 上限）
3. 读日志；把各档 **`alloc_peak_GiB`**（及 `oom`）填入 **`temp/vram-probe/alloc.md`** 对应行
4. **不要**把探针结论写回 recipe 默认 YAML；可在 `temp/auto-research/<idea>/` 记一笔链接到该表
5. **禁止**未填/过期表直接开训

仅改 loss 日志文案等明显不影响显存的改动可跳过本步；有疑虑时宁可跑探针。

**开训选型**（步骤 4）：查 `alloc.md` 该模型各列，结合**当前目标卡** `total−2` 与本次 `global_batch_size` / `world_size`，取最大安全且整除的 micro-batch（公式见 skill `vram-probe`）。默认 YAML 已是该值则不必 `--set`；否则专用 `scripts/train/<name>.sh` 加 `--set batch.batch_size=…`。

**4. 推送与远端训练**

```text
- [前置] 已确认 <服务名> 与 scheduler，并写入 README
- [前置] 工作区已干净（第 3.5 步已提交；已在 master、已释锁）
- [前置] 若本轮改动影响显存：第 3.6 步已完成，alloc.md 对应行已更新
- [前置] 已按表 + 当前卡 + global_bs 定好 batch_size（必要时 scripts/train 含 --set）
- bash scripts/sync.sh <服务名> push
- 确认 `scripts/train/<name>.sh` 为 full 配置（禁止 preprocess）
- slurm：bash slurm/remote_status.sh → 若 agent_gpu_sum+4>4 则睡 60min 再看
         → AVAIL 不足仍 sbatch 排队 → ssh 后 bash slurm/sbatch-train.sh <name>
         → 写 active/<job_id>.json（gpus:4, holder:auto-train:<idea>, scheduler:slurm）
- common：扫该机 active → 选不冲突 --gpus（张数≤csv 单任务上限）
         → bash scripts/ssh.sh <服务名> -- bash scripts/launch-train.sh <name> \
              --server <服务名> --gpus … --holder auto-train:<idea>
         （自动写 agent + logs/<服务名>/<时间戳>/；见远端 common 规则）
- 若已 RUNNING → 启动「5 分钟后首次唤醒」（见「唤醒调度」）
- slurm 仍 PENDING → 按「资源等待」睡 60min 再看，拉起后改用「唤醒调度」
```

slurm：登记合计 GPU ≤ 该机 csv「最大使用显卡数量」（ovan 默认本作业 4 卡计入，通常一次一作业）。集群无空闲卡时靠排队，不靠轮询 AVAIL。

**5. 唤醒循环与判据**

唤醒节奏见下节「唤醒调度」（每次 sbatch / 续训重提后重新计数）。
每次唤醒：

1. **优先只读探查**：`bash slurm/remote_status.sh`（队列/登记/GPU），再按需 `tail_remote_logs`、
   列 `checkpoints/<NAME>/`、读 `config.json`——确认拉起状态与进度，**不必先 pull**。
   若本轮要 `scancel` / 重提 sbatch，同样先跑 `remote_status.sh`。
2. 需要本地对照元数据时再 `pull --mode fast [NAME]`（禁 full）。
3. 看训练数据：loss/step、gen_ppl 等；需要权重时 `pull-file` 拉某个 checkpoint。
4. 在本机跑 generate / eval：确认在 **`master`** → generate（**不要在远端测**；generate 不占工作区锁）。
5. 四选一：
   - 曲线健康、还值得训 → 继续循环（5）。
   - 需要调整 → 6。
   - 已收敛/无需再调 → 9。
   - **卡死在同一 bug**（见「卡住即停」）→ 9，勿再修。

**6-7. 同思路调整（仍在 master）**

- 可溯源的小问题（代码 bug、超参）：抢锁 → 确认 `master` → 按「改动兼容性」修改 → **commit**（3.5）→ 释锁。
  若属不向前兼容 / 可能影响其他模型 → **先二次确认用户**。
- 若改动影响显存：必须再跑 **3.6**（探针 → 填 `alloc.md` → 查表选型），**禁止**缺测直接重提 full。
- 处理旧 checkpoint：
  - **保留一份最佳/最近的基准 checkpoint** 于原 `cache/checkpoints/<run>/`，并在
    `temp/auto-research/<idea>/` 注明对照 run；
  - 其余旧 run 的 checkpoint 删除（仅限本 `holder` 登记范围）。
- 在 `temp/auto-research/<idea>/` 记本次调整原因与基准位置。
- 重新 `push` 后回 3/4。

**8. 架构调整 / 换向（仍在 master；靠 temp 目录区分思路）**

当前方向无望或需调整时：

- **架构变体（同一思路）**：通读笔记与实现 → 抢锁 → 在 `master` 上做**兼容**改动（优先新配置/新可选字段）→
  原 `temp/auto-research/<idea>/` 增 `<variant>.md` → commit → 释锁。旧 run 留一份权重、其余按 holder 清理。
- **全新想法**：新建 `temp/auto-research/<newidea>/`；旧想法放弃原因写回旧目录；代码仍在 `master`
  上按兼容性规则增加独立模型/配置，避免拖垮旧模型。
- **破坏性改动**（修实现错误会改旧数值、改共享 API 等）→ **必须二次确认**后再做。
- 然后回步骤 2，而非重建 git 分支。

**9. 完成与总结**

- 判定标准：当前方向已收敛、继续调优收益趋近于零、资源/时间到限、或触发「卡住即停」。
- 写 `temp/auto-research/<idea>/SUMMARY.md`：最终架构、最佳配置、数据、优缺点、后续建议；若因 bug 放弃，写清症状、已试修复与放弃原因。
- 结束循环，向用户给出**完整总结**（实验记录目录、跑了哪些实验、结论、checkpoint 位置）。

## 唤醒调度（重要）

Cursor agent 无自主闹钟；用 ``scripts/agent_wakeup.py`` 后台 sleep，结束后向 stdout 打
``AGENT_WAKEUP {...}``（含 ``prompt``），由终端输出通知唤醒。

间隔按**本轮作业已成功唤醒次数**递进（每次 sbatch / 续训重提后重新计数）：

| 次序 | 间隔 | 调用 |
|------|------|------|
| 第 1 次 | 5 分钟 | `--nth 1` |
| 第 2 次 | 15 分钟 | `--nth 2` |
| 第 3 次 | 30 分钟 | `--nth 3` |
| 第 4 次起 | 每 60 分钟 | `--nth 4`（及更大） |

```bash
# 仓库根；后台跑（Shell: block_until_ms=0），并 notify_on_output 匹配 AGENT_WAKEUP
.venv/bin/python scripts/agent_wakeup.py --nth 1 -- \
  '跑 bash slurm/remote_status.sh；按需 tail 日志；决定继续/调整/完成'

.venv/bin/python scripts/agent_wakeup.py --after 60m --tag resource-wait -- \
  '额度或排队等待结束：再 remote_status，满则继续等，已 RUNNING 则改用 --nth 唤醒'

.venv/bin/python scripts/agent_wakeup.py --after 30m --tag lock-wait -- \
  '再抢 scripts/workspace_lock.py acquire'
```

- ``--`` 之后（或位置参数）为**唤醒后给 agent 的明确下一步**（必填）。
- ``--after 5m|30s|1h|300`` 任意时长；``--nth N`` 用上表。
- **依赖对话保持开启**；会话中断后，下次从 ssh 读 job/日志恢复，勿从头重来。
- ssh 连不上时的重试：``--after 5m --tag ssh-retry -- '…'``。
- 收敛退出：多次不可救报错、换向、判定完成、或「卡住即停」。

## 资源等待（正常流程，不是失败）

下列情况**不算异常**，属于排队/互斥，继续循环即可，不必向用户确认、也不因此进「卡住即停」或结束：

| 条件 | 动作 |
|------|------|
| `agent_gpu_sum` + 本作业卡数将 > 4 | **先不提交**；`agent_wakeup.py --after 60m --tag resource-wait -- '…'` → 再 `remote_status` → 仍满则重复 |
| 已 sbatch 但仍 PENDING（集群无空闲卡、在排队） | `agent_wakeup.py --after 60m --tag queue-wait -- '…'` → 再看队列；已 RUNNING 则改用 `--nth` 唤醒 |
| 本机锁：`scripts/workspace_lock.py acquire` 失败 | `agent_wakeup.py --after 30m --tag lock-wait -- '再抢锁…'` → 再抢 → 仍失败则重复 |

**禁止**：因 `AVAIL=0` 而空等、不提交——永远等不到「先有空卡再 sbatch」；正确做法是先排队，再 60m 看是否已 RUNNING。

与「唤醒调度」独立：额度满 / 排队 PENDING / 抢锁 属于提交前后的等待；作业已 RUNNING 后的 progress 探查用「唤醒调度」。

## 卡住即停（防空烧 token）

同一阻塞问题（编译/运行崩溃、loss NaN、数据管线挂死、反复相同 traceback 等）**不要无限修—重跑**。满足任一即结束本轮 auto-train（进 9，写 SUMMARY，向用户说明），**禁止**为「再试一次」继续烧对话 / 训练 token：

- **同因修复 ≥3 次**仍不过本地冒烟，或远端仍以同类错误失败；
- **连续 ≥2 次换向/大改**仍卡在同一根因（说明方向或问题理解不对，继续叠改无意义）；
- 已清楚根因但修复依赖外部条件（缺数据、集群策略、非本仓库权限等），短期无法推进。

未达阈值前：每次失败在 `temp/auto-research/<idea>/` 记清报错摘要与已试手段，避免重复无效路径。到阈值后**立刻停**，不要再开一轮 sbatch「碰运气」。

## 异常 / 失败处理

- **ssh 连不上**：睡 5 分钟重试（可能是网络波动）。
- **连续 3 次连不上**：退出任务，向用户总结已经做到哪一步、远端状态如何、如何恢复。
- **远端 job 报错 / 崩溃**：看日志定位；可救则在 master 按兼容性规则修（7），不可救则换向记笔记（8）；反复同类错误按「卡住即停」。
- **本地训练/推理卡死**：结束本次占用进程重试；不随意杀用户自己启动的进程；同因多次卡死同样适用「卡住即停」。

## 边界（不要自动做）

- 没有明确"请自动执行"→ 只讨论/给方案，不实际开训。
- **未确认训练服务名**→ 不得 push / 提交 / 登记开训（不得默认 `ovan-server`）。
- 不用 `pull --mode full`（体积风险，见 sync skill 硬性禁令）。
- 不 push 非 full 的 slurm 脚本；不用 preprocess 作业。
- 不删 / 不 `scancel` 非本 `holder` 登记范围内他人的 job / checkpoint。
- slurm：ovan 默认每作业 4 GPU（与 csv 单任务上限一致）；提交前确认 `agent_gpu_sum+4≤4`。
- **不为 idea fork git 分支**；代码改动只在 `master`，思路隔离用 `temp/auto-research/`。
- **不自动落地非向前兼容 / 可能影响其他模型训练或推理的改动**；须向用户二次确认。
