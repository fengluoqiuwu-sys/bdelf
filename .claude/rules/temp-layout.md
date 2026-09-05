# temp/ 布局

`temp/` 在 `.gitignore` 中，本地与远端**互不同步**。Claude 以只读已有产物为主；写 `temp/` 仅限用户明确要求且不越界。

| 路径 | 用途 |
|------|------|
| `temp/auto-research/<idea>/` | 自动训练/优化记录（Claude 可只读；不开训） |
| `temp/ideas/<name>/` | 人确认可行后的 idea 工作夹（**当前阶段只写 `stage.md`**；专用代码 `source/`，数据与结果 `result/`） |
| `temp/idea/<idea>/` | 人工认可后的旧规格文稿（新流程用 `temp/ideas/`） |
| `temp/papers/<name>/` | 论文（`paper/` + `INDEX.md`）与可选 `sources/`（Claude 只读 INDEX；下载/编索引交给 Cursor） |
| `temp/research-scout/<run>/` | 自由探索找 idea 的 run 记录；可行 `ideas/I-{n}/`，失败 `ideas/D-{n}/`（Claude 只读，不创建新 run） |
| `temp/vram-probe/` | `alloc.md`：model×batch→alloc_peak_GiB（Cursor 开训查表） |
| `temp/web/` | 本机 `scripts/web.sh` 状态（pid / 隧道 / 端口）；远端另有 `temp/web/monitor.pid` |
| `temp/agent/`（本机） | 本机调试作业登记（`scheduler: "local"`；见「本机计算约束」） |
| `temp/agent/`（common 远端） | 占 GPU 作业登记（`gpu_ids` + `gpus`；不登记 CPU；见「远端 common 计算约束」） |
| `temp/sar-jobs/` | slurm-auto-run 生成的作业脚本（gitignore；勿手改、勿直接 sbatch） |

另：`logs/<服务名>/<时间戳>/`（仓库根、gitignore）存放作业 `.out` / `.err` / `gpu-*.log`（Slurm 与 common 统一）。**pull 会拉取**；push 不上传且不因 `--delete` 清远端。

- `<idea>` / `<name>` / `<run>`：短横线小写 slug。
- `temp/ideas/<name>/README.md` **只**写跨阶段身份；当前阶段、闸、本阶段产物、下一步写 `stage.md`。
- scout / idea-explore **不写入** `temp/ideas/` 或 `temp/idea/`；人说某条可行后再由 Cursor 走 `idea-kickoff`。
- 本题专用代码在 `temp/ideas/<name>/source/`，本题数据与结果在 `result/`。全局训练、共享数据、checkpoint **不要**放进 idea 夹。
- 勿把大 checkpoint / 权重放进 `temp/`（含 `result/`）。
