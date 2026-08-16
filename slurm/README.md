# slurm/

实验室通用状态/日志工具（ovan-server / cls1）。**不含**训练/评测作业模板。

复制到新仓库后：

1. `remote_status.sh` 的远端路径来自 `scripts/servers.csv` 的「工作目录」，不要写死旧仓库。
2. **`sbatch-*.sh`、`prototype.slurm`、`eval.slurm`、预处理/VRAM 探针模板未收录**：它们绑定具体入口脚本与 BeeGFS/HF 路径，**必须对照新项目重写**（`partition=cls1`、单节点、16 CPU / 128G 可沿用；**GPU 张数须与该机 csv「单个ai任务最大使用显卡数量」一致**，合计额度读「最大使用显卡数量」，不要写死）。
3. `gpu_availability.py` 默认节点 `cls1-srv[1-4]` 是实验室集群；换分区则改 `DEFAULT_NODES`。

| 文件 | 用途 |
|------|------|
| `remote_status.sh` | 本机 ssh 汇总 GPU / squeue / `temp/agent`；合计额度现场读 csv（作业前必跑） |
| `gpu_availability.py` | 在目标机跑；不发起 SSH |
| `tail_remote_logs.py` | 读 `logs/<服务>/<时间戳>/`；不发起 SSH |
