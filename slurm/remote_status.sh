#!/usr/bin/env bash
# 本机工具（ovan-server / Slurm 专属）：一次 ssh 汇总远端 GPU / 队列 / agent 登记。
# 远端作业操作（sbatch / scancel 等）之前必须先跑本脚本（见 rule「远端 Slurm 计算约束」）。
# 合计 GPU 额度现场读本机 scripts/servers.csv「最大使用显卡数量」，不写死。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SCRIPT_DIR="$ROOT/scripts"
# shellcheck source=../scripts/servers_lib.sh
source "${SCRIPT_DIR}/servers_lib.sh"

REMOTE_HOST="${REMOTE_HOST:-ovan-server}"
JSON=0

usage() {
  cat <<'EOF'
用法: bash slurm/remote_status.sh [--json]

本机调用，经 ssh 在 ovan-server 上只读查询：
  - slurm/gpu_availability.py（空闲 GPU）
  - squeue -u $USER
  - temp/agent/active/*.json（AI 作业登记；兼容旧 current.json）

环境变量：REMOTE_HOST（默认 ovan-server）。
远端工作目录与 GPU 额度：scripts/servers.csv 该行的「工作目录」/
「最大使用显卡数量」（合计）/「单个ai任务最大使用显卡数量」。
可用 REMOTE_ROOT 覆盖工作目录。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
done

load_server "$REMOTE_HOST" || exit 1
REMOTE_ROOT="${REMOTE_ROOT:-$REMOTE_DIR}"

# 远端脚本内不 SSH；本工具负责本机 → 登录节点。
# 额度从本机 csv 传入（csv 不同步到远端）。
# shellcheck disable=SC2029
ssh "$REMOTE_HOST" "cd $REMOTE_ROOT && JSON=$JSON AGENT_GPU_BUDGET=$SERVER_GPU_MAX GPU_PER_JOB=$SERVER_GPU_PER_JOB bash -s" <<'REMOTE'
set -euo pipefail

if [[ ! -x .venv/bin/python ]]; then
  echo "远端缺少 .venv/bin/python（$(pwd)）" >&2
  exit 1
fi

if [[ "${JSON:-0}" == "1" ]]; then
  .venv/bin/python - <<'PY'
import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
gpu = json.loads(
    subprocess.check_output(
        [str(root / ".venv/bin/python"), "slurm/gpu_availability.py", "--json"],
        text=True,
    )
)
sq = subprocess.run(
    ["squeue", "-u", os.environ["USER"], "-h", "-o", "%i|%j|%t|%D|%C|%b|%R"],
    text=True,
    capture_output=True,
)
rows = []
if sq.returncode == 0:
    for line in sq.stdout.splitlines():
        parts = line.split("|", 6)
        if len(parts) < 7:
            continue
        jid, name, st, nodes, cpus, gres, reason = parts
        rows.append(
            {
                "job_id": jid,
                "name": name,
                "state": st,
                "nodes": nodes,
                "cpus": cpus,
                "gres": gres,
                "reason": reason,
            }
        )
elif sq.stderr.strip():
    print(sq.stderr.strip(), file=sys.stderr)

def _gpus_from_record(rec: dict) -> int:
    if isinstance(rec.get("gpus"), int) and rec["gpus"] > 0:
        return rec["gpus"]
    gres = str(rec.get("gres") or "")
    m = re.search(r"gpu(?::[^=]+)?(?:=|:)(\d+)", gres, re.I)
    if m:
        return int(m.group(1))
    return int(os.environ["GPU_PER_JOB"])  # 缺 gpus 字段时用 csv 单任务上限


def _load_agent_active(agent_root: Path) -> list[dict]:
    active_dir = agent_root / "active"
    jobs: list[dict] = []
    if active_dir.is_dir():
        for path in sorted(active_dir.glob("*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(rec, dict):
                continue
            rec.setdefault("job_id", path.stem)
            rec["gpus"] = _gpus_from_record(rec)
            jobs.append(rec)
    # 兼容旧单文件 current.json
    legacy = agent_root / "current.json"
    if legacy.is_file() and not jobs:
        try:
            rec = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rec = None
        if isinstance(rec, dict) and rec.get("job_id"):
            rec["gpus"] = _gpus_from_record(rec)
            jobs.append(rec)
    return jobs


agent_jobs = _load_agent_active(root / "temp/agent")
agent_gpu_sum = sum(int(j.get("gpus") or 0) for j in agent_jobs)

print(
    json.dumps(
        {
            "gpu": gpu,
            "squeue": rows,
            "agent_active": agent_jobs,
            "agent_gpu_sum": agent_gpu_sum,
            "agent_gpu_budget": int(os.environ["AGENT_GPU_BUDGET"]),
            "gpu_per_job": int(os.environ["GPU_PER_JOB"]),
        },
        ensure_ascii=False,
        indent=2,
    )
)
PY
else
  echo "=== GPU availability ==="
  .venv/bin/python slurm/gpu_availability.py
  echo
  echo "=== squeue ($USER) ==="
  squeue -u "$USER" || true
  echo
  echo "=== agent active (AI jobs) ==="
  if [[ -d temp/agent/active ]] && compgen -G "temp/agent/active/*.json" > /dev/null; then
    sum=0
    for f in temp/agent/active/*.json; do
      echo "--- $(basename "$f") ---"
      cat "$f"
      echo
      g=$(.venv/bin/python -c "import json,os,sys; d=json.load(open(sys.argv[1])); print(int(d['gpus']) if d.get('gpus') else int(os.environ['GPU_PER_JOB']))" "$f")
      sum=$((sum + g))
    done
    echo "agent_gpu_sum=${sum} / budget=${AGENT_GPU_BUDGET} (csv 最大使用显卡数量; per_job=${GPU_PER_JOB})"
  elif [[ -f temp/agent/current.json ]]; then
    echo "(legacy current.json)"
    cat temp/agent/current.json
    echo
    echo "agent_gpu_sum=? / budget=${AGENT_GPU_BUDGET} (csv 最大使用显卡数量; per_job=${GPU_PER_JOB})"
  else
    echo "none"
    echo "agent_gpu_sum=0 / budget=${AGENT_GPU_BUDGET} (csv 最大使用显卡数量; per_job=${GPU_PER_JOB})"
  fi
fi
REMOTE
