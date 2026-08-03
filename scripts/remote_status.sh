#!/usr/bin/env bash
# 本机工具：一次 ssh 汇总远端 GPU / 队列 / agent 登记。
# 远端作业操作（sbatch / scancel 等）之前必须先跑本脚本（见 rule「远端计算约束」）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE_HOST="${REMOTE_HOST:-ovan-server}"
REMOTE_ROOT="${REMOTE_ROOT:-~/source/bdelf}"
JSON=0

usage() {
  cat <<'EOF'
用法: bash scripts/remote_status.sh [--json]

本机调用，经 ssh 在 ovan-server 上只读查询：
  - slurm/gpu_availability.py（空闲 GPU）
  - squeue -u $USER
  - temp/agent/current.json

环境变量：REMOTE_HOST（默认 ovan-server）、REMOTE_ROOT（默认 ~/source/bdelf）
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# 远端脚本内不 SSH；本工具负责本机 → 登录节点。
# shellcheck disable=SC2029
ssh "$REMOTE_HOST" "cd $REMOTE_ROOT && JSON=$JSON bash -s" <<'REMOTE'
set -euo pipefail

if [[ ! -x .venv/bin/python ]]; then
  echo "远端缺少 .venv/bin/python（$(pwd)）" >&2
  exit 1
fi

if [[ "${JSON:-0}" == "1" ]]; then
  .venv/bin/python - <<'PY'
import json
import os
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

agent_path = root / "temp/agent/current.json"
if agent_path.is_file():
    agent = json.loads(agent_path.read_text(encoding="utf-8"))
else:
    agent = None

print(
    json.dumps(
        {"gpu": gpu, "squeue": rows, "agent_current": agent},
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
  echo "=== agent current ==="
  if [[ -f temp/agent/current.json ]]; then
    cat temp/agent/current.json
    echo
  else
    echo "none"
  fi
fi
REMOTE
