"""监控站常量（不 import torch / models）。"""

from __future__ import annotations

from pathlib import Path

CHECKPOINT_ROOT = Path("cache/checkpoints")
EVAL_ROOT = Path("cache/eval")
MONITOR_STORE = Path("cache/monitor/charts.json")  # 只推不拉（见 skill sync）
MONITOR_INSTANCE = Path("cache/monitor/instance.json")  # 不进 git；push 排除，远端另写 remote
HASH_GUIDE_NAME = "hash_guide.csv"

LIVE_THRESHOLD_SEC = 60.0
PORT_MIN = 16385
PORT_MAX = 65535
PORT_BIND_TRIES = 32

DEFAULT_MAX_POINTS = 4096
MAX_POINTS_CAP = 8192
EVAL_SERIES_CAP = 5000

CHART_REFRESH_SEC = 60
SCAN_CACHE_SEC = 15.0

SOURCE_FILES = {
    "train": "train_log.csv",
    "eval": "eval_log.csv",
    "train_official": "train_metrics/official.csv",
    "eval_official": "eval_components/official.csv",
}

TRAIN_SOURCES = frozenset({"train", "train_official"})
EVAL_SOURCES = frozenset({"eval", "eval_official"})
