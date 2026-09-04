# Python 虚拟环境

本机任何 Python（`generate.py` / 脚本 / `pip`）必须用仓库根 `.venv`，禁止系统 `python`/`python3`。

```bash
.venv/bin/python generate.py ...
.venv/bin/python scripts/resolve_checkpoint.py ...
.venv/bin/pip install -r requirements.txt
# 或：source .venv/bin/activate && python ...
```

缺环境时：`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`。
