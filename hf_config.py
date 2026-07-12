"""Global HuggingFace settings.

Importing this module applies the global HuggingFace settings as a side effect:
disable XET and use the hf-mirror endpoint. Import this module BEFORE importing
any ``huggingface_hub`` / ``datasets`` package so the settings take effect.
"""

import os
from pathlib import Path

# hf-mirror endpoint.
HF_ENDPOINT = "https://hf-mirror.com"

# Disable XET.
os.environ["HF_HUB_DISABLE_XET"] = "1"
# Use the hf-mirror endpoint.
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# Keep HuggingFace caches under project cache/ (typically symlinked off C:).
_PROJECT_ROOT = Path(__file__).resolve().parent
_HF_CACHE_ROOT = _PROJECT_ROOT / "cache" / "huggingface"
for _var, _path in {
    "HF_HOME": _HF_CACHE_ROOT,
    "HF_DATASETS_CACHE": _HF_CACHE_ROOT / "datasets",
    "TRANSFORMERS_CACHE": _HF_CACHE_ROOT / "transformers",
    "HUGGINGFACE_HUB_CACHE": _HF_CACHE_ROOT / "hub",
}.items():
    os.environ.setdefault(_var, str(_path))

# Bypass proxy for HuggingFace hosts. The domestic mirror is accessed directly;
# routing through a SOCKS/HTTP proxy often causes SSL errors or missing socksio.
_HF_NO_PROXY_HOSTS = "hf-mirror.com,huggingface.co,cdn-lfs.huggingface.co"
_existing_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
if _existing_no_proxy:
    _merged = f"{_existing_no_proxy},{_HF_NO_PROXY_HOSTS}"
else:
    _merged = _HF_NO_PROXY_HOSTS
os.environ["NO_PROXY"] = _merged
os.environ["no_proxy"] = _merged
