"""Global HuggingFace settings.

Importing this module applies the global HuggingFace settings as a side effect:
disable XET and use the hf-mirror endpoint. Import this module BEFORE importing
any ``huggingface_hub`` / ``datasets`` package so the settings take effect.
"""

import os

# hf-mirror endpoint.
HF_ENDPOINT = "https://hf-mirror.com"

# Disable XET.
os.environ["HF_HUB_DISABLE_XET"] = "1"
# Use the hf-mirror endpoint.
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

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
