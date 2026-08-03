"""Legacy state_dict key remaps for Cola / Cola-VAE MLP renames."""

from __future__ import annotations

import re
from typing import Mapping

from models.cola_vae.state_dict import remap_vae_mlp_keys

# Only DiT blocks used Sequential indices; TimestepEmbedder still does.
_DIT_MLP0_RE = re.compile(r"\.blocks\.(\d+)\.mlp\.0\.")
_DIT_MLP2_RE = re.compile(r"\.blocks\.(\d+)\.mlp\.2\.")


def remap_cola_mlp_keys(state: Mapping[str, object]) -> dict[str, object]:
    """Map pre-rename MLP keys onto current ``c_fc`` / ``c_proj`` names.

    Legacy:
    - VAE ``mlp.fc`` / ``mlp.proj``
    - DiT block ``nn.Sequential`` indices ``blocks.*.mlp.0`` / ``mlp.2``
    """
    state = remap_vae_mlp_keys(state)
    out: dict[str, object] = {}
    for key, value in state.items():
        new_key = _DIT_MLP0_RE.sub(r".blocks.\1.mlp.c_fc.", key)
        new_key = _DIT_MLP2_RE.sub(r".blocks.\1.mlp.c_proj.", new_key)
        out[new_key] = value
    return out
