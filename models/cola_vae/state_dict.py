"""Legacy state_dict key remaps for Cola-VAE MLP renames."""

from __future__ import annotations

from typing import Mapping


def remap_vae_mlp_keys(state: Mapping[str, object]) -> dict[str, object]:
    """Map legacy ``mlp.fc`` / ``mlp.proj`` onto ``c_fc`` / ``c_proj``."""
    return {
        key.replace(".mlp.fc.", ".mlp.c_fc.").replace(".mlp.proj.", ".mlp.c_proj."): value
        for key, value in state.items()
    }
