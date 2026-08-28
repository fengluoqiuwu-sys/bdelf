"""BELF / RELF 共享原语。不是可训家族：无 ``CONFIG_CLS`` / ``build_model``。"""

from __future__ import annotations

from .cfg import (
    blend_v_tgt,
    hide_left_keys,
    keep_params_in_graph,
    maybe_drop_left,
    pad_after_first_eos,
    sample_w_sc,
)
from .exit import CausalExit, ExitMap
from .flow import interpolate, v_star, x_to_v
from .flex_mask import (
    FLEX_ATTN_AVAILABLE,
    build_belf_flex_block_mask,
    build_belf_flex_left_mask,
    build_belf_flex_right_mask,
    build_relf_flex_block_mask,
    build_relf_flex_left_mask,
    build_relf_flex_right_mask,
    relf_windows_visible,
)
from .gen_buf import SeqBuf, alloc_capacity, ensure_seq_buf
from .layers import (
    AdaLNZeroStack,
    LeftKVCache,
    ScaleEmbedder,
    TimestepEmbedder,
    as_sdpa_mask,
)
from .latent import LatentBundle, validate_joint_tune, validate_loaded_block
from .pack import (
    group_causal_mask,
    hide_right_pad_from_unknown,
    pack_2l,
    pack_2l_mask,
    pack_2l_parallel_blocks_mask,
)
from .time import check_time_step, ladder_levels

__all__ = [
    "AdaLNZeroStack",
    "LeftKVCache",
    "CausalExit",
    "ExitMap",
    "LatentBundle",
    "ScaleEmbedder",
    "TimestepEmbedder",
    "FLEX_ATTN_AVAILABLE",
    "SeqBuf",
    "alloc_capacity",
    "ensure_seq_buf",
    "as_sdpa_mask",
    "build_belf_flex_block_mask",
    "build_belf_flex_left_mask",
    "build_belf_flex_right_mask",
    "build_relf_flex_block_mask",
    "build_relf_flex_left_mask",
    "build_relf_flex_right_mask",
    "relf_windows_visible",
    "blend_v_tgt",
    "hide_left_keys",
    "hide_right_pad_from_unknown",
    "keep_params_in_graph",
    "check_time_step",
    "group_causal_mask",
    "interpolate",
    "ladder_levels",
    "maybe_drop_left",
    "pad_after_first_eos",
    "pack_2l",
    "pack_2l_mask",
    "pack_2l_parallel_blocks_mask",
    "sample_w_sc",
    "v_star",
    "validate_joint_tune",
    "validate_loaded_block",
    "x_to_v",
]
