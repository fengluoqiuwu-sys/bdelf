"""对照 Flex mask_mod 与现有布尔可见性（不比 bf16 输出）。

用法（仓库根）：
  .venv/bin/python scripts/check_belf_relf_flex_mask.py
"""

from __future__ import annotations

from functools import partial

import repo_env

repo_env.ensure_repo_root()

import torch

from models.lm.belf_relf_core.flex_mask import (
    belf_left_prefill_mask_mod,
    make_belf_right_mask_mod,
    make_belf_train_mask_mod,
    make_relf_right_mask_mod,
    make_relf_windows_mask_mod,
    materialize_mask_mod,
    relf_windows_visible,
)
from models.lm.belf_relf_core.pack import (
    hide_right_pad_from_unknown,
    pack_2l_parallel_blocks_mask,
)


def _belf_sdpa_vis(
    n: int,
    w: int,
    device: torch.device,
    *,
    is_pad: torch.Tensor | None = None,
    unknown: torch.Tensor | None = None,
    drop_left: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = pack_2l_parallel_blocks_mask(n, w, device)
    if is_pad is not None and unknown is not None:
        raw = hide_right_pad_from_unknown(raw, is_pad, unknown, w)
    elif drop_left is not None:
        raw = raw.unsqueeze(0).expand(drop_left.size(0), -1, -1).clone()
    if drop_left is not None and n > 0:
        raw = raw.clone()
        raw[drop_left, n:, :n] = False
    return raw


def _check_belf(device: torch.device) -> None:
    n, w, bsz = 16, 4, 3
    two = 2 * n
    is_pad = torch.zeros(bsz, n, dtype=torch.bool, device=device)
    is_pad[0, 14:16] = True
    unknown = torch.ones(bsz, n, dtype=torch.bool, device=device)
    unknown[0, :2] = False
    drop = torch.zeros(bsz, dtype=torch.bool, device=device)
    drop[1] = True

    ref = _belf_sdpa_vis(
        n, w, device, is_pad=is_pad, unknown=unknown, drop_left=drop,
    )
    mod = make_belf_train_mask_mod(
        n, w, is_pad=is_pad, unknown=unknown, drop_left=drop,
    )
    got = materialize_mask_mod(mod, q_len=two, kv_len=two, device=device, batch=bsz)
    if not torch.equal(ref, got):
        raise AssertionError(
            f"BELF 可见性不一致：diff={(ref != got).sum().item()} / {ref.numel()}"
        )

    raw = pack_2l_parallel_blocks_mask(n, w, device)
    static = materialize_mask_mod(
        make_belf_train_mask_mod(n, w),
        q_len=two, kv_len=two, device=device,
    )
    if not torch.equal(raw, static):
        raise AssertionError("BELF 静态 2L 与 pack_2l_parallel_blocks_mask 不一致")

    right_mod = make_belf_right_mask_mod(
        n, w, is_pad=is_pad, unknown=unknown, drop_left=drop,
    )
    right = materialize_mask_mod(
        right_mod, q_len=n, kv_len=two, device=device, batch=bsz,
    )
    if not torch.equal(got[:, n:, :], right):
        raise AssertionError(
            f"BELF 右段 vs 2L 右块不一致：diff={(got[:, n:, :] != right).sum().item()}"
        )
    right_static = materialize_mask_mod(
        make_belf_right_mask_mod(n, w), q_len=n, kv_len=two, device=device,
    )
    if not torch.equal(static[n:, :], right_static):
        raise AssertionError("BELF 静态右段与 2L 右块不一致")
    left = materialize_mask_mod(
        partial(belf_left_prefill_mask_mod, block_size=w),
        q_len=n, kv_len=n, device=device,
    )
    if not torch.equal(static[:n, :n], left):
        raise AssertionError("BELF 左 prefill 与 2L 左-左块不一致")
    print("BELF mask_mod 对照通过")


def _check_relf(device: torch.device) -> None:
    left, w, step, n_win, bsz = 8, 4, 2, 3, 2
    two = left + n_win * w
    u = torch.tensor([[-2, 2, 6], [0, 4, 8]], device=device)
    k0 = torch.tensor([[0, 1, 0], [2, 0, 0]], device=device)
    active = torch.tensor(
        [[True, True, False], [True, True, True]], device=device,
    )
    in_win = torch.ones(bsz, n_win, w, dtype=torch.bool, device=device)
    in_win[0, 1, 0] = False
    drop = torch.tensor([True, False], device=device)

    ref = relf_windows_visible(
        left, w, step, u, active, k0=k0, in_win=in_win, drop_left=drop,
    )
    mod, two_m = make_relf_windows_mask_mod(
        left, w, step, n_win, u, active, k0=k0, in_win=in_win, drop_left=drop,
    )
    if two_m != two:
        raise AssertionError(f"RELF two 不一致：{two_m} vs {two}")
    got = materialize_mask_mod(mod, q_len=two, kv_len=two, device=device, batch=bsz)
    if not torch.equal(ref, got):
        raise AssertionError(
            f"RELF 可见性不一致：diff={(ref != got).sum().item()} / {ref.numel()}"
        )
    right_mod, right_len, two_r = make_relf_right_mask_mod(
        left, w, step, n_win, u, active, k0=k0, in_win=in_win, drop_left=drop,
    )
    if two_r != two:
        raise AssertionError(f"RELF 右段 two 不一致：{two_r} vs {two}")
    right = materialize_mask_mod(
        right_mod, q_len=right_len, kv_len=two, device=device, batch=bsz,
    )
    if not torch.equal(got[:, left:, :], right):
        raise AssertionError(
            f"RELF 右段 vs 2L 右块不一致：diff={(got[:, left:, :] != right).sum().item()}"
        )
    left_got = materialize_mask_mod(
        partial(belf_left_prefill_mask_mod, block_size=1),
        q_len=left, kv_len=left, device=device,
    )
    for bi in range(bsz):
        if not torch.equal(got[bi, :left, :left], left_got):
            raise AssertionError(f"RELF 左 prefill 与 2L 左-左块不一致 (b={bi})")
    print("RELF mask_mod 对照通过")


def main() -> int:
    device = torch.device("cpu")
    _check_belf(device)
    _check_relf(device)
    print("全部对照通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
