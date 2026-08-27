"""生成缓冲：容量规则、KV filled、last_n 与全长 slice 对齐。

用法（仓库根）：
  .venv/bin/python scripts/check_gen_buf.py
"""

from __future__ import annotations

import repo_env

repo_env.ensure_repo_root()

import torch

from models.lm.belf_relf_core.gen_buf import SeqBuf, alloc_capacity, ensure_seq_buf
from models.lm.belf_relf_core.layers import AdaLNZeroStack
from models.lm.belf_relf_core.pack import group_causal_mask


def _check_alloc() -> None:
    assert alloc_capacity(1, 64) == 64
    assert alloc_capacity(80, 64) == 80
    assert alloc_capacity(1, None) == 1024
    assert alloc_capacity(1025, None) == 2048
    assert alloc_capacity(0, 0) == 0


def _check_ensure() -> None:
    x = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    # seq_dim=1，当前长 2，扩到 5
    y = ensure_seq_buf(x, 5, known_max=8, seq_dim=1, filled=2)
    assert int(y.size(1)) == 8
    assert torch.equal(y[:, :2], x)
    z = ensure_seq_buf(y, 3, known_max=8, seq_dim=1, filled=2)
    assert z.data_ptr() == y.data_ptr()


def _check_seqbuf() -> None:
    like = torch.zeros(2, 1, 4)
    buf = SeqBuf(known_max=16, seq_dim=1, like=like)
    assert int(buf.buf.size(1)) == 16
    a = torch.randn(2, 3, 4)
    v = buf.replace(a)
    assert v.shape == (2, 3, 4)
    b = torch.randn(2, 2, 4)
    v = buf.append(b)
    assert v.shape == (2, 5, 4)
    assert torch.allclose(v[:, :3], a)
    assert torch.allclose(v[:, 3:], b)


def _check_kv() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    D, H, L0, L1, seqlen = 32, 4, 8, 8, 32
    stack = AdaLNZeroStack(
        D, n_layer=2, n_head=H, dropout=0.0, attn_backend="sdpa",
    ).to(device).eval()
    x0 = torch.randn(2, L0, D)
    mask0 = group_causal_mask(L0, 1, device=device)
    cache = stack.prefill_left(x0, attn_mask=mask0, known_max=seqlen)
    assert cache.filled == L0
    assert cache.left_len == L0
    assert cache.capacity == seqlen
    assert cache.layers[0][0].size(2) == seqlen
    assert cache.x_hat_filled is not None and cache.x_hat_filled.size(1) == L0
    x1 = torch.randn(2, L1, D)
    pos1 = torch.arange(L0, L0 + L1)
    cache = stack.extend_left(x1, cache, positions=pos1, left_group=1, known_max=seqlen)
    assert cache.filled == L0 + L1
    assert cache.capacity == seqlen
    xr = torch.randn(2, 4, D)
    t = torch.ones(2, 4)
    m = torch.ones(2, 4, dtype=torch.long)
    out = stack.forward_right(xr, t, None, m, cache)
    assert out.shape == (2, 4, D)
    # 未写槽须仍为 0，避免 SDPA 误读
    unused = cache.layers[0][0][:, :, cache.filled :].detach().abs().max()
    assert float(unused) == 0.0


def _check_last_n_slice() -> None:
    """``last_n`` 与全长 lm_head 再 slice 对齐。"""
    from models.latent.latent_vae.model import _LatentVAEBackbone
    from tokenizer.tokenizer import FL_TokenLayout

    torch.manual_seed(1)
    layout = FL_TokenLayout(
        vocab_size=32, bos_token_id=0, eos_token_id=1, pad_token_id=2,
    )
    vae = _LatentVAEBackbone(
        layout,
        max_seq_len=64,
        n_layer_enc=1,
        n_layer_dec=1,
        n_head=2,
        n_embd=16,
        d_kv=8,
        d_ff=32,
        latent_dim=8,
        use_flash=False,
        attn_backend="sdpa",
        block_size=1,
    ).eval()
    z = torch.randn(2, 16, 8)
    with torch.no_grad():
        full = vae.decode_logits(z)
        part = vae.decode_logits(z, last_n=4)
    assert full.shape[-1] == 32
    assert part.shape == (2, 4, 32)
    assert torch.allclose(full[:, -4:], part, atol=1e-5, rtol=1e-4)


def main() -> None:
    _check_alloc()
    _check_ensure()
    _check_seqbuf()
    _check_kv()
    _check_last_n_slice()
    print("check_gen_buf: ok")


if __name__ == "__main__":
    main()
