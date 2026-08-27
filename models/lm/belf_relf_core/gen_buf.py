"""生成序列维缓冲：长度已知则一次预扩，否则从 1024 起 ×2。"""

from __future__ import annotations

import torch

_UNKNOWN_START = 1024


def alloc_capacity(need: int, known_max: int | None) -> int:
    """``known_max`` 有则 ``max(need, known_max)``；否则至少 1024，不够再翻倍。"""
    need = int(need)
    if need < 0:
        raise ValueError(f"need 须非负，收到 {need}")
    if known_max is not None:
        km = int(known_max)
        if km < 0:
            raise ValueError(f"known_max 须非负，收到 {km}")
        return max(need, km)
    cap = _UNKNOWN_START
    while cap < need:
        cap *= 2
    return cap


def ensure_seq_buf(
    buf: torch.Tensor | None,
    need: int,
    *,
    known_max: int | None,
    seq_dim: int,
    filled: int | None = None,
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    """保证 ``seq_dim`` 至少 ``need``；不足则扩容并拷有效前缀。"""
    need = int(need)
    if need < 0:
        raise ValueError(f"need 须非负，收到 {need}")
    ref = buf if buf is not None else like
    if ref is None:
        raise ValueError("ensure_seq_buf 须提供 buf 或 like")
    if seq_dim < 0 or seq_dim >= ref.ndim:
        raise ValueError(f"seq_dim={seq_dim} 超出 ndim={ref.ndim}")
    if buf is not None and int(buf.size(seq_dim)) >= need:
        return buf
    cur = int(buf.size(seq_dim)) if buf is not None else 0
    if known_max is not None:
        cap = max(need, int(known_max))
    elif buf is None:
        cap = alloc_capacity(need, None)
    else:
        cap = max(cur, _UNKNOWN_START)
        while cap < need:
            cap *= 2
    shape = list(ref.shape)
    shape[seq_dim] = cap
    out = ref.new_zeros(*shape)
    if buf is None or cur == 0:
        return out
    ncopy = cur if filled is None else min(cur, int(filled))
    if ncopy > 0:
        src = [slice(None)] * buf.ndim
        dst = [slice(None)] * out.ndim
        src[seq_dim] = slice(0, ncopy)
        dst[seq_dim] = slice(0, ncopy)
        out[tuple(dst)].copy_(buf[tuple(src)])
    return out


class SeqBuf:
    """沿 ``seq_dim`` 的可扩缓冲；``view()`` 只露出 ``[:filled]``。"""

    def __init__(
        self,
        *,
        known_max: int | None,
        seq_dim: int = 1,
        like: torch.Tensor,
        fill: int | float | None = None,
    ) -> None:
        self.known_max = known_max
        self.seq_dim = int(seq_dim)
        self.filled = 0
        self._fill = fill
        cap = alloc_capacity(0, known_max)
        shape = list(like.shape)
        if self.seq_dim < 0 or self.seq_dim >= like.ndim:
            raise ValueError(f"seq_dim={self.seq_dim} 超出 ndim={like.ndim}")
        shape[self.seq_dim] = cap
        if fill is None:
            self.buf = like.new_zeros(*shape)
        else:
            self.buf = like.new_full(shape, fill)

    def view(self) -> torch.Tensor:
        sl = [slice(None)] * self.buf.ndim
        sl[self.seq_dim] = slice(0, self.filled)
        return self.buf[tuple(sl)]

    def _ensure(self, need: int) -> None:
        old_cap = int(self.buf.size(self.seq_dim))
        self.buf = ensure_seq_buf(
            self.buf,
            need,
            known_max=self.known_max,
            seq_dim=self.seq_dim,
            filled=self.filled,
            like=self.buf,
        )
        if self._fill is not None:
            new_cap = int(self.buf.size(self.seq_dim))
            if new_cap > old_cap:
                sl = [slice(None)] * self.buf.ndim
                sl[self.seq_dim] = slice(old_cap, new_cap)
                self.buf[tuple(sl)].fill_(self._fill)

    def replace(self, src: torch.Tensor) -> torch.Tensor:
        n = int(src.size(self.seq_dim))
        self._ensure(n)
        if n > 0:
            sl = [slice(None)] * self.buf.ndim
            sl[self.seq_dim] = slice(0, n)
            self.buf[tuple(sl)].copy_(src)
        self.filled = n
        return self.view()

    def append(self, src: torch.Tensor) -> torch.Tensor:
        n = int(src.size(self.seq_dim))
        need = self.filled + n
        self._ensure(need)
        if n > 0:
            sl = [slice(None)] * self.buf.ndim
            sl[self.seq_dim] = slice(self.filled, need)
            self.buf[tuple(sl)].copy_(src)
        self.filled = need
        return self.view()
