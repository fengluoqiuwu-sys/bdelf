"""BDELF inference acceleration: prefix/suffix dual streams + stride-level prefix cache (P0-P4)."""

from __future__ import annotations

import torch

from models.bd3lm.model import bool_mask_to_sdpa_additive
from models.rope import pair_positions


def build_window_pair_mask(
  window_start: int,
  window_len: int,
  diffusion_block_size: int,
  device: torch.device,
) -> torch.Tensor:
  """Sliding-window block-diffusion mask with shape (2*window_len, 2*window_len)."""
  n = window_len
  q = torch.arange(n * 2, device=device)[:, None]
  kv = torch.arange(n * 2, device=device)[None, :]

  def global_pos(idx: torch.Tensor) -> torch.Tensor:
    local = torch.where(idx >= n, idx - n, idx)
    return window_start + local

  gq, gkv = global_pos(q), global_pos(kv)
  x0_q = q >= n
  x0_kv = kv >= n

  block_q = torch.where(
    x0_q, gq // diffusion_block_size, gq // diffusion_block_size,
  )
  block_kv = torch.where(
    x0_kv, gkv // diffusion_block_size, gkv // diffusion_block_size,
  )

  block_diagonal = (block_q == block_kv) & (x0_q == x0_kv)
  offset_block_causal = (
    (block_q > block_kv) & x0_kv & (~x0_q)
  )
  block_causal = (block_q >= block_kv) & x0_kv & x0_q
  return block_diagonal | offset_block_causal | block_causal


def prefix_full_indices(
  win_len: int,
  stride: int,
  db: int,
  device: torch.device,
) -> torch.Tensor:
  """Indices of completed blocks in pair space; length 2 * stride * db."""
  z_idx = torch.arange(0, stride * db, device=device, dtype=torch.long)
  x0_idx = torch.arange(win_len, win_len + stride * db, device=device, dtype=torch.long)
  return torch.cat([z_idx, x0_idx])


def suffix_full_indices(
  win_len: int,
  stride: int,
  db: int,
  device: torch.device,
) -> torch.Tensor:
  """Indices of the current block in pair space; length 2 * db."""
  z_idx = torch.arange(stride * db, (stride + 1) * db, device=device, dtype=torch.long)
  x0_idx = torch.arange(
    win_len + stride * db, win_len + (stride + 1) * db, device=device, dtype=torch.long,
  )
  return torch.cat([z_idx, x0_idx])


def build_prefix_self_mask(
  win_len: int,
  stride: int,
  db: int,
  device: torch.device,
) -> torch.Tensor:
  if stride == 0:
    return bool_mask_to_sdpa_additive(
      torch.zeros(0, 0, dtype=torch.bool, device=device),
    )
  full = build_window_pair_mask(0, win_len, db, device)
  idx = prefix_full_indices(win_len, stride, db, device)
  sub = full[idx[:, None], idx[None, :]]
  return bool_mask_to_sdpa_additive(sub)


def build_suffix_cross_mask(
  win_len: int,
  stride: int,
  db: int,
  device: torch.device,
) -> torch.Tensor:
  full = build_window_pair_mask(0, win_len, db, device)
  q_idx = suffix_full_indices(win_len, stride, db, device)
  if stride == 0:
    kv_idx = q_idx
  else:
    kv_idx = torch.cat([
      prefix_full_indices(win_len, stride, db, device),
      q_idx,
    ])
  sub = full[q_idx[:, None], kv_idx[None, :]]
  return bool_mask_to_sdpa_additive(sub)


def build_append_cross_mask(
  win_len: int,
  stride: int,
  db: int,
  device: torch.device,
) -> torch.Tensor:
  """P4: mask for new-block queries against [old prefix KV; new block KV]."""
  full = build_window_pair_mask(0, win_len, db, device)
  q_idx = suffix_full_indices(win_len, stride, db, device)
  prefix_idx = prefix_full_indices(win_len, stride, db, device)
  kv_idx = torch.cat([prefix_idx, q_idx])
  sub = full[q_idx[:, None], kv_idx[None, :]]
  return bool_mask_to_sdpa_additive(sub)


class BDELFInferState:
  """Inference state: buffer reuse (P0), prefix KV cache (P2/P3), cross-stride incremental updates (P4)."""

  def __init__(
    self,
    backbone,
    n_samples: int,
    seqlen: int,
    device: torch.device,
    dtype: torch.dtype,
  ) -> None:
    self.bb = backbone
    self.db = backbone.diffusion_block_size
    self.n_samples = n_samples
    self.seqlen = seqlen
    self.device = device
    self.dtype = dtype
    d = backbone.wte.embedding_dim

    self.z_buf = torch.zeros(n_samples, seqlen, d, device=device, dtype=dtype)
    self.x0_buf = torch.zeros_like(self.z_buf)
    self.emb_len = 0

    self.tokens_buf = torch.zeros(n_samples, seqlen, dtype=torch.long, device=device)
    self.token_len = 0

    self._prefix_layer_x: list[torch.Tensor] | None = None
    self._prefix_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    self._mask_cache: dict[tuple, torch.Tensor] = {}

  def _get_mask(self, key: tuple, builder) -> torch.Tensor:
    cached = self._mask_cache.get(key)
    if cached is None:
      cached = builder()
      self._mask_cache[key] = cached
    return cached

  def set_emb_accum(self, emb_accum: torch.Tensor) -> None:
    length = emb_accum.size(1)
    self.z_buf[:, :length] = emb_accum
    self.x0_buf[:, :length] = emb_accum
    self.emb_len = length

  def write_suffix(self, z_block: torch.Tensor, stride: int) -> None:
    off = stride * self.db
    self.z_buf[:, off:off + self.db] = z_block
    self.x0_buf[:, off:off + self.db] = z_block

  def _gather_prefix_pair(self, stride: int) -> torch.Tensor | None:
    if stride == 0:
      return None
    win_len = (stride + 1) * self.db
    z_p = self.bb.drop(self.z_buf[:, :stride * self.db])
    x0_p = self.bb.drop(self.x0_buf[:, :stride * self.db])
    return torch.cat([z_p, x0_p], dim=1)

  def _gather_suffix_pair(self, z_block: torch.Tensor) -> torch.Tensor:
    z = self.bb.drop(z_block)
    return torch.cat([z, z], dim=1)

  def _add_mode(self, emb: torch.Tensor, *, decode: bool) -> torch.Tensor:
    mode = self.bb.mode_decode if decode else self.bb.mode_denoise
    return emb + mode.expand(emb.size(0), emb.size(1), -1)

  def _prefix_positions(self, stride: int) -> torch.Tensor:
    return pair_positions(stride * self.db, self.device)

  def _suffix_positions(self, stride: int) -> torch.Tensor:
    return pair_positions(self.db, self.device, start=stride * self.db)

  def _run_prefix_layers(
    self,
    x: torch.Tensor,
    stride: int,
  ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    win_len = (stride + 1) * self.db
    mask = self._get_mask(
      ("prefix", win_len, stride),
      lambda: build_prefix_self_mask(win_len, stride, self.db, self.device),
    )
    pos = self._prefix_positions(stride)
    layer_inputs = [x]
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block in self.bb.h:
      x, k, v = block.forward_infer_prefix(layer_inputs[-1], mask, pos)
      kv_cache.append((k, v))
      layer_inputs.append(x)
    return layer_inputs, kv_cache

  def _extend_prefix_cache(self, block_emb: torch.Tensor, stride: int) -> None:
    """P4: after decode completes, incrementally write the new block embedding into prefix cache."""
    pair_chunk = self._gather_suffix_pair(block_emb)
    pair_chunk = self._add_mode(pair_chunk, decode=False)

    if self._prefix_layer_x is None:
      win_len = self.db
      mask = build_suffix_cross_mask(win_len, 0, self.db, self.device)
      pos = pair_positions(self.db, self.device, start=0)
      layer_inputs = [pair_chunk]
      x = pair_chunk
      for block in self.bb.h:
        x, _, _ = block.forward_infer_prefix(x, mask, pos)
        layer_inputs.append(x)
      self._prefix_layer_x = layer_inputs
      return

    win_len = (stride + 1) * self.db
    prefix_self_mask = self._get_mask(
      ("prefix", win_len, stride),
      lambda: build_prefix_self_mask(win_len, stride, self.db, self.device),
    )
    cross_mask = self._get_mask(
      ("append", win_len, stride),
      lambda: build_append_cross_mask(win_len, stride, self.db, self.device),
    )
    prefix_pos = self._prefix_positions(stride)
    suffix_pos = self._suffix_positions(stride)

    new_layer_x: list[torch.Tensor] = []
    x_new = pair_chunk
    for layer_idx, block in enumerate(self.bb.h):
      x_prefix = self._prefix_layer_x[layer_idx]
      x_prefix, x_new = block.forward_infer_append(
        x_new,
        x_prefix,
        prefix_self_mask,
        cross_mask,
        prefix_pos,
        suffix_pos,
      )
      new_layer_x.append(x_prefix)
    new_layer_x.append(x_new)
    self._prefix_layer_x = new_layer_x

  def begin_stride(self, stride: int, emb_accum: torch.Tensor) -> None:
    self.set_emb_accum(emb_accum)
    win_len = (stride + 1) * self.db

    if stride == 0 or self._prefix_layer_x is None:
      self._prefix_kv_cache = []
      return

    expected_layers = len(self.bb.h) + 1
    if (
      len(self._prefix_layer_x) == expected_layers
      and self._prefix_layer_x[0].size(1) == 2 * stride * self.db
    ):
      kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
      mask = self._get_mask(
        ("prefix", win_len, stride),
        lambda: build_prefix_self_mask(win_len, stride, self.db, self.device),
      )
      pos = self._prefix_positions(stride)
      for layer_idx, block in enumerate(self.bb.h):
        h = block.ln_1(self._prefix_layer_x[layer_idx])
        _, k, v = block.attn.forward_prefix_infer(h, mask, pos)
        kv_cache.append((k, v))
      self._prefix_kv_cache = kv_cache
      return

    prefix_pair = self._gather_prefix_pair(stride)
    if prefix_pair is None:
      self._prefix_kv_cache = []
      return
    prefix_pair = self._add_mode(prefix_pair, decode=False)
    layer_inputs, kv_cache = self._run_prefix_layers(prefix_pair, stride)
    self._prefix_layer_x = layer_inputs
    self._prefix_kv_cache = kv_cache

  def _suffix_forward(
    self,
    z_block: torch.Tensor,
    stride: int,
    t: torch.Tensor,
    *,
    decode: bool,
  ) -> torch.Tensor:
    win_len = (stride + 1) * self.db
    suffix_pair = self._gather_suffix_pair(z_block)
    suffix_pair = self._add_mode(suffix_pair, decode=decode)
    cond = self.bb.time_embed(t)

    cross_mask = self._get_mask(
      ("suffix", win_len, stride),
      lambda: build_suffix_cross_mask(win_len, stride, self.db, self.device),
    )
    suffix_pos = self._suffix_positions(stride)

    x = suffix_pair
    for layer_idx, block in enumerate(self.bb.h):
      k_p, v_p = (None, None)
      if self._prefix_kv_cache:
        k_p, v_p = self._prefix_kv_cache[layer_idx]
      x = block.forward_infer_suffix(
        x, cond, k_p, v_p, cross_mask, suffix_pos,
      )
    x = self.bb.ln_f(x)
    return x[:, : self.db]

  def ode_step(
    self,
    z_block: torch.Tensor,
    stride: int,
    t: torch.Tensor,
    t_next: torch.Tensor,
  ) -> torch.Tensor:
    self.write_suffix(z_block, stride)
    bsz = z_block.size(0)
    t_batch = t.expand(bsz)
    x_pred_block = self._suffix_forward(z_block, stride, t_batch, decode=False)
    denom = torch.clamp(1.0 - t, min=self.bb.t_eps)
    v = (x_pred_block - z_block) / denom
    return z_block + (t_next - t) * v

  def decode_block(self, z_block: torch.Tensor, stride: int) -> torch.Tensor:
    self.write_suffix(z_block, stride)
    bsz = z_block.size(0)
    t_batch = torch.ones(bsz, device=self.device)
    x_pred_block = self._suffix_forward(z_block, stride, t_batch, decode=True)
    return self.bb.lm_head(x_pred_block).argmax(dim=-1)

  def on_stride_complete(self, block_emb: torch.Tensor, stride: int) -> None:
    self._extend_prefix_cache(block_emb, stride)

  def append_tokens(self, block_tokens: torch.Tensor) -> None:
    bsz, db = block_tokens.shape
    off = self.token_len
    self.tokens_buf[:, off:off + db] = block_tokens
    self.token_len += db

  def tokens(self) -> torch.Tensor:
    return self.tokens_buf[:, : self.token_len]
