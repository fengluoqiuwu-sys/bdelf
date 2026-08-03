"""BDELF inference acceleration: prefix/suffix dual streams + stride-level prefix cache (P0-P4).

条件（time / SC-CFG / mode）以 in-context token 形式只注入 suffix 流；
已解码 prefix 不依赖 t，仍可缓存（对齐「无 AdaLN」后的 ELF 式条件）。
"""

from __future__ import annotations

import torch

from models.bd3lm.model import bool_mask_to_sdpa_additive
from models.model import sample_from_logits
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
  *,
  cond_len: int = 0,
) -> torch.Tensor:
  """Q=[cond|suffix], KV=[prefix|cond|suffix]（cond 双向可见）。"""
  full = build_window_pair_mask(0, win_len, db, device)
  q_content = suffix_full_indices(win_len, stride, db, device)
  if stride == 0:
    kv_content = q_content
  else:
    kv_content = torch.cat([
      prefix_full_indices(win_len, stride, db, device),
      q_content,
    ])
  content = full[q_content[:, None], kv_content[None, :]]

  if cond_len <= 0:
    return bool_mask_to_sdpa_additive(content)

  q_len = cond_len + content.size(0)
  kv_len = cond_len + kv_content.numel()
  # KV 布局：[prefix_content | cond | suffix]；Q 布局：[cond | suffix]
  # prefix 段长度：
  prefix_kv = 0 if stride == 0 else 2 * stride * db
  out = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
  # content→content 子块：Q 的 content 对 KV 的 [prefix|suffix]
  # KV indices after inserting cond after prefix:
  #   [0, prefix_kv) = prefix
  #   [prefix_kv, prefix_kv+cond_len) = cond
  #   [prefix_kv+cond_len, kv_len) = suffix
  # content mask 原布局 KV=[prefix|suffix]，需拆到新布局。
  q_off = cond_len
  if prefix_kv > 0:
    out[q_off:, :prefix_kv] = content[:, :prefix_kv]
    out[q_off:, prefix_kv + cond_len:] = content[:, prefix_kv:]
  else:
    out[q_off:, cond_len:] = content
  # cond 行：可见全部 KV；content 行：可见全部 cond（已由 ones 覆盖）
  return bool_mask_to_sdpa_additive(out)


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
    d = backbone.n_embd

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
    z_p = self.bb.drop(self.z_buf[:, :stride * self.db])
    x0_p = self.bb.drop(self.x0_buf[:, :stride * self.db])
    return torch.cat([z_p, x0_p], dim=1)

  def _gather_suffix_pair(self, z_block: torch.Tensor) -> torch.Tensor:
    z = self.bb.drop(z_block)
    return torch.cat([z, z], dim=1)

  def _prefix_positions(self, stride: int) -> torch.Tensor:
    return pair_positions(stride * self.db, self.device)

  def _suffix_positions(self, stride: int, cond_len: int) -> torch.Tensor:
    """cond 用 position=0；suffix pair 用全局块位置。"""
    content = pair_positions(self.db, self.device, start=stride * self.db)
    if cond_len <= 0:
      return content
    pref = torch.zeros(cond_len, device=self.device, dtype=torch.long)
    return torch.cat([pref, content])

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

    if self._prefix_layer_x is None:
      win_len = self.db
      mask = build_suffix_cross_mask(win_len, 0, self.db, self.device, cond_len=0)
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
    suffix_pos = pair_positions(self.db, self.device, start=stride * self.db)

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
    sc_prev: torch.Tensor | None = None,
    self_cond_cfg_scale: torch.Tensor | None = None,
  ) -> torch.Tensor:
    win_len = (stride + 1) * self.db
    z_in = self.bb._apply_self_cond(z_block, sc_prev)
    suffix_pair = self._gather_suffix_pair(z_in)
    cond = self.bb.build_context(
      t, self_cond_cfg_scale=self_cond_cfg_scale, decode=decode,
    ).to(dtype=suffix_pair.dtype)
    cond_len = cond.size(1)
    x = torch.cat([cond, suffix_pair], dim=1)

    cross_mask = self._get_mask(
      ("suffix", win_len, stride, cond_len),
      lambda: build_suffix_cross_mask(
        win_len, stride, self.db, self.device, cond_len=cond_len,
      ),
    )
    suffix_pos = self._suffix_positions(stride, cond_len)

    # Prefix KV 在 KV 中排在 cond 之前；cond 的 K/V 与 suffix 一起现算。
    # forward_suffix_cross: K = cat([k_prefix, k_from_x])，x=[cond|suffix]
    # → KV = [prefix | cond | suffix]，与 mask 一致。
    for layer_idx, block in enumerate(self.bb.h):
      k_p, v_p = (None, None)
      if self._prefix_kv_cache:
        k_p, v_p = self._prefix_kv_cache[layer_idx]
      x = block.forward_infer_suffix(x, k_p, v_p, cross_mask, suffix_pos)
    x = self.bb.ln_f(x)
    # z 半：跳过 cond，取 suffix 前 db
    return self.bb.x_pred_head(x[:, cond_len:cond_len + self.db])

  def ode_step(
    self,
    z_block: torch.Tensor,
    stride: int,
    t: torch.Tensor,
    t_next: torch.Tensor,
    *,
    sc_prev: torch.Tensor | None = None,
    self_cond_cfg_scale: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    self.write_suffix(z_block, stride)
    bsz = z_block.size(0)
    t_batch = t.expand(bsz)
    x_pred_block = self._suffix_forward(
      z_block, stride, t_batch, decode=False,
      sc_prev=sc_prev,
      self_cond_cfg_scale=self_cond_cfg_scale,
    )
    denom = torch.clamp(1.0 - t, min=self.bb.t_eps)
    v = (x_pred_block - z_block) / denom
    return z_block + (t_next - t) * v, x_pred_block

  def decode_block(
    self,
    z_block: torch.Tensor,
    stride: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    self_cond_cfg_scale: torch.Tensor | None = None,
  ) -> torch.Tensor:
    self.write_suffix(z_block, stride)
    bsz = z_block.size(0)
    t_batch = torch.ones(bsz, device=self.device, dtype=z_block.dtype)
    # decode 头用 lm_head；先取 hidden 再投影
    win_len = (stride + 1) * self.db
    z_in = self.bb._apply_self_cond(z_block, None)
    suffix_pair = self._gather_suffix_pair(z_in)
    cond = self.bb.build_context(
      t_batch, self_cond_cfg_scale=self_cond_cfg_scale, decode=True,
    ).to(dtype=suffix_pair.dtype)
    cond_len = cond.size(1)
    x = torch.cat([cond, suffix_pair], dim=1)
    cross_mask = self._get_mask(
      ("suffix", win_len, stride, cond_len),
      lambda: build_suffix_cross_mask(
        win_len, stride, self.db, self.device, cond_len=cond_len,
      ),
    )
    suffix_pos = self._suffix_positions(stride, cond_len)
    for layer_idx, block in enumerate(self.bb.h):
      k_p, v_p = (None, None)
      if self._prefix_kv_cache:
        k_p, v_p = self._prefix_kv_cache[layer_idx]
      x = block.forward_infer_suffix(x, k_p, v_p, cross_mask, suffix_pos)
    x = self.bb.ln_f(x)
    hidden = x[:, cond_len:cond_len + self.db]
    return sample_from_logits(
      self.bb.lm_head(hidden),
      temperature=temperature,
      top_k=top_k,
    )

  def on_stride_complete(self, block_emb: torch.Tensor, stride: int) -> None:
    self._extend_prefix_cache(block_emb, stride)

  def append_tokens(self, block_tokens: torch.Tensor) -> None:
    bsz, db = block_tokens.shape
    off = self.token_len
    self.tokens_buf[:, off:off + db] = block_tokens
    self.token_len += db

  def tokens(self) -> torch.Tensor:
    return self.tokens_buf[:, : self.token_len]
