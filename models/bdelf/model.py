"""Block Diffusion + Embedded Language Flow (BDELF).

Within the BD3LM block-diffusion attention framework, replaces in-block MDLM
discrete masking with ELF-style continuous flow matching:
  - Training: 80% denoising branch (MSE, [z_t, x0] concat + block-diffusion mask)
  - Training: 20% decoding branch (CE, t=1 corrupted embedding -> token)
  - Inference: semi-AR per-block ODE integration + final decode

References:
  - Block Diffusion: https://arxiv.org/abs/2503.09573
  - ELF: https://arxiv.org/abs/2605.10938
"""

from __future__ import annotations

import math
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bd3lm.model import (
    FLEX_ATTN_AVAILABLE,
    BlockDiffusionAttention,
    MLP,
    block_diff_mask,
    bool_mask_to_sdpa_additive,
    build_block_diff_mask,
)
from models.bdelf.config import FL_BDELFConfig
from models.bdelf.infer import BDELFInferState, build_window_pair_mask
from models.model import FL_PreTrainedModel, ensure_token_layout, split_model_cfg
from models.rope import pair_positions, window_positions
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg

try:
  from torch.nn.attention.flex_attention import create_block_mask

  _FLEX_IMPORT_OK = True
except ImportError:
  _FLEX_IMPORT_OK = False


class TimestepEmbedder(nn.Module):
  """Sinusoidal timestep embedding (DiT / ELF style)."""

  def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size),
    )
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
      -math.log(max_period)
      * torch.arange(0, half, device=t.device, dtype=torch.float32)
      / half
    )
    args = t.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
    return self.mlp(t_emb)


class FlowBlock(nn.Module):
  """Transformer block with timestep conditioning."""

  def __init__(
    self, n_embd: int, n_head: int, dropout: float, attn_backend: str = "flex",
  ) -> None:
    super().__init__()
    self.ln_1 = nn.LayerNorm(n_embd)
    self.attn = BlockDiffusionAttention(n_embd, n_head, dropout, attn_backend)
    self.ln_2 = nn.LayerNorm(n_embd)
    self.mlp = MLP(n_embd, dropout)
    self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(n_embd, n_embd * 2))

  def forward(
    self,
    x: torch.Tensor,
    cond: torch.Tensor,
    flex_block_mask=None,
    sdpa_attn_mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    scale, shift = self.ada_ln(cond).chunk(2, dim=-1)
    h = self.ln_1(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = x + self.attn(h, flex_block_mask, sdpa_attn_mask, positions)
    h = self.ln_2(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = x + self.mlp(h)
    return x

  def forward_infer_prefix(
    self,
    x: torch.Tensor,
    sdpa_attn_mask: torch.Tensor,
    positions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inference prefix stream: no AdaLN; prefix hidden states do not vary with ODE timestep t."""
    h = self.ln_1(x)
    attn_out, k, v = self.attn.forward_prefix_infer(h, sdpa_attn_mask, positions)
    x = x + attn_out
    x = x + self.mlp(self.ln_2(x))
    return x, k, v

  def forward_infer_suffix(
    self,
    x: torch.Tensor,
    cond: torch.Tensor,
    k_prefix: torch.Tensor | None,
    v_prefix: torch.Tensor | None,
    sdpa_attn_mask: torch.Tensor,
    positions: torch.Tensor,
  ) -> torch.Tensor:
    """Inference suffix stream: only the current block is modulated by AdaLN(t)."""
    scale, shift = self.ada_ln(cond).chunk(2, dim=-1)
    h = self.ln_1(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = x + self.attn.forward_suffix_cross_infer(
      h, k_prefix, v_prefix, sdpa_attn_mask, positions,
    )
    h = self.ln_2(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = x + self.mlp(h)
    return x

  def forward_infer_append(
    self,
    x_new: torch.Tensor,
    x_prefix: torch.Tensor,
    prefix_self_mask: torch.Tensor,
    cross_mask: torch.Tensor,
    prefix_positions: torch.Tensor,
    suffix_positions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """P4: propagate only the new block through layers and append to prefix hidden states."""
    h_old = self.ln_1(x_prefix)
    _, k_old, v_old = self.attn.forward_prefix_infer(
      h_old, prefix_self_mask, prefix_positions,
    )
    h_new = self.ln_1(x_new)
    attn_out = self.attn.forward_suffix_cross_infer(
      h_new, k_old, v_old, cross_mask, suffix_positions,
    )
    x_new = x_new + attn_out
    x_new = x_new + self.mlp(self.ln_2(x_new))
    return torch.cat([x_prefix, x_new], dim=1), x_new


class _BDELFBackbone(nn.Module):
  """Block diffusion continuous-flow LM backbone used for training."""

  full_sequence_training = True
  dual_branch_logging = True

  def __init__(
    self,
    token_layout: FL_TokenLayout,
    max_seq_len: int = 4096,
    diffusion_block_size: int = 32,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 672,
    dropout: float = 0.1,
    attn_backend: str = "flex",
    denoiser_p_mean: float = -1.5,
    denoiser_p_std: float = 0.8,
    denoiser_noise_scale: float = 2.0,
    decoder_prob: float = 0.2,
    decoder_p_mean: float = 0.8,
    decoder_p_std: float = 0.8,
    decoder_noise_scale: float = 5.0,
    t_eps: float = 0.05,
    time_schedule: str = "logit_normal",
    fix_bos: bool = True,
  ) -> None:
    super().__init__()
    if attn_backend == "flex" and not FLEX_ATTN_AVAILABLE:
      raise RuntimeError(
        "attn_backend=flex requires PyTorch FlexAttention; upgrade PyTorch or use sdpa"
      )
    if attn_backend not in ("flex", "sdpa"):
      raise ValueError(f"unknown attn_backend: {attn_backend}")

    self.max_seq_len = max_seq_len
    self.diffusion_block_size = diffusion_block_size
    self.attn_backend = attn_backend
    self.token_layout = token_layout
    self.vocab_size = token_layout.vocab_size
    self.fix_bos = fix_bos
    self.denoiser_p_mean = denoiser_p_mean
    self.denoiser_p_std = denoiser_p_std
    self.denoiser_noise_scale = denoiser_noise_scale
    self.decoder_prob = decoder_prob
    self.decoder_p_mean = decoder_p_mean
    self.decoder_p_std = decoder_p_std
    self.decoder_noise_scale = decoder_noise_scale
    self.t_eps = t_eps
    self.time_schedule = time_schedule

    self.wte = nn.Embedding(token_layout.vocab_size, n_embd)
    self.drop = nn.Dropout(dropout)
    self.time_embed = TimestepEmbedder(n_embd)
    self.mode_denoise = nn.Parameter(torch.randn(1, 1, n_embd) * 0.02)
    self.mode_decode = nn.Parameter(torch.randn(1, 1, n_embd) * 0.02)

    self.h = nn.ModuleList(
      FlowBlock(n_embd, n_head, dropout, attn_backend) for _ in range(n_layer)
    )
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, token_layout.vocab_size, bias=False)
    self.lm_head.weight = self.wte.weight

    self.apply(self._init_weights)

    self._pair_sdpa_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
    self._flex_block_mask_cache: dict[tuple[int, torch.device], object] = {}
    self.last_loss_branch = "mixed"

  def _validate_seq_len(self, seq_len: int) -> None:
    if seq_len > self.max_seq_len:
      raise ValueError(
        f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
      )
    db = self.diffusion_block_size
    if seq_len % db != 0:
      raise ValueError(
        f"sequence length {seq_len} must be divisible by diffusion_block_size ({db})"
      )

  def _init_weights(self, module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def _get_pair_sdpa_mask(self, n: int, device: torch.device) -> torch.Tensor:
    key = (n, device)
    cached = self._pair_sdpa_mask_cache.get(key)
    if cached is None:
      bool_mask = build_block_diff_mask(n, self.diffusion_block_size, device)
      cached = bool_mask_to_sdpa_additive(bool_mask)
      if len(self._pair_sdpa_mask_cache) >= 32:
        self._pair_sdpa_mask_cache.pop(next(iter(self._pair_sdpa_mask_cache)))
      self._pair_sdpa_mask_cache[key] = cached
    return cached

  def _get_flex_block_mask(self, n: int, device: torch.device):
    key = (n, device)
    cached = self._flex_block_mask_cache.get(key)
    if cached is None:
      mask_mod = partial(
        block_diff_mask,
        block_size=self.diffusion_block_size,
        n=n,
      )
      cached = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=n * 2, KV_LEN=n * 2, device=device,
      )
      if len(self._flex_block_mask_cache) >= 32:
        self._flex_block_mask_cache.pop(next(iter(self._flex_block_mask_cache)))
      self._flex_block_mask_cache[key] = cached
    return cached

  def _get_infer_pair_sdpa_mask(
    self, window_len: int, device: torch.device,
  ) -> torch.Tensor:
    key = ("infer", window_len, device)
    cached = self._pair_sdpa_mask_cache.get(key)
    if cached is None:
      bool_mask = build_window_pair_mask(
        0, window_len, self.diffusion_block_size, device,
      )
      cached = bool_mask_to_sdpa_additive(bool_mask)
      if len(self._pair_sdpa_mask_cache) >= 32:
        self._pair_sdpa_mask_cache.pop(next(iter(self._pair_sdpa_mask_cache)))
      self._pair_sdpa_mask_cache[key] = cached
    return cached

  def _tokens_to_emb(self, idx: torch.Tensor) -> torch.Tensor:
    return self.wte(idx)

  def _embed_continuous_pair(
    self, z_half: torch.Tensor, x0_half: torch.Tensor,
  ) -> torch.Tensor:
    """Concatenate [z_t, x0] (RoPE is applied inside attention)."""
    z = self.drop(z_half)
    x0 = self.drop(x0_half)
    return torch.cat([z, x0], dim=1)

  def _backbone(
    self,
    pair_emb: torch.Tensor,
    t: torch.Tensor,
    *,
    decode: bool = False,
    window_len: int | None = None,
    window_start: int = 0,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Block-diffusion backbone.

    Args:
      pair_emb: (B, 2*L, D); L varies per batch during training; at inference L is the
        current full-length prefix
      t: (B,) timesteps
      decode: whether to run in decode mode (output logits)
      window_len: single-half length at inference; None means training
    """
    bsz = pair_emb.size(0)
    cond = self.time_embed(t)
    mode = self.mode_decode if decode else self.mode_denoise
    x = pair_emb + mode.expand(bsz, pair_emb.size(1), -1)

    half = pair_emb.size(1) // 2
    if window_len is None:
      positions = pair_positions(half, x.device)
    else:
      positions = pair_positions(window_len, x.device, start=window_start)
    flex_mask = None
    sdpa_mask = None
    if self.attn_backend == "flex":
      if window_len is None:
        flex_mask = self._get_flex_block_mask(half, x.device)
      else:
        flex_mask = self._get_flex_block_mask(window_len, x.device)
    elif window_len is None:
      sdpa_mask = self._get_pair_sdpa_mask(half, x.device)
    else:
      sdpa_mask = self._get_infer_pair_sdpa_mask(window_len, x.device)

    for block in self.h:
      x = block(x, cond, flex_mask, sdpa_mask, positions)

    x = self.ln_f(x)
    x_pred = x[:, :half]
    logits = self.lm_head(x_pred) if decode else None
    return x_pred, logits

  def _sample_train_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
    if self.time_schedule == "logit_normal":
      z = torch.randn(batch_size, device=device) * self.denoiser_p_std + self.denoiser_p_mean
      t = torch.sigmoid(z)
    else:
      t = torch.rand(batch_size, device=device)
    return t.clamp(max=1.0 - self.t_eps)

  def _sample_decoder_lambda(
    self, shape: tuple[int, ...], device: torch.device,
  ) -> torch.Tensor:
    z = torch.randn(shape, device=device) * self.decoder_p_std + self.decoder_p_mean
    return torch.sigmoid(z)

  def _fix_bos_emb(self, z: torch.Tensor, x0_emb: torch.Tensor) -> torch.Tensor:
    if self.fix_bos:
      z = z.clone()
      z[:, 0] = x0_emb[:, 0]
    return z

  def _denoise_loss(self, x0_emb: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, _ = x0_emb.shape
    t = self._sample_train_t(bsz, x0_emb.device)
    t_exp = t[:, None, None]
    noise = torch.randn_like(x0_emb) * self.denoiser_noise_scale
    z_t = t_exp * x0_emb + (1.0 - t_exp) * noise
    z_t = self._fix_bos_emb(z_t, x0_emb)
    pair = self._embed_continuous_pair(z_t, x0_emb)
    x_pred, _ = self._backbone(pair, t, decode=False)
    weight = 1.0 / torch.clamp(1.0 - t_exp, min=self.t_eps) ** 2
    return (weight * (x_pred - x0_emb).pow(2)).mean()

  def _decode_loss(self, x0_emb: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, _ = x0_emb.shape
    lam = self._sample_decoder_lambda((bsz, seq_len, 1), x0_emb.device)
    noise = torch.randn_like(x0_emb) * self.decoder_noise_scale
    z_tilde = lam * x0_emb + (1.0 - lam) * noise
    z_tilde = self._fix_bos_emb(z_tilde, x0_emb)
    pair = self._embed_continuous_pair(z_tilde, x0_emb)
    t = torch.ones(bsz, device=x0_emb.device)
    _, logits = self._backbone(pair, t, decode=True)
    return F.cross_entropy(
      logits.reshape(-1, self.vocab_size),
      tokens.reshape(-1),
      ignore_index=self.token_layout.ignore_index,
    )

  def forward(
    self,
    idx: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    branch: Literal["denoise", "decode"] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Training/evaluation forward pass.

    Args:
      branch: when ``None``, randomly choose a branch per ``decoder_prob``;
        ``"denoise"`` / ``"decode"`` fix the branch for eval or debugging.
    """
    del targets
    bsz, seq_len = idx.shape
    self._validate_seq_len(seq_len)
    x0_emb = self._tokens_to_emb(idx)
    if branch == "decode":
      loss = self._decode_loss(x0_emb, idx)
      self.last_loss_branch = "decode"
    elif branch == "denoise":
      loss = self._denoise_loss(x0_emb, idx)
      self.last_loss_branch = "denoise"
    else:
      denoise_loss = self._denoise_loss(x0_emb, idx)
      decode_loss = self._decode_loss(x0_emb, idx)
      p = self.decoder_prob
      loss = (1.0 - p) * denoise_loss + p * decode_loss
      self.last_loss_branch = "mixed"
    return torch.empty(0), loss

  # -------------------------------------------------------------------------
  # Inference: semi-AR in-block ODE + decode
  # -------------------------------------------------------------------------

  def _get_sampling_steps(self, num_steps: int, device: torch.device) -> torch.Tensor:
    """Build ODE time grid; logit-normal matches ELF Appendix C.2."""
    schedule = getattr(self, "_infer_time_schedule", self.time_schedule)
    if schedule == "logit_normal":
      z = (
        torch.randn(num_steps - 1, device=device) * self.denoiser_p_std
        + self.denoiser_p_mean
      )
      interior = torch.sigmoid(z).sort().values
      return torch.cat(
        [torch.zeros(1, device=device), interior, torch.ones(1, device=device)],
      )
    return torch.linspace(0.0, 1.0, num_steps + 1, device=device)

  @torch.no_grad()
  def _ode_step_block(
    self,
    z: torch.Tensor,
    x0_ctx: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    window_start: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-block Euler ODE step."""
    bsz, win_len, _ = z.shape
    pair = self._embed_continuous_pair(z, x0_ctx)
    t_batch = t.expand(bsz)
    x_pred, _ = self._backbone(
      pair, t_batch, decode=False,
      window_len=win_len, window_start=window_start,
    )
    denom = torch.clamp(1.0 - t, min=self.t_eps)
    v = (x_pred - z) / denom
    z_new = z + (t_next - t) * v
    return z_new, x_pred

  @torch.no_grad()
  def _decode_block(
    self,
    z: torch.Tensor,
    x0_ctx: torch.Tensor,
    window_start: int,
  ) -> torch.Tensor:
    bsz, win_len, _ = z.shape
    pair = self._embed_continuous_pair(z, x0_ctx)
    t_batch = torch.ones(bsz, device=z.device)
    _, logits = self._backbone(
      pair, t_batch, decode=True,
      window_len=win_len, window_start=window_start,
    )
    return logits[:, -self.diffusion_block_size:].argmax(dim=-1)

  @torch.no_grad()
  def _build_window(
    self,
    emb_accum: torch.Tensor,
    z_block: torch.Tensor,
    start_idx: int,
    end_idx: int,
    device: torch.device,
    dtype: torch.dtype,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build full-length prefix z/x0 halves (each win_len); window_start is always 0."""
    n_samples = z_block.size(0)
    win_len = end_idx - start_idx
    db = self.diffusion_block_size
    ctx_len = emb_accum.size(1)
    z_win = torch.zeros(n_samples, win_len, self.wte.embedding_dim, device=device, dtype=dtype)
    x0_win = torch.zeros_like(z_win)

    g = torch.arange(start_idx, end_idx, device=device)
    valid = g < ctx_len
    if valid.any():
      z_win[:, valid] = emb_accum[:, g[valid]]
      x0_win[:, valid] = emb_accum[:, g[valid]]

    cur_off = end_idx - db - start_idx
    z_win[:, cur_off:cur_off + db] = z_block
    x0_win[:, cur_off:cur_off + db] = z_block
    return z_win, x0_win

  @torch.no_grad()
  def _semi_ar_flow_sampler(
    self,
    n_samples: int,
    seqlen: int,
    num_ode_steps: int,
    *,
    bos_token_id: int | None = None,
    use_fast_infer: bool = True,
  ) -> tuple[torch.Tensor, int]:
    bos = self.token_layout.bos_token_id if bos_token_id is None else bos_token_id
    db = self.diffusion_block_size
    device = next(self.parameters()).device
    dtype = next(self.parameters()).dtype
    num_strides = seqlen // db
    nfe = 0
    t_steps = self._get_sampling_steps(num_ode_steps, device)

    if not use_fast_infer:
      return self._semi_ar_flow_sampler_legacy(
        n_samples, seqlen, num_ode_steps,
        bos_token_id=bos_token_id,
        t_steps=t_steps,
      )

    state = BDELFInferState(self, n_samples, seqlen, device, dtype)
    emb_accum = torch.zeros(
      n_samples, 0, self.wte.embedding_dim, device=device, dtype=dtype,
    )

    for stride in range(num_strides):
      z_block = torch.randn(
        n_samples, db, self.wte.embedding_dim, device=device, dtype=dtype,
      ) * self.denoiser_noise_scale

      if stride == 0 and bos is not None:
        bos_emb = self._tokens_to_emb(
          torch.full((n_samples,), bos, device=device, dtype=torch.long),
        )
        z_block[:, 0] = bos_emb[:, 0]

      state.begin_stride(stride, emb_accum)

      for i in range(len(t_steps) - 1):
        t = t_steps[i]
        t_next = t_steps[i + 1]
        z_block = state.ode_step(z_block, stride, t, t_next)
        nfe += 1

      block_tokens = state.decode_block(z_block, stride)
      nfe += 1

      if stride == 0 and bos is not None:
        block_tokens[:, 0] = bos

      emb_block = self._tokens_to_emb(block_tokens)
      emb_accum = torch.cat([emb_accum, emb_block], dim=1)
      state.on_stride_complete(emb_block, stride)
      state.append_tokens(block_tokens)

    return state.tokens(), nfe

  @torch.no_grad()
  def _semi_ar_flow_sampler_legacy(
    self,
    n_samples: int,
    seqlen: int,
    num_ode_steps: int,
    *,
    bos_token_id: int | None = None,
    t_steps: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, int]:
    """Unoptimized inference path for numerical alignment checks."""
    bos = self.token_layout.bos_token_id if bos_token_id is None else bos_token_id
    db = self.diffusion_block_size
    device = next(self.parameters()).device
    dtype = next(self.parameters()).dtype
    num_strides = seqlen // db
    nfe = 0
    if t_steps is None:
      t_steps = self._get_sampling_steps(num_ode_steps, device)

    tokens = torch.zeros(n_samples, 0, dtype=torch.long, device=device)
    emb_accum = torch.zeros(
      n_samples, 0, self.wte.embedding_dim, device=device, dtype=dtype,
    )

    for stride in range(num_strides):
      z_block = torch.randn(
        n_samples, db, self.wte.embedding_dim, device=device, dtype=dtype,
      ) * self.denoiser_noise_scale

      if stride == 0 and bos is not None:
        bos_emb = self._tokens_to_emb(
          torch.full((n_samples,), bos, device=device, dtype=torch.long),
        )
        z_block[:, 0] = bos_emb[:, 0]

      end_idx = (stride + 1) * db
      start_idx = 0

      for i in range(len(t_steps) - 1):
        t = t_steps[i]
        t_next = t_steps[i + 1]
        z_win, x0_win = self._build_window(
          emb_accum, z_block, start_idx, end_idx, device, dtype,
        )
        z_new, _ = self._ode_step_block(
          z_win, x0_win, t, t_next, window_start=start_idx,
        )
        cur_off = end_idx - db - start_idx
        z_block = z_new[:, cur_off:cur_off + db]
        nfe += 1

      z_win, x0_win = self._build_window(
        emb_accum, z_block, start_idx, end_idx, device, dtype,
      )
      block_tokens = self._decode_block(z_win, x0_win, window_start=start_idx)
      nfe += 1

      if stride == 0 and bos is not None:
        block_tokens[:, 0] = bos

      emb_block = self._tokens_to_emb(block_tokens)
      emb_accum = torch.cat([emb_accum, emb_block], dim=1)
      tokens = torch.cat([tokens, block_tokens], dim=1)

    return tokens, nfe

  @torch.no_grad()
  def generate(
    self,
    num_samples: int = 1,
    seqlen: int | None = None,
    num_steps: int | None = None,
    *,
    bos_token_id: int | None = None,
    sampling_cfg: dict | None = None,
  ) -> tuple[torch.Tensor, int]:
    cfg = sampling_cfg or {}
    if seqlen is None:
      raise ValueError("generate requires an explicit seqlen")
    num_ode_steps = num_steps if num_steps is not None else cfg.get("num_ode_steps", 8)
    bos = self.token_layout.bos_token_id
    if bos_token_id is not None:
      bos = bos_token_id
    infer_schedule = cfg.get("time_schedule")
    if infer_schedule is not None:
      self._infer_time_schedule = infer_schedule

    self._validate_seq_len(seqlen)
    use_fast_infer = cfg.get("use_fast_infer", True)

    return self._semi_ar_flow_sampler(
      n_samples=num_samples,
      seqlen=seqlen,
      num_ode_steps=num_ode_steps,
      bos_token_id=bos,
      use_fast_infer=use_fast_infer,
    )



class FL_BDELFModel(FL_PreTrainedModel):
  config_class = FL_BDELFConfig

  def __init__(self, config: FL_BDELFConfig) -> None:
    super().__init__(config)
    self.backbone = _BDELFBackbone(**config.backbone_kwargs())


def build_model_from_config(config: FL_BDELFConfig) -> FL_BDELFModel:
  ensure_token_layout(config)
  return FL_BDELFModel(config)


def build_model(cfg: dict) -> FL_BDELFModel:
  data, sampling = split_model_cfg(cfg)
  layout = token_layout_from_cfg(data)
  data.pop("tokenizer", None)
  config = FL_BDELFConfig(**data)
  apply_token_layout_to_config(config, layout)
  if sampling is not None:
    config.sampling = sampling
  return build_model_from_config(config)
