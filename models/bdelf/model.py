"""Block Diffusion + Embedded Language Flow (BDELF).

Within the BD3LM block-diffusion attention framework, replaces in-block MDLM
discrete masking with ELF-style continuous flow matching:
  - Conditioning: in-context time / mode / SC-CFG tokens（对齐 ELF，无 AdaLN）
  - Training: per-example denoise MSE + decode CE mix（mixed_branch）+ SC-CFG
  - Inference: semi-AR per-block ODE + final decode，支持 self-cond / SC-CFG

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
from models.model import (
  FL_PreTrainedModel,
  ensure_token_layout,
  sample_from_logits,
  split_model_cfg,
)
from models.rope import pair_positions
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg

try:
  from torch.nn.attention.flex_attention import create_block_mask

  _FLEX_IMPORT_OK = True
except ImportError:
  _FLEX_IMPORT_OK = False


def block_diff_mask_with_prefix(
  b, h, q_idx, kv_idx, *, block_size: int, n: int, prefix_len: int,
) -> torch.Tensor:
  """Block-diffusion mask with a bidirectional in-context condition prefix."""
  del b, h
  q_pref = q_idx < prefix_len
  kv_pref = kv_idx < prefix_len
  both_content = (~q_pref) & (~kv_pref)
  q_c = q_idx - prefix_len
  kv_c = kv_idx - prefix_len
  content = block_diff_mask(
    None, None, q_c, kv_c, block_size=block_size, n=n,
  )
  return q_pref | kv_pref | (both_content & content)


def build_block_diff_mask_with_prefix(
  seq_len: int,
  diffusion_block_size: int,
  prefix_len: int,
  device: torch.device,
) -> torch.Tensor:
  """SDPA bool mask of shape (prefix+2*seq_len, prefix+2*seq_len)."""
  total = prefix_len + seq_len * 2
  q_idx = torch.arange(total, device=device)[:, None]
  kv_idx = torch.arange(total, device=device)[None, :]
  return block_diff_mask_with_prefix(
    None, None, q_idx, kv_idx,
    block_size=diffusion_block_size, n=seq_len, prefix_len=prefix_len,
  )


def pair_positions_with_prefix(
  n: int, prefix_len: int, device: torch.device, start: int = 0,
) -> torch.Tensor:
  """Prefix 用 position=0（RoPE 恒等）；内容半段用 pair_positions。"""
  if prefix_len <= 0:
    return pair_positions(n, device, start=start)
  pref = torch.zeros(prefix_len, device=device, dtype=torch.long)
  return torch.cat([pref, pair_positions(n, device, start=start)])


class TimestepEmbedder(nn.Module):
  """Sinusoidal timestep embedding（DiT / ELF style）。"""

  def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size),
    )
    self.frequency_embedding_size = frequency_embedding_size
    nn.init.normal_(self.mlp[0].weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.mlp[0].bias)
    nn.init.normal_(self.mlp[2].weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.mlp[2].bias)

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
  """Transformer block；无 AdaLN，条件全部走 in-context tokens（对齐 ELF）。"""

  def __init__(
    self, n_embd: int, n_head: int, dropout: float, attn_backend: str = "flex",
  ) -> None:
    super().__init__()
    self.ln_1 = nn.LayerNorm(n_embd)
    self.attn = BlockDiffusionAttention(n_embd, n_head, dropout, attn_backend)
    self.ln_2 = nn.LayerNorm(n_embd)
    self.mlp = MLP(n_embd, dropout)

  def forward(
    self,
    x: torch.Tensor,
    flex_block_mask=None,
    sdpa_attn_mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    x = x + self.attn(self.ln_1(x), flex_block_mask, sdpa_attn_mask, positions)
    x = x + self.mlp(self.ln_2(x))
    return x

  def forward_infer_prefix(
    self,
    x: torch.Tensor,
    sdpa_attn_mask: torch.Tensor,
    positions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inference prefix：已解码干净块，不依赖 ODE 时间 t。"""
    h = self.ln_1(x)
    attn_out, k, v = self.attn.forward_prefix_infer(h, sdpa_attn_mask, positions)
    x = x + attn_out
    x = x + self.mlp(self.ln_2(x))
    return x, k, v

  def forward_infer_suffix(
    self,
    x: torch.Tensor,
    k_prefix: torch.Tensor | None,
    v_prefix: torch.Tensor | None,
    sdpa_attn_mask: torch.Tensor,
    positions: torch.Tensor,
  ) -> torch.Tensor:
    """Inference suffix：[cond|current-block]；条件 token 每步重算。"""
    h = self.ln_1(x)
    x = x + self.attn.forward_suffix_cross_infer(
      h, k_prefix, v_prefix, sdpa_attn_mask, positions,
    )
    x = x + self.mlp(self.ln_2(x))
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
    """P4：新块写入 prefix 缓存。"""
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
  """Block diffusion continuous-flow LM；ELF 式 in-context 条件 + SC-CFG。"""

  full_sequence_training = True
  dual_branch_logging = True
  # 与 ELF 一致：单次 forward 内按样本混合 denoise/decode。
  mixed_branch_training = True

  def __init__(
    self,
    token_layout: FL_TokenLayout,
    max_seq_len: int = 4096,
    diffusion_block_size: int = 32,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 672,
    dropout: float = 0.0,
    attn_backend: str = "flex",
    num_time_tokens: int = 4,
    num_self_cond_cfg_tokens: int = 0,
    num_model_mode_tokens: int = 4,
    self_cond_prob: float = 0.5,
    self_cond_cfg_min: float = 0.5,
    self_cond_cfg_max: float = 5.0,
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
    if num_time_tokens <= 0:
      raise ValueError("num_time_tokens must be positive")

    self.max_seq_len = max_seq_len
    self.diffusion_block_size = diffusion_block_size
    self.attn_backend = attn_backend
    self.token_layout = token_layout
    self.vocab_size = token_layout.vocab_size
    self.n_embd = n_embd
    self.fix_bos = fix_bos
    self.num_time_tokens = num_time_tokens
    self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
    self.num_model_mode_tokens = num_model_mode_tokens
    self.self_cond_prob = self_cond_prob
    self.self_cond_cfg_min = self_cond_cfg_min
    self.self_cond_cfg_max = self_cond_cfg_max
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

    self.t_embedder = TimestepEmbedder(n_embd)
    self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, n_embd))
    nn.init.normal_(self.t_emb_tokens, mean=0.0, std=0.02)

    if num_self_cond_cfg_tokens > 0:
      self.self_cond_cfg_embedder = TimestepEmbedder(n_embd)
      self.self_cond_cfg_tokens = nn.Parameter(
        torch.empty(1, num_self_cond_cfg_tokens, n_embd)
      )
      nn.init.normal_(self.self_cond_cfg_tokens, mean=0.0, std=0.02)
    else:
      self.self_cond_cfg_embedder = None
      self.self_cond_cfg_tokens = None

    if num_model_mode_tokens > 0:
      self.mode_tokens = nn.Parameter(
        torch.empty(1, num_model_mode_tokens, n_embd)
      )
      nn.init.normal_(self.mode_tokens, mean=0.0, std=0.02)
    else:
      self.mode_tokens = None

    self.self_cond_proj = nn.Linear(2 * n_embd, n_embd, bias=True)
    nn.init.xavier_uniform_(self.self_cond_proj.weight)
    nn.init.zeros_(self.self_cond_proj.bias)

    # Mid-depth dropout（官方 ELF）：仅 [depth/4, 3*depth/4) 层启用。
    q1, q3 = n_layer // 4, n_layer // 4 * 3
    blocks = []
    for i in range(n_layer):
      layer_drop = dropout if (q3 > i >= q1) else 0.0
      blocks.append(FlowBlock(n_embd, n_head, layer_drop, attn_backend))
    self.h = nn.ModuleList(blocks)

    self.ln_f = nn.LayerNorm(n_embd)
    # ELF FinalLayer 风格：零初始化 x-pred 头，起步接近恒等流。
    self.x_pred_head = nn.Linear(n_embd, n_embd, bias=True)
    nn.init.zeros_(self.x_pred_head.weight)
    nn.init.zeros_(self.x_pred_head.bias)
    self.lm_head = nn.Linear(n_embd, token_layout.vocab_size, bias=False)
    self.lm_head.weight = self.wte.weight

    self.apply(self._init_weights)
    # 覆盖：embedding / x_pred_head / 条件 token 已在上面设好。
    nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.x_pred_head.weight)
    nn.init.zeros_(self.x_pred_head.bias)

    self._pair_sdpa_mask_cache: dict[tuple, torch.Tensor] = {}
    self._flex_block_mask_cache: dict[tuple, object] = {}
    self.last_loss_branch = ""
    self.last_l2_loss = float("nan")
    self.last_ce_loss = float("nan")

  @property
  def cond_prefix_len(self) -> int:
    return (
      self.num_time_tokens
      + max(self.num_self_cond_cfg_tokens, 0)
      + max(self.num_model_mode_tokens, 0)
    )

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
    key = ("train", n, self.cond_prefix_len, device)
    cached = self._pair_sdpa_mask_cache.get(key)
    if cached is None:
      bool_mask = build_block_diff_mask_with_prefix(
        n, self.diffusion_block_size, self.cond_prefix_len, device,
      )
      cached = bool_mask_to_sdpa_additive(bool_mask)
      if len(self._pair_sdpa_mask_cache) >= 32:
        self._pair_sdpa_mask_cache.pop(next(iter(self._pair_sdpa_mask_cache)))
      self._pair_sdpa_mask_cache[key] = cached
    return cached

  def _get_flex_block_mask(self, n: int, device: torch.device):
    key = ("train", n, self.cond_prefix_len, device)
    cached = self._flex_block_mask_cache.get(key)
    if cached is None:
      prefix_len = self.cond_prefix_len
      mask_mod = partial(
        block_diff_mask_with_prefix,
        block_size=self.diffusion_block_size,
        n=n,
        prefix_len=prefix_len,
      )
      total = prefix_len + n * 2
      cached = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device,
      )
      if len(self._flex_block_mask_cache) >= 32:
        self._flex_block_mask_cache.pop(next(iter(self._flex_block_mask_cache)))
      self._flex_block_mask_cache[key] = cached
    return cached

  def _get_infer_pair_sdpa_mask(
    self, window_len: int, device: torch.device,
  ) -> torch.Tensor:
    key = ("infer", window_len, self.cond_prefix_len, device)
    cached = self._pair_sdpa_mask_cache.get(key)
    if cached is None:
      content = build_window_pair_mask(
        0, window_len, self.diffusion_block_size, device,
      )
      prefix_len = self.cond_prefix_len
      if prefix_len <= 0:
        cached = bool_mask_to_sdpa_additive(content)
      else:
        total = prefix_len + content.size(0)
        full = torch.ones(total, total, dtype=torch.bool, device=device)
        full[prefix_len:, prefix_len:] = content
        cached = bool_mask_to_sdpa_additive(full)
      if len(self._pair_sdpa_mask_cache) >= 32:
        self._pair_sdpa_mask_cache.pop(next(iter(self._pair_sdpa_mask_cache)))
      self._pair_sdpa_mask_cache[key] = cached
    return cached

  def _tokens_to_emb(self, idx: torch.Tensor) -> torch.Tensor:
    return self.wte(idx)

  def build_context(
    self,
    t: torch.Tensor,
    self_cond_cfg_scale: torch.Tensor | None = None,
    *,
    decode: bool | torch.Tensor = False,
  ) -> torch.Tensor:
    """In-context 条件前缀：time + 可选 SC-CFG + mode（长度固定）。"""
    bsz = t.shape[0]
    time_emb = self.t_embedder(t)
    parts = [self.t_emb_tokens.expand(bsz, -1, -1) + time_emb.unsqueeze(1)]

    if self.num_self_cond_cfg_tokens > 0:
      if self_cond_cfg_scale is None:
        self_cond_cfg_scale = torch.ones(bsz, device=t.device, dtype=t.dtype)
      sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale.float())
      parts.append(
        self.self_cond_cfg_tokens.expand(bsz, -1, -1) + sc_emb.unsqueeze(1)
      )

    if self.mode_tokens is not None:
      mode = self.mode_tokens.expand(bsz, -1, -1)
      if isinstance(decode, torch.Tensor):
        gate = decode.to(dtype=mode.dtype).view(-1, 1, 1)
      else:
        gate = 1.0 if decode else 0.0
      parts.append(mode * gate)

    return torch.cat(parts, dim=1)

  def _apply_self_cond(
    self, z: torch.Tensor, sc: torch.Tensor | None,
  ) -> torch.Tensor:
    if self.self_cond_prob <= 0:
      return z
    sc_half = torch.zeros_like(z) if sc is None else sc
    return self.self_cond_proj(torch.cat([z, sc_half], dim=-1))

  def _embed_continuous_pair(
    self, z_half: torch.Tensor, x0_half: torch.Tensor,
  ) -> torch.Tensor:
    z = self.drop(z_half)
    x0 = self.drop(x0_half)
    return torch.cat([z, x0], dim=1)

  def _backbone(
    self,
    pair_emb: torch.Tensor,
    t: torch.Tensor,
    *,
    decode: bool | torch.Tensor = False,
    self_cond_cfg_scale: torch.Tensor | None = None,
    window_len: int | None = None,
    window_start: int = 0,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Block-diffusion backbone with ELF-style condition prefix.

    Args:
      pair_emb: (B, 2*L, D)
      t: (B,) timesteps
      decode: bool 或 (B,) gate；控制 mode tokens
      window_len: 推理时单半长度；None 表示训练
    """
    bsz = pair_emb.size(0)
    prefix = self.build_context(
      t, self_cond_cfg_scale=self_cond_cfg_scale, decode=decode,
    ).to(dtype=pair_emb.dtype)
    x = torch.cat([prefix, pair_emb], dim=1)
    prefix_len = prefix.size(1)

    half = pair_emb.size(1) // 2
    if window_len is None:
      positions = pair_positions_with_prefix(half, prefix_len, x.device)
    else:
      positions = pair_positions_with_prefix(
        window_len, prefix_len, x.device, start=window_start,
      )

    flex_mask = None
    sdpa_mask = None
    if self.attn_backend == "flex":
      if window_len is None:
        flex_mask = self._get_flex_block_mask(half, x.device)
      else:
        # 推理窗口走 SDPA 扩展 mask（fast infer 在 infer.py 内处理）。
        sdpa_mask = self._get_infer_pair_sdpa_mask(window_len, x.device)
    elif window_len is None:
      sdpa_mask = self._get_pair_sdpa_mask(half, x.device)
    else:
      sdpa_mask = self._get_infer_pair_sdpa_mask(window_len, x.device)

    for block in self.h:
      x = block(x, flex_mask, sdpa_mask, positions)

    x = self.ln_f(x)
    x_content = x[:, prefix_len:]
    x_pred = self.x_pred_head(x_content[:, :half])

    need_logits = (
      True if isinstance(decode, torch.Tensor)
      else bool(decode)
    )
    logits = self.lm_head(x_content[:, :half]) if need_logits else None
    return x_pred, logits

  def _sample_train_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
    if self.time_schedule == "logit_normal":
      z = torch.randn(batch_size, device=device) * self.denoiser_p_std + self.denoiser_p_mean
      return torch.sigmoid(z)
    return torch.rand(batch_size, device=device)

  def _sample_cfg_scale(
    self, batch_size: int, device: torch.device,
  ) -> torch.Tensor:
    """Log-uniform CFG scale in [1+min, 1+max]-1（对齐 ELF）。"""
    u = torch.rand(batch_size, device=device, dtype=torch.float32)
    a = torch.as_tensor(1.0 + self.self_cond_cfg_min, device=device, dtype=u.dtype)
    b = torch.as_tensor(1.0 + self.self_cond_cfg_max, device=device, dtype=u.dtype)
    return (a * torch.exp(u * torch.log(b / a)) - 1.0).to(
      dtype=next(self.parameters()).dtype,
    )

  def _x_to_v(
    self, x_pred: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
  ) -> torch.Tensor:
    t_exp = t.reshape(-1, 1, 1)
    return (x_pred - z) / torch.clamp(1.0 - t_exp, min=self.t_eps)

  def _fix_bos_emb(self, z: torch.Tensor, x0_emb: torch.Tensor) -> torch.Tensor:
    if self.fix_bos:
      z = z.clone()
      z[:, 0] = x0_emb[:, 0]
    return z

  def _denoise_loss(self, x0_emb: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    del tokens
    bsz = x0_emb.shape[0]
    t = self._sample_train_t(bsz, x0_emb.device).to(dtype=x0_emb.dtype)
    t_exp = t[:, None, None]
    noise = torch.randn_like(x0_emb) * self.denoiser_noise_scale
    z_t = t_exp * x0_emb + (1.0 - t_exp) * noise
    z_t = self._fix_bos_emb(z_t, x0_emb)
    v_target = (x0_emb - z_t) / torch.clamp(1.0 - t_exp, min=self.t_eps)

    if self.self_cond_prob > 0:
      use_sc = (
        (torch.rand((bsz,), device=x0_emb.device, dtype=x0_emb.dtype) < self.self_cond_prob)
        .reshape(-1, 1, 1)
        .to(dtype=x0_emb.dtype)
      )
      with torch.no_grad():
        z_in0 = self._apply_self_cond(z_t, None)
        pair0 = self._embed_continuous_pair(z_in0, x0_emb)
        x_init, _ = self._backbone(pair0, t, decode=False)
      sc_half = x_init.detach() * use_sc
      z_in = self._apply_self_cond(z_t, sc_half)
    else:
      z_in = z_t

    pair = self._embed_continuous_pair(z_in, x0_emb)
    x_pred, _ = self._backbone(pair, t, decode=False)
    v_pred = self._x_to_v(x_pred, z_t, t)
    return ((v_pred - v_target) ** 2).mean()

  def _decode_loss(self, x0_emb: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, _ = x0_emb.shape
    lam = torch.sigmoid(
      torch.randn(bsz, seq_len, 1, device=x0_emb.device, dtype=x0_emb.dtype)
      * self.decoder_p_std
      + self.decoder_p_mean
    )
    noise = torch.randn_like(x0_emb) * self.decoder_noise_scale
    z_tilde = lam * x0_emb + (1.0 - lam) * noise
    z_tilde = self._fix_bos_emb(z_tilde, x0_emb)
    t = torch.ones(bsz, device=x0_emb.device, dtype=x0_emb.dtype)
    z_in = self._apply_self_cond(z_tilde, None)
    pair = self._embed_continuous_pair(z_in, x0_emb)
    _, logits = self._backbone(pair, t, decode=True)
    assert logits is not None
    return F.cross_entropy(
      logits.reshape(-1, self.vocab_size),
      tokens.reshape(-1),
      ignore_index=self.token_layout.ignore_index,
    )

  def _mixed_branch_loss(
    self, x0_emb: torch.Tensor, tokens: torch.Tensor,
  ) -> torch.Tensor:
    """Per-example denoise/decode mix + training-time SC-CFG（对齐 ELF）。"""
    bsz, seq_len, _ = x0_emb.shape
    device = x0_emb.device
    dtype = x0_emb.dtype

    t = self._sample_train_t(bsz, device).to(dtype=dtype)
    noise = torch.randn_like(x0_emb) * self.denoiser_noise_scale
    t_exp = t.reshape(-1, 1, 1)
    denoiser_z = t_exp * x0_emb + (1.0 - t_exp) * noise
    denoiser_z = self._fix_bos_emb(denoiser_z, x0_emb)
    v_target = (x0_emb - denoiser_z) / torch.clamp(1.0 - t_exp, min=self.t_eps)

    decoder_step_active = torch.bernoulli(
      torch.full((bsz,), self.decoder_prob, dtype=torch.float32, device=device),
    ).to(dtype=dtype)
    decoder_mask_b11 = decoder_step_active.view(-1, 1, 1)
    decoder_mask_b1 = decoder_step_active.view(-1, 1)

    decoder_lam = torch.sigmoid(
      torch.randn((bsz, seq_len, 1), dtype=dtype, device=device)
      * self.decoder_p_std
      + self.decoder_p_mean
    )
    decoder_noise = torch.randn_like(x0_emb) * self.decoder_noise_scale
    decoder_z = decoder_lam * x0_emb + (1.0 - decoder_lam) * decoder_noise
    decoder_z = self._fix_bos_emb(decoder_z, x0_emb)

    decoder_t = torch.ones_like(t)
    t_mixed = decoder_step_active * decoder_t + (1.0 - decoder_step_active) * t
    z_mixed = decoder_mask_b11 * decoder_z + (1.0 - decoder_mask_b11) * denoiser_z

    self_cond_cfg: torch.Tensor | None = None
    sc_half = torch.zeros_like(denoiser_z)

    if self.num_self_cond_cfg_tokens > 0:
      self_cond_cfg = self._sample_cfg_scale(bsz, device)
      use_sc = (
        (torch.rand((bsz,), device=device, dtype=dtype) < self.self_cond_prob)
        .reshape(-1, 1, 1)
        .to(dtype=dtype)
      )
      with torch.no_grad():
        z_in0 = self._apply_self_cond(denoiser_z, None)
        pair0 = self._embed_continuous_pair(z_in0, x0_emb)
        x_init, _ = self._backbone(
          pair0, t, decode=False, self_cond_cfg_scale=self_cond_cfg,
        )
      v_uncond = self._x_to_v(x_init, denoiser_z, t)
      x_uncond = x_init.detach()
      with torch.no_grad():
        z_in1 = self._apply_self_cond(denoiser_z, x_uncond)
        pair1 = self._embed_continuous_pair(z_in1, x0_emb)
        x_cond, _ = self._backbone(
          pair1, t, decode=False, self_cond_cfg_scale=self_cond_cfg,
        )
      v_cond = self._x_to_v(x_cond, denoiser_z, t)

      sc_w = self_cond_cfg.reshape(-1, 1, 1)
      sc_guidance = (1.0 - 1.0 / sc_w) * (v_cond - v_uncond)
      sc_guidance = torch.where(
        use_sc.bool(), sc_guidance, torch.zeros_like(sc_guidance),
      )
      v_target = (v_target + sc_guidance).detach()
      sc_half = x_uncond * use_sc * (1.0 - decoder_mask_b11)
    elif self.self_cond_prob > 0:
      use_sc = (
        (torch.rand((bsz,), device=device, dtype=dtype) < self.self_cond_prob)
        .reshape(-1, 1, 1)
        .to(dtype=dtype)
      )
      with torch.no_grad():
        z_in0 = self._apply_self_cond(denoiser_z, None)
        pair0 = self._embed_continuous_pair(z_in0, x0_emb)
        x_init, _ = self._backbone(pair0, t, decode=False)
      sc_half = x_init.detach() * use_sc * (1.0 - decoder_mask_b11)

    z_in = self._apply_self_cond(z_mixed, sc_half)
    pair = self._embed_continuous_pair(z_in, x0_emb)
    x_pred, logits = self._backbone(
      pair,
      t_mixed,
      decode=decoder_step_active,
      self_cond_cfg_scale=self_cond_cfg,
    )

    v_pred = self._x_to_v(x_pred, denoiser_z, t)
    l2_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)

    assert logits is not None
    log_probs = F.log_softmax(logits.float(), dim=-1)
    ce_per_token = -log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    # 忽略 pad / ignore_index
    valid = (tokens != self.token_layout.ignore_index).to(dtype=ce_per_token.dtype)
    if self.token_layout.pad_token_id is not None:
      valid = valid * (tokens != self.token_layout.pad_token_id).to(dtype=valid.dtype)

    ce_mask = valid * decoder_mask_b1
    l2_mask = valid * (1.0 - decoder_mask_b1)
    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total_sum / torch.clamp(valid.sum(), min=1.0)

    ce_denom = ce_mask.sum()
    l2_denom = l2_mask.sum()
    # 本微批无 decode 样本时记 nan，避免 0/1→ce=0→假 ppl=1.0。
    self.last_ce_loss = torch.where(
      ce_denom > 0,
      (ce_per_token * ce_mask).sum() / ce_denom.clamp(min=1.0),
      torch.full((), float("nan"), device=ce_per_token.device, dtype=ce_per_token.dtype),
    ).detach()
    self.last_l2_loss = torch.where(
      l2_denom > 0,
      (l2_per_token * l2_mask).sum() / l2_denom.clamp(min=1.0),
      torch.full((), float("nan"), device=l2_per_token.device, dtype=l2_per_token.dtype),
    ).detach()
    return loss

  def _touch_unused_heads(self, loss: torch.Tensor) -> torch.Tensor:
    touch = (
      self.x_pred_head.weight.sum()
      + self.x_pred_head.bias.sum()
      + self.lm_head.weight.sum()
    )
    if self.mode_tokens is not None:
      touch = touch + self.mode_tokens.sum()
    if self.self_cond_cfg_tokens is not None:
      touch = (
        touch
        + self.self_cond_cfg_embedder.mlp[0].weight.sum()
        + self.self_cond_cfg_embedder.mlp[2].weight.sum()
        + self.self_cond_cfg_tokens.sum()
      )
    if self.self_cond_prob > 0:
      touch = touch + self.self_cond_proj.weight.sum() + self.self_cond_proj.bias.sum()
    return loss + 0.0 * touch

  def forward(
    self,
    idx: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    branch: Literal["denoise", "decode"] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del targets
    bsz, seq_len = idx.shape
    self._validate_seq_len(seq_len)
    x0_emb = self._tokens_to_emb(idx)

    if branch == "decode":
      loss = self._decode_loss(x0_emb, idx)
      self.last_loss_branch = "decode"
      self.last_ce_loss = loss.detach()
      self.last_l2_loss = float("nan")
      loss = self._touch_unused_heads(loss)
    elif branch == "denoise":
      loss = self._denoise_loss(x0_emb, idx)
      self.last_loss_branch = "denoise"
      self.last_l2_loss = loss.detach()
      self.last_ce_loss = float("nan")
      loss = self._touch_unused_heads(loss)
    else:
      loss = self._mixed_branch_loss(x0_emb, idx)
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
    *,
    sc_prev: torch.Tensor | None = None,
    self_cond_cfg_scale: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-block Euler ODE step；返回 (z_new, x_pred)。"""
    bsz, win_len, _ = z.shape
    z_in = self._apply_self_cond(z, sc_prev)
    pair = self._embed_continuous_pair(z_in, x0_ctx)
    t_batch = t.expand(bsz)
    x_pred, _ = self._backbone(
      pair, t_batch, decode=False,
      self_cond_cfg_scale=self_cond_cfg_scale,
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
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    self_cond_cfg_scale: torch.Tensor | None = None,
  ) -> torch.Tensor:
    bsz, win_len, _ = z.shape
    z_in = self._apply_self_cond(z, None)
    pair = self._embed_continuous_pair(z_in, x0_ctx)
    t_batch = torch.ones(bsz, device=z.device, dtype=z.dtype)
    _, logits = self._backbone(
      pair, t_batch, decode=True,
      self_cond_cfg_scale=self_cond_cfg_scale,
      window_len=win_len, window_start=window_start,
    )
    return sample_from_logits(
      logits[:, -self.diffusion_block_size:],
      temperature=temperature,
      top_k=top_k,
    )

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
    """Build full-length prefix z/x0 halves；window_start 恒为 0。"""
    n_samples = z_block.size(0)
    win_len = end_idx - start_idx
    db = self.diffusion_block_size
    ctx_len = emb_accum.size(1)
    z_win = torch.zeros(n_samples, win_len, self.n_embd, device=device, dtype=dtype)
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

  def _resolve_sc_cfg(
    self,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    sc_cfg_w: float,
  ) -> torch.Tensor | None:
    if self.num_self_cond_cfg_tokens > 0:
      return torch.full((n_samples,), sc_cfg_w, dtype=dtype, device=device)
    if sc_cfg_w != 1.0 and self.self_cond_prob > 0:
      return torch.full((n_samples,), sc_cfg_w, dtype=dtype, device=device)
    return None

  @torch.no_grad()
  def _semi_ar_flow_sampler(
    self,
    n_samples: int,
    seqlen: int,
    num_ode_steps: int,
    *,
    bos_token_id: int | None = None,
    use_fast_infer: bool = True,
    prefix_tokens: torch.Tensor | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    self_cond_cfg_scale: float = 3.0,
  ) -> tuple[torch.Tensor, int]:
    bos = self.token_layout.bos_token_id if bos_token_id is None else bos_token_id
    db = self.diffusion_block_size
    device = next(self.parameters()).device
    dtype = next(self.parameters()).dtype
    num_strides = seqlen // db
    nfe = 0
    t_steps = self._get_sampling_steps(num_ode_steps, device)
    sc_cfg = self._resolve_sc_cfg(n_samples, device, dtype, self_cond_cfg_scale)

    if not use_fast_infer:
      return self._semi_ar_flow_sampler_legacy(
        n_samples, seqlen, num_ode_steps,
        bos_token_id=bos_token_id,
        t_steps=t_steps,
        prefix_tokens=prefix_tokens,
        temperature=temperature,
        top_k=top_k,
        self_cond_cfg_scale=sc_cfg,
      )

    state = BDELFInferState(self, n_samples, seqlen, device, dtype)
    start_stride = 0
    if prefix_tokens is not None:
      prefix_len = prefix_tokens.size(1)
      if prefix_tokens.size(0) != n_samples:
        raise ValueError("prefix_tokens batch size must match n_samples")
      if prefix_len == 0 or prefix_len % db != 0 or prefix_len >= seqlen:
        raise ValueError(
          f"prefix length {prefix_len} must be a positive multiple of "
          f"diffusion_block_size ({db}) and less than seqlen ({seqlen})"
        )
      emb_accum = self._tokens_to_emb(prefix_tokens)
      state.set_emb_accum(emb_accum)
      state.tokens_buf[:, :prefix_len] = prefix_tokens
      state.token_len = prefix_len
      start_stride = prefix_len // db
      for stride in range(start_stride):
        block_emb = emb_accum[:, stride * db : (stride + 1) * db]
        state.on_stride_complete(block_emb, stride)
    else:
      emb_accum = torch.zeros(
        n_samples, 0, self.n_embd, device=device, dtype=dtype,
      )

    for stride in range(start_stride, num_strides):
      z_block = torch.randn(
        n_samples, db, self.n_embd, device=device, dtype=dtype,
      ) * self.denoiser_noise_scale

      if stride == 0 and bos is not None:
        bos_emb = self._tokens_to_emb(
          torch.full((n_samples, 1), bos, device=device, dtype=torch.long),
        )
        z_block[:, 0] = bos_emb[:, 0]

      state.begin_stride(stride, emb_accum)
      x_pred: torch.Tensor | None = None

      for i in range(len(t_steps) - 1):
        t = t_steps[i]
        t_next = t_steps[i + 1]
        z_block, x_pred = state.ode_step(
          z_block, stride, t, t_next,
          sc_prev=x_pred,
          self_cond_cfg_scale=sc_cfg,
        )
        nfe += 1

      block_tokens = state.decode_block(
        z_block, stride,
        temperature=temperature, top_k=top_k,
        self_cond_cfg_scale=sc_cfg,
      )
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
    prefix_tokens: torch.Tensor | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    self_cond_cfg_scale: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, int]:
    """未优化推理路径（数值对齐 / 调试）。"""
    bos = self.token_layout.bos_token_id if bos_token_id is None else bos_token_id
    db = self.diffusion_block_size
    device = next(self.parameters()).device
    dtype = next(self.parameters()).dtype
    num_strides = seqlen // db
    nfe = 0
    if t_steps is None:
      t_steps = self._get_sampling_steps(num_ode_steps, device)

    start_stride = 0
    if prefix_tokens is not None:
      prefix_len = prefix_tokens.size(1)
      if prefix_tokens.size(0) != n_samples:
        raise ValueError("prefix_tokens batch size must match n_samples")
      if prefix_len == 0 or prefix_len % db != 0 or prefix_len >= seqlen:
        raise ValueError(
          f"prefix length {prefix_len} must be a positive multiple of "
          f"diffusion_block_size ({db}) and less than seqlen ({seqlen})"
        )
      emb_accum = self._tokens_to_emb(prefix_tokens)
      tokens = prefix_tokens.clone()
      start_stride = prefix_len // db
    else:
      tokens = torch.zeros(n_samples, 0, dtype=torch.long, device=device)
      emb_accum = torch.zeros(
        n_samples, 0, self.n_embd, device=device, dtype=dtype,
      )

    for stride in range(start_stride, num_strides):
      z_block = torch.randn(
        n_samples, db, self.n_embd, device=device, dtype=dtype,
      ) * self.denoiser_noise_scale

      if stride == 0 and bos is not None:
        bos_emb = self._tokens_to_emb(
          torch.full((n_samples, 1), bos, device=device, dtype=torch.long),
        )
        z_block[:, 0] = bos_emb[:, 0]

      end_idx = (stride + 1) * db
      start_idx = 0
      x_pred: torch.Tensor | None = None

      for i in range(len(t_steps) - 1):
        t = t_steps[i]
        t_next = t_steps[i + 1]
        z_win, x0_win = self._build_window(
          emb_accum, z_block, start_idx, end_idx, device, dtype,
        )
        # self-cond 只作用在当前块；扩展到整窗时其余位置填 0
        sc_win = None
        if x_pred is not None and self.self_cond_prob > 0:
          sc_win = torch.zeros_like(z_win)
          cur_off = end_idx - db - start_idx
          sc_win[:, cur_off:cur_off + db] = x_pred
        z_new, x_pred_full = self._ode_step_block(
          z_win, x0_win, t, t_next, window_start=start_idx,
          sc_prev=sc_win,
          self_cond_cfg_scale=self_cond_cfg_scale,
        )
        cur_off = end_idx - db - start_idx
        z_block = z_new[:, cur_off:cur_off + db]
        x_pred = x_pred_full[:, cur_off:cur_off + db]
        nfe += 1

      z_win, x0_win = self._build_window(
        emb_accum, z_block, start_idx, end_idx, device, dtype,
      )
      block_tokens = self._decode_block(
        z_win, x0_win, window_start=start_idx,
        temperature=temperature, top_k=top_k,
        self_cond_cfg_scale=self_cond_cfg_scale,
      )
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
    temperature: float = 1.0,
    top_k: int | None = None,
    bos_token_id: int | None = None,
    prefix_tokens: torch.Tensor | None = None,
    sampling_cfg: dict | None = None,
  ) -> tuple[torch.Tensor, int]:
    cfg = sampling_cfg or {}
    if seqlen is None:
      raise ValueError("generate requires an explicit seqlen")
    num_ode_steps = num_steps if num_steps is not None else cfg.get("num_ode_steps", 8)
    temperature = float(cfg.get("temperature", temperature))
    top_k = cfg.get("top_k", top_k)
    if top_k is not None:
      top_k = int(top_k)
    bos = self.token_layout.bos_token_id
    if bos_token_id is not None:
      bos = bos_token_id
    infer_schedule = cfg.get("time_schedule")
    if infer_schedule is not None:
      self._infer_time_schedule = infer_schedule
    sc_cfg_w = float(cfg.get("self_cond_cfg_scale", 3.0))

    self._validate_seq_len(seqlen)
    use_fast_infer = cfg.get("use_fast_infer", True)

    return self._semi_ar_flow_sampler(
      n_samples=num_samples,
      seqlen=seqlen,
      num_ode_steps=num_ode_steps,
      bos_token_id=bos,
      use_fast_infer=use_fast_infer,
      prefix_tokens=prefix_tokens,
      temperature=temperature,
      top_k=top_k,
      self_cond_cfg_scale=sc_cfg_w,
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
