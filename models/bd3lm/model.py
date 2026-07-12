"""Block Diffusion Language Model (BD3LM) 最简实现。

参考论文: Block Diffusion: Interpolating Between Autoregressive and Diffusion
Language Models (https://arxiv.org/abs/2503.09573)
参考代码: bd3lms (kuleshov-group/bd3lms)

核心思路:
  1. 训练时将噪声序列 xt 与干净序列 x0 拼接为 [xt, x0]
  2. 使用块扩散注意力掩码（块对角 + 偏移块因果 + 块因果）
  3. SUBS 参数化预测被 mask 位置的干净 token
  4. Log-linear 噪声调度

接口约定: forward(x0, targets) -> (logits, loss)
  - full_sequence_training=True，训练/评估均使用完整序列（不做 AR shift）
"""

from __future__ import annotations

import math
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bd3lm.config import FL_BD3LMConfig
from models.model import FL_PreTrainedModel, ensure_token_layout, split_model_cfg
from models.rope import RotaryEmbedding, pair_positions, window_positions
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg

try:
  from torch.nn.attention.flex_attention import create_block_mask, flex_attention

  FLEX_ATTN_AVAILABLE = True
except ImportError:
  FLEX_ATTN_AVAILABLE = False


def block_diff_mask(
  b, h, q_idx, kv_idx, block_size: int | None = None, n: int | None = None,
) -> torch.Tensor:
  """FlexAttention 用块扩散掩码（与 bd3lms models/dit.py 一致）。"""
  del b, h
  x0_flag_q = q_idx >= n
  x0_flag_kv = kv_idx >= n

  block_q = torch.where(
    x0_flag_q,
    (q_idx - n) // block_size,
    q_idx // block_size,
  )
  block_kv = torch.where(
    x0_flag_kv,
    (kv_idx - n) // block_size,
    kv_idx // block_size,
  )

  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
  offset_block_causal = (
    (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
  )
  block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
  return block_diagonal | offset_block_causal | block_causal


if FLEX_ATTN_AVAILABLE:
  def fused_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask=None,
  ) -> torch.Tensor:
    return flex_attention(q, k, v, block_mask=block_mask)
else:
  fused_flex_attention = None  # type: ignore[assignment]


def bool_mask_to_sdpa_additive(bool_mask: torch.Tensor) -> torch.Tensor:
  """(L, L) bool -> (1, 1, L, L) float additive mask for SDPA."""
  additive = torch.zeros(
    1, 1, bool_mask.size(0), bool_mask.size(1),
    dtype=torch.float32, device=bool_mask.device,
  )
  additive.masked_fill_(~bool_mask.view(1, 1, bool_mask.size(0), bool_mask.size(1)), float("-inf"))
  return additive


def build_block_diff_mask(
  seq_len: int,
  diffusion_block_size: int,
  device: torch.device,
) -> torch.Tensor:
  """构造块扩散 bool 掩码（SDPA 回退路径），形状 (2*seq_len, 2*seq_len)。"""
  n = seq_len
  q_idx = torch.arange(n * 2, device=device)[:, None]
  kv_idx = torch.arange(n * 2, device=device)[None, :]
  return block_diff_mask(
    None, None, q_idx, kv_idx,
    block_size=diffusion_block_size, n=n,
  )


class LogLinearNoise(nn.Module):
  """Log-linear 噪声调度: move_chance = t, loss_scale = -1/t。"""

  def __init__(self, eps: float = 1e-3) -> None:
    super().__init__()
    self.eps = eps

  def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = t.clamp(min=self.eps)
    loss_scale = -1.0 / t
    move_chance = t
    return loss_scale, move_chance


class BlockDiffusionAttention(nn.Module):
  """带块扩散掩码的多头自注意力（FlexAttention，SDPA 回退）。"""

  def __init__(
    self,
    n_embd: int,
    n_head: int,
    dropout: float,
    attn_backend: str = "flex",
  ) -> None:
    super().__init__()
    if n_embd % n_head != 0:
      raise ValueError(f"n_embd ({n_embd}) 必须能被 n_head ({n_head}) 整除")
    self.n_head = n_head
    self.n_embd = n_embd
    self.head_dim = n_embd // n_head
    self.dropout = dropout
    self.attn_backend = attn_backend

    self.c_attn = nn.Linear(n_embd, 3 * n_embd)
    self.c_proj = nn.Linear(n_embd, n_embd)
    self.resid_dropout = nn.Dropout(dropout)
    self.rope = RotaryEmbedding(self.head_dim)

  def forward(
    self,
    x: torch.Tensor,
    flex_block_mask=None,
    sdpa_attn_mask: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    bsz, seq_len, _ = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)

    def reshape_heads(t: torch.Tensor) -> torch.Tensor:
      return t.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

    if positions is None:
      positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
    q, k = self.rope.apply_qk(q, k, positions)

    if self.attn_backend == "flex" and flex_block_mask is not None:
      y = fused_flex_attention(q, k, v, block_mask=flex_block_mask)
    elif sdpa_attn_mask is not None:
      if sdpa_attn_mask.dtype == torch.bool:
        attn_mask = bool_mask_to_sdpa_additive(sdpa_attn_mask)
      else:
        attn_mask = sdpa_attn_mask
      y = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
      )
    else:
      raise RuntimeError("BlockDiffusionAttention 需要 flex_block_mask 或 sdpa_attn_mask")

    y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
    return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
  def __init__(self, n_embd: int, dropout: float) -> None:
    super().__init__()
    hidden = 4 * n_embd
    self.c_fc = nn.Linear(n_embd, hidden)
    self.c_proj = nn.Linear(hidden, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.c_fc(x)
    x = F.gelu(x, approximate="tanh")
    x = self.c_proj(x)
    return self.dropout(x)


class Block(nn.Module):
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


class _BD3LMBackbone(nn.Module):
  """Block Diffusion Language Model backbone used for training."""

  full_sequence_training = True

  def __init__(
    self,
    token_layout: FL_TokenLayout,
    max_seq_len: int = 4096,
    diffusion_block_size: int = 32,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 672,
    dropout: float = 0.1,
    sampling_eps_min: float = 1e-3,
    sampling_eps_max: float = 1.0,
    attn_backend: str = "flex",
    fix_bos: bool = True,
  ) -> None:
    super().__init__()

    if attn_backend == "flex" and not FLEX_ATTN_AVAILABLE:
      raise RuntimeError(
        "attn_backend=flex 需要 PyTorch FlexAttention，请升级 PyTorch 或改用 attn_backend=sdpa"
      )
    if attn_backend not in ("flex", "sdpa"):
      raise ValueError(f"未知 attn_backend: {attn_backend}")

    self.max_seq_len = max_seq_len
    self.diffusion_block_size = diffusion_block_size
    self.attn_backend = attn_backend
    self.token_layout = token_layout
    self.vocab_size = token_layout.vocab_size
    self.mask_index = token_layout.mask_token_id
    self.model_vocab_size = token_layout.bd3lm_vocab_size

    self.sampling_eps_min = sampling_eps_min
    self.sampling_eps_max = sampling_eps_max
    self.fix_bos = fix_bos
    self.neg_infinity = -1e7

    self.wte = nn.Embedding(self.model_vocab_size, n_embd)
    self.drop = nn.Dropout(dropout)
    self.h = nn.ModuleList(
      Block(n_embd, n_head, dropout, attn_backend) for _ in range(n_layer)
    )
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, self.model_vocab_size, bias=False)
    self.lm_head.weight = self.wte.weight

    self.noise = LogLinearNoise()
    self.apply(self._init_weights)

    self._pair_sdpa_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
    self._flex_block_mask_cache: dict[tuple[int, torch.device], object] = {}
    self._sample_mask_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

  def _validate_seq_len(self, seq_len: int) -> None:
    if seq_len > self.max_seq_len:
      raise ValueError(
        f"序列长度 {seq_len} 超过 max_seq_len {self.max_seq_len}"
      )
    db = self.diffusion_block_size
    if seq_len % db != 0:
      raise ValueError(
        f"序列长度 {seq_len} 必须能被 diffusion_block_size ({db}) 整除"
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

  def _get_sample_causal_mask(
    self, window_start: int, window_len: int, device: torch.device,
  ) -> torch.Tensor:
    key = (window_start, window_len, device)
    cached = self._sample_mask_cache.get(key)
    if cached is None:
      cached = self._build_sample_block_causal_mask(window_start, window_len, device)
      if len(self._sample_mask_cache) >= 16:
        self._sample_mask_cache.pop(next(iter(self._sample_mask_cache)))
      self._sample_mask_cache[key] = cached
    return cached

  def _embed(self, idx: torch.Tensor) -> torch.Tensor:
    """对拼接序列 [xt, x0] 做 token embedding（RoPE 在 attention 内应用）。"""
    return self.drop(self.wte(idx))

  def _backbone(self, x_input: torch.Tensor) -> torch.Tensor:
    """输入拼接序列，输出 xt 部分的 logits。"""
    n = x_input.size(1) // 2
    x = self._embed(x_input)
    positions = pair_positions(n, x.device)
    flex_mask = None
    sdpa_mask = None
    if self.attn_backend == "flex":
      flex_mask = self._get_flex_block_mask(n, x.device)
    else:
      sdpa_mask = self._get_pair_sdpa_mask(n, x.device)
    for block in self.h:
      x = block(x, flex_mask, sdpa_mask, positions)
    x = self.ln_f(x)
    logits = self.lm_head(x[:, :n])
    return logits

  def _subs_parameterization(
    self, logits: torch.Tensor, xt: torch.Tensor,
  ) -> torch.Tensor:
    """SUBS 参数化：被 mask 位置预测 x0，未 mask 位置 log prob 为 0。"""
    logits = logits.clone()
    logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    unmasked = xt != self.mask_index
    logits[unmasked] = self.neg_infinity
    logits[unmasked, xt[unmasked]] = 0.0
    return logits

  def _sample_t(
    self, batch_size: int, seq_len: int, device: torch.device,
  ) -> torch.Tensor:
    """按块均匀采样噪声时间 t ∈ [sampling_eps_min, sampling_eps_max]。"""
    num_blocks = seq_len // self.diffusion_block_size
    t = torch.rand(batch_size, num_blocks, device=device)
    t = t * (self.sampling_eps_max - self.sampling_eps_min) + self.sampling_eps_min
    return t.repeat_interleave(self.diffusion_block_size, dim=-1)

  def _q_xt(self, x0: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """前向扩散：以概率 p 将 token 替换为 mask。"""
    move = torch.rand_like(x0, dtype=torch.float32) <= p
    return torch.where(move, self.mask_index, x0)

  def _diffusion_loss(self, x0: torch.Tensor) -> torch.Tensor:
    """计算扩散训练损失（token 级 NLL 均值）。"""
    bsz, seq_len = x0.shape
    self._validate_seq_len(seq_len)

    t = self._sample_t(bsz, seq_len, x0.device)
    loss_scale, move_chance = self.noise(t)
    p = move_chance

    xt = self._q_xt(x0, p)
    if self.fix_bos:
      xt[:, 0] = x0[:, 0]
    x_input = torch.cat([xt, x0], dim=1)

    logits = self._backbone(x_input)

    # fp32：50258 维 logsumexp 与 loss 加权在 bf16 下不够稳定
    with torch.amp.autocast("cuda", enabled=False):
      log_score = self._subs_parameterization(logits.float(), xt)
      log_p = torch.gather(
        log_score, dim=-1, index=x0.unsqueeze(-1),
      ).squeeze(-1)
      loss = loss_scale.float() * log_p
      return loss.mean()

  def forward(
    self,
    idx: torch.Tensor,
    targets: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """训练/评估接口。

    Args:
      idx: 完整 token 序列 (B, L)，L 可变且须能被 diffusion_block_size 整除
      targets: 忽略（兼容 train.py 接口）
    """
    del targets
    loss = self._diffusion_loss(idx)
    return torch.empty(0), loss

  # -------------------------------------------------------------------------
  # 推理 / 采样（移植自 bd3lms diffusion/base.py）
  # -------------------------------------------------------------------------

  def _sample_prior(self, *batch_dims: int) -> torch.Tensor:
    return self.mask_index * torch.ones(
      *batch_dims, dtype=torch.long, device=next(self.parameters()).device,
    )

  def _embed_sample(self, idx: torch.Tensor, window_start: int) -> torch.Tensor:
    """采样模式 embedding（RoPE 在 attention 内应用）。"""
    del window_start
    return self.drop(self.wte(idx))

  def _build_sample_block_causal_mask(
    self, window_start: int, window_len: int, device: torch.device,
  ) -> torch.Tensor:
    """采样用块因果掩码（官方 sample_mode 取 full mask 的 x0|x0 子块）。"""
    pos = torch.arange(
      window_start, window_start + window_len, device=device,
    )
    block_q = pos[:, None] // self.diffusion_block_size
    block_kv = pos[None, :] // self.diffusion_block_size
    return block_q >= block_kv

  def _backbone_sample(
    self, xt_window: torch.Tensor, window_start: int = 0,
  ) -> torch.Tensor:
    """采样前向：仅 xt 窗口 + 块因果注意力（对齐官方 sample_mode）。"""
    x = self._embed_sample(xt_window, window_start)
    window_len = xt_window.size(1)
    positions = window_positions(window_start, window_len, xt_window.device)
    sdpa_mask = bool_mask_to_sdpa_additive(
      self._get_sample_causal_mask(window_start, window_len, xt_window.device),
    )
    for block in self.h:
      x = block(x, None, sdpa_mask, positions)
    x = self.ln_f(x)
    return self.lm_head(x)

  @torch.no_grad()
  def _predict_log_score(
    self, xt_window: torch.Tensor, window_start: int = 0,
  ) -> torch.Tensor:
    logits = self._backbone_sample(xt_window, window_start)
    with torch.amp.autocast("cuda", enabled=False):
      return self._subs_parameterization(logits.float(), xt_window)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0: torch.Tensor, nucleus_p: float) -> torch.Tensor:
    if nucleus_p >= 1.0:
      return p_x0
    db = self.diffusion_block_size
    p_x0_ = p_x0[:, -db:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= nucleus_p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -db:] = p_x0_
    return p_x0

  @torch.no_grad()
  def _ddpm_caching_update(
    self,
    x: torch.Tensor,
    window_start: int,
    t: torch.Tensor,
    dt: float,
    p_x0_cache: torch.Tensor | None,
    *,
    first_hitting: bool,
    nucleus_p: float,
  ) -> tuple[torch.Tensor | None, torch.Tensor]:
    """单步 DDPM 去噪更新（bd3lms diffusion/base.py）。"""
    db = self.diffusion_block_size
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise(t - dt)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    if p_x0_cache is None:
      log_score = self._predict_log_score(x, window_start=window_start)
      p_x0 = log_score[:, -db:].exp().to(torch.float64)
      p_x0 = self._nucleus_sample(p_x0, nucleus_p)
    else:
      p_x0 = p_x0_cache

    x_tail = x[:, -db:]
    if first_hitting:
      x_block = _sample_categorical(p_x0)
      for b in range(x_block.shape[0]):
        mask_positions = (x_tail[b] == self.mask_index).nonzero(as_tuple=True)[0]
        if mask_positions.numel() == 0:
          continue
        pick = torch.randint(0, mask_positions.numel(), (1,), device=x.device)
        col = mask_positions[pick.squeeze()]
        keep = torch.arange(db, device=x.device) == col
        x_block[b] = torch.where(keep, x_block[b], x_tail[b])
    else:
      q_xs = p_x0 * (1 - mask_prob.unsqueeze(-1))
      q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
      x_block = _sample_categorical(q_xs)

    unmasked = (x_tail != self.mask_index).to(x.dtype)
    x_block = unmasked * x_tail + (1 - unmasked) * x_block
    x_new = torch.cat((x[:, :-db], x_block), dim=-1)

    if not torch.allclose(x_new, x):
      return None, x_new
    return p_x0, x_new

  @torch.no_grad()
  def _semi_ar_sampler(
    self,
    n_samples: int,
    num_steps: int,
    seqlen: int,
    *,
    first_hitting: bool = True,
    nucleus_p: float = 1.0,
    bos_token_id: int | None = None,
  ) -> tuple[torch.Tensor, int]:
    """Semi-AR 块扩散采样；始终使用从位置 0 起的全长上下文。"""
    bos = self.token_layout.bos_token_id if bos_token_id is None else bos_token_id
    db = self.diffusion_block_size
    num_strides = seqlen // db
    device = next(self.parameters()).device
    sampling_steps = 0
    ones = torch.ones((n_samples, 1), device=device)

    for stride_num in range(num_strides):
      if stride_num == 0:
        x_accum = self._sample_prior(n_samples, db).to(device)
        if bos is not None:
          x_accum[:, 0] = bos
      else:
        x_new = self._sample_prior(n_samples, db).to(device)
        x_accum = torch.cat((x_accum, x_new), dim=1)

      end_idx = (stride_num + 1) * db
      x_window = x_accum[:, :end_idx]

      dt = 1.0 / num_steps
      p_x0_cache = None
      timesteps = torch.linspace(1, 0, num_steps, device=device)
      t = torch.ones((), device=device)
      for step_i in range(num_steps):
        if not (x_accum == self.mask_index).any():
          break
        if not (x_window == self.mask_index).any():
          break

        if first_hitting:
          num_masked = (x_window == self.mask_index).sum()
          u = torch.rand((), device=device)
          t = t * u.pow(1.0 / num_masked.float())
          t_tensor = t * ones
        else:
          t_tensor = timesteps[step_i] * ones

        p_x0_cache, x_next = self._ddpm_caching_update(
          x_window,
          window_start=0,
          t=t_tensor,
          dt=dt,
          p_x0_cache=p_x0_cache,
          first_hitting=first_hitting,
          nucleus_p=nucleus_p,
        )
        if p_x0_cache is None:
          sampling_steps += 1
        x_accum[:, :end_idx] = x_next

    return x_accum, sampling_steps

  @torch.no_grad()
  def generate(
    self,
    num_samples: int = 1,
    seqlen: int | None = None,
    num_steps: int | None = None,
    *,
    sampler: Literal["semi_ar"] = "semi_ar",
    first_hitting: bool | None = None,
    nucleus_p: float | None = None,
    bos_token_id: int | None = None,
    sampling_cfg: dict | None = None,
  ) -> tuple[torch.Tensor, int]:
    """从噪声生成 token 序列。

    Returns:
      tokens: (num_samples, seqlen)
      nfe: 实际模型前向次数（官方 metrics.nfes）
    """
    cfg = sampling_cfg or {}
    if seqlen is None:
      raise ValueError("generate 需要显式指定 seqlen")
    num_steps = num_steps if num_steps is not None else cfg.get("num_steps", 5000)
    first_hitting = (
      first_hitting if first_hitting is not None
      else cfg.get("first_hitting", True)
    )
    nucleus_p = nucleus_p if nucleus_p is not None else cfg.get("nucleus_p", 1.0)
    bos = self.token_layout.bos_token_id
    if bos_token_id is not None:
      bos = bos_token_id

    self._validate_seq_len(seqlen)
    if sampler != "semi_ar":
      raise NotImplementedError(f"采样器 {sampler!r} 尚未移植")

    return self._semi_ar_sampler(
      n_samples=num_samples,
      num_steps=num_steps,
      seqlen=seqlen,
      first_hitting=first_hitting,
      nucleus_p=nucleus_p,
      bos_token_id=bos,
    )


def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
  """Gumbel-max 采样（bd3lms diffusion/base.py）。"""
  gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
  return (categorical_probs / gumbel_norm).argmax(dim=-1)

class FL_BD3LMModel(FL_PreTrainedModel):
  config_class = FL_BD3LMConfig

  def __init__(self, config: FL_BD3LMConfig) -> None:
    super().__init__(config)
    self.backbone = _BD3LMBackbone(**config.backbone_kwargs())


def build_model_from_config(config: FL_BD3LMConfig) -> FL_BD3LMModel:
  ensure_token_layout(config)
  return FL_BD3LMModel(config)


def build_model(cfg: dict) -> FL_BD3LMModel:
  data, sampling = split_model_cfg(cfg)
  layout = token_layout_from_cfg(data)
  data.pop("tokenizer", None)
  config = FL_BD3LMConfig(**data)
  apply_token_layout_to_config(config, layout)
  if sampling is not None:
    config.sampling = sampling
  return build_model_from_config(config)
