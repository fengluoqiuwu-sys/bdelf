"""Block Diffusion + Embedded Language Flow (BDELF).

在 BD3LM 块扩散注意力框架下，将块内 MDLM 离散掩码替换为 ELF 式连续 Flow Matching：
  - 训练：80% 去噪分支（MSE，[z_t, x0] 拼接 + 块扩散掩码）
  - 训练：20% 解码分支（CE，t=1 腐蚀 embedding → token）
  - 推理：semi-AR 逐块 ODE 积分 + 最终 decode

参考:
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
from models.model import FL_PreTrainedModel, ensure_token_layout, split_model_cfg
from models.rope import pair_positions, window_positions
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg

try:
  from torch.nn.attention.flex_attention import create_block_mask

  _FLEX_IMPORT_OK = True
except ImportError:
  _FLEX_IMPORT_OK = False


def build_window_pair_mask(
  window_start: int,
  window_len: int,
  diffusion_block_size: int,
  device: torch.device,
) -> torch.Tensor:
  """滑窗版块扩散掩码，形状 (2*window_len, 2*window_len)。"""
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


class TimestepEmbedder(nn.Module):
  """正弦时间步嵌入（DiT / ELF 风格）。"""

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
  """带时间条件化的 Transformer 块。"""

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
        "attn_backend=flex 需要 PyTorch FlexAttention，请升级或改用 sdpa"
      )
    if attn_backend not in ("flex", "sdpa"):
      raise ValueError(f"未知 attn_backend: {attn_backend}")

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
    """拼接 [z_t, x0]（RoPE 在 attention 内应用）。"""
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
    """块扩散 backbone。

    Args:
      pair_emb: (B, 2*L, D) 训练时 L 随 batch 变化；推理时 L 为当前全长前缀
      t: (B,) 时间步
      decode: 是否解码模式（输出 logits）
      window_len: 推理时单半长度；None 表示训练
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
    """训练/评估前向。

    Args:
      branch: ``None`` 时按 ``decoder_prob`` 随机选分支；``"denoise"`` / ``"decode"``
        用于 eval 或调试时固定分支。
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

  @torch.no_grad()
  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  # -------------------------------------------------------------------------
  # 推理：semi-AR 块内 ODE + decode
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
    """单块 Euler ODE 步。"""
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
    """构造全长前缀 z/x0 两半（各 win_len），window_start 恒为 0。"""
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
  ) -> tuple[torch.Tensor, int]:
    bos = self.token_layout.bos_token_id if bos_token_id is None else bos_token_id
    db = self.diffusion_block_size
    device = next(self.parameters()).device
    dtype = next(self.parameters()).dtype
    num_strides = seqlen // db
    nfe = 0
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
      raise ValueError("generate 需要显式指定 seqlen")
    num_ode_steps = num_steps if num_steps is not None else cfg.get("num_ode_steps", 8)
    bos = self.token_layout.bos_token_id
    if bos_token_id is not None:
      bos = bos_token_id
    infer_schedule = cfg.get("time_schedule")
    if infer_schedule is not None:
      self._infer_time_schedule = infer_schedule

    self._validate_seq_len(seqlen)

    return self._semi_ar_flow_sampler(
      n_samples=num_samples,
      seqlen=seqlen,
      num_ode_steps=num_ode_steps,
      bos_token_id=bos,
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
