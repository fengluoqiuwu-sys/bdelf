"""readout=none 专用：对齐 HuggingFace 原版 t5-small 算子（非 t5.1.1 gated-GELU）。

对照 ``T5LayerNorm`` / ``T5Attention``（相对位置偏置、无 1/sqrt(d_kv) 缩放）/
``T5DenseActDense``（ReLU、wi/wo、无 gate）。不用于 readout=e|b（仍走 encdec RoPE 栈）。

有意偏差（用户拍板）：decoder self-attn 与其 relative bias 均为**双向**（原版 T5
decoder 为单向 bias）；词表仍为 GPT-2；sentinel 只扩 encoder embed、不进 lm_head。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 与 HF T5Config 默认（t5-small）一致。
T5_LN_EPS = 1e-6
T5_RELATIVE_BUCKETS = 32
T5_RELATIVE_MAX_DISTANCE = 128


def _key_padding_additive(
    pad: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``(B, S)`` True=pad → ``(B, 1, 1, S)`` 加性 mask，pad 键为 ``-inf``。"""
    return torch.zeros(
        pad.size(0), 1, 1, pad.size(1),
        dtype=dtype or torch.float32, device=pad.device,
    ).masked_fill(pad.unsqueeze(1).unsqueeze(1), float("-inf"))


class T5LayerNorm(nn.Module):
    """T5 RMSNorm：无均值、无 bias，只缩放；eps=1e-6。"""

    def __init__(self, hidden_size: int, eps: float = T5_LN_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def relative_position_bucket(
    relative_position: torch.Tensor,
    *,
    bidirectional: bool,
    num_buckets: int = T5_RELATIVE_BUCKETS,
    max_distance: int = T5_RELATIVE_MAX_DISTANCE,
) -> torch.Tensor:
    """HF T5Attention._relative_position_bucket（Mesh TensorFlow 桶）。"""
    relative_buckets = torch.zeros_like(relative_position)
    n_buckets = num_buckets
    pos = relative_position
    if bidirectional:
        n_buckets //= 2
        relative_buckets = relative_buckets + (pos > 0).to(torch.long) * n_buckets
        pos = torch.abs(pos)
    else:
        pos = -torch.min(pos, torch.zeros_like(pos))

    max_exact = n_buckets // 2
    is_small = pos < max_exact
    pos_if_large = max_exact + (
        torch.log(pos.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (n_buckets - max_exact)
    ).to(torch.long)
    pos_if_large = torch.min(
        pos_if_large, torch.full_like(pos_if_large, n_buckets - 1),
    )
    relative_buckets = relative_buckets + torch.where(is_small, pos, pos_if_large)
    return relative_buckets


class T5Attention(nn.Module):
    """独立 q/k/v/o、无 RoPE；self-attn 可带相对位置偏置。注意力分数不除 sqrt(d_kv)。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        dropout: float,
        *,
        has_relative_attention_bias: bool = False,
        bias_bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_kv = d_kv
        self.inner_dim = n_head * d_kv
        self.dropout = dropout
        self.has_relative_attention_bias = has_relative_attention_bias
        self.bias_bidirectional = bias_bidirectional
        self.relative_attention_num_buckets = T5_RELATIVE_BUCKETS
        self.relative_attention_max_distance = T5_RELATIVE_MAX_DISTANCE
        # 与 HF T5 相同：投影无 bias。
        self.q = nn.Linear(n_embd, self.inner_dim, bias=False)
        self.k = nn.Linear(n_embd, self.inner_dim, bias=False)
        self.v = nn.Linear(n_embd, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, n_embd, bias=False)
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, n_head,
            )
        self._bucket_cache: dict[tuple, torch.Tensor] = {}

    @torch.compiler.disable
    def _relative_buckets(
        self,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """相对位置桶（与长度有关、与权重无关）；eager 缓存以免每步重建 L×L。"""
        key = (query_length, key_length, device)
        cached = self._bucket_cache.get(key)
        if cached is None:
            context_position = torch.arange(
                query_length, dtype=torch.long, device=device,
            )[:, None]
            memory_position = torch.arange(
                key_length, dtype=torch.long, device=device,
            )[None, :]
            cached = relative_position_bucket(
                memory_position - context_position,
                bidirectional=self.bias_bidirectional,
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            if len(self._bucket_cache) > 8:
                self._bucket_cache.pop(next(iter(self._bucket_cache)))
            self._bucket_cache[key] = cached
        return cached

    def compute_bias(
        self,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        buckets = self._relative_buckets(query_length, key_length, device)
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_value_states: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        kv_in = hidden_states if key_value_states is None else key_value_states
        kv_len = kv_in.size(1)

        def reshape_heads(t: torch.Tensor, length: int) -> torch.Tensor:
            return t.view(bsz, length, self.n_head, self.d_kv).transpose(1, 2)

        q = reshape_heads(self.q(hidden_states), q_len)
        k = reshape_heads(self.k(kv_in), kv_len)
        v = reshape_heads(self.v(kv_in), kv_len)

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    q_len, kv_len, device=hidden_states.device,
                ).to(dtype=q.dtype)
            # 无相对偏置时不要物化全零 (1,H,L,L)，否则 SDPA 走不出 Flash。
            if key_padding_mask is not None:
                pad_add = _key_padding_additive(
                    key_padding_mask, dtype=q.dtype,
                )
                position_bias = (
                    pad_add if position_bias is None else position_bias + pad_add
                )

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=position_bias,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, q_len, self.inner_dim)
        return self.o(y), position_bias


class T5DenseReluDense(nn.Module):
    """原版 T5 FFN：ReLU、wi/wo、无 gate、无 bias。"""

    def __init__(self, n_embd: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.wi = nn.Linear(n_embd, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.relu(self.wi(hidden_states))
        hidden_states = self.dropout(hidden_states)
        return self.wo(hidden_states)


class T5SelfAttnLayer(nn.Module):
    """Pre-norm self-attn + 残差 dropout。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        dropout: float,
        *,
        has_relative_attention_bias: bool,
        bias_bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.layer_norm = T5LayerNorm(n_embd)
        self.attn = T5Attention(
            n_embd, n_head, d_kv, dropout,
            has_relative_attention_bias=has_relative_attention_bias,
            bias_bidirectional=bias_bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, position_bias = self.attn(
            self.layer_norm(hidden_states),
            key_padding_mask=key_padding_mask,
            position_bias=position_bias,
        )
        return hidden_states + self.dropout(y), position_bias


class T5CrossAttnLayer(nn.Module):
    """Pre-norm cross-attn（无相对偏置）+ 残差 dropout。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer_norm = T5LayerNorm(n_embd)
        self.attn = T5Attention(
            n_embd, n_head, d_kv, dropout,
            has_relative_attention_bias=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, position_bias = self.attn(
            self.layer_norm(hidden_states),
            key_value_states=memory,
            key_padding_mask=key_padding_mask,
            position_bias=position_bias,
        )
        return hidden_states + self.dropout(y), position_bias


class T5FFLayer(nn.Module):
    """Pre-norm DenseReluDense + 残差 dropout。"""

    def __init__(self, n_embd: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm = T5LayerNorm(n_embd)
        self.dense = T5DenseReluDense(n_embd, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.dropout(self.dense(self.layer_norm(hidden_states)))


class T5EncoderBlock(nn.Module):
    """self-attn → FFN（T5 encoder 层）。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        d_ff: int,
        dropout: float,
        *,
        has_relative_attention_bias: bool,
    ) -> None:
        super().__init__()
        self.self_attn = T5SelfAttnLayer(
            n_embd, n_head, d_kv, dropout,
            has_relative_attention_bias=has_relative_attention_bias,
            bias_bidirectional=True,
        )
        self.ff = T5FFLayer(n_embd, d_ff, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, position_bias = self.self_attn(
            hidden_states,
            key_padding_mask=key_padding_mask,
            position_bias=position_bias,
        )
        return self.ff(hidden_states), position_bias


class T5DecoderBlock(nn.Module):
    """self-attn → cross-attn → FFN。self-attn relative bias 写死双向。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        d_kv: int,
        d_ff: int,
        dropout: float,
        *,
        has_relative_attention_bias: bool,
    ) -> None:
        super().__init__()
        self.self_attn = T5SelfAttnLayer(
            n_embd, n_head, d_kv, dropout,
            has_relative_attention_bias=has_relative_attention_bias,
            bias_bidirectional=True,
        )
        self.cross_attn = T5CrossAttnLayer(n_embd, n_head, d_kv, dropout)
        self.ff = T5FFLayer(n_embd, d_ff, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        memory_pad_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        encoder_decoder_position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, position_bias = self.self_attn(
            hidden_states,
            key_padding_mask=key_padding_mask,
            position_bias=position_bias,
        )
        hidden_states, encoder_decoder_position_bias = self.cross_attn(
            hidden_states,
            memory,
            key_padding_mask=memory_pad_mask,
            position_bias=encoder_decoder_position_bias,
        )
        hidden_states = self.ff(hidden_states)
        return hidden_states, position_bias, encoder_decoder_position_bias


class T5StyleEncoder(nn.Module):
    """T5 encoder 栈：token embed（含 sentinel 扩展行）+ 层 + 最终 RMSNorm。"""

    def __init__(
        self,
        vocab_size: int,
        extra_vocab: int,
        *,
        n_embd: int,
        n_head: int,
        d_kv: int,
        d_ff: int,
        n_layer: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.sentinel_embed: nn.Embedding | None = (
            nn.Embedding(extra_vocab, n_embd) if extra_vocab > 0 else None
        )
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5EncoderBlock(
                n_embd, n_head, d_kv, d_ff, dropout,
                has_relative_attention_bias=(i == 0),
            )
            for i in range(n_layer)
        ])
        self.final_layer_norm = T5LayerNorm(n_embd)
        self.final_dropout = nn.Dropout(dropout)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.sentinel_embed is None:
            return self.drop(self.wte(tokens))
        base_mask = tokens < self.vocab_size
        base_ids = tokens.clamp(max=self.vocab_size - 1)
        x = self.wte(base_ids)
        sent_ids = (tokens - self.vocab_size).clamp(
            min=0, max=self.sentinel_embed.num_embeddings - 1,
        )
        x = torch.where(base_mask.unsqueeze(-1), x, self.sentinel_embed(sent_ids))
        return self.drop(x)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(tokens)
        position_bias: torch.Tensor | None = None
        for block in self.blocks:
            x, position_bias = block(
                x, key_padding_mask=key_padding_mask, position_bias=position_bias,
            )
        return self.final_dropout(self.final_layer_norm(x))
