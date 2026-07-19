"""AR1.5: AR2-like anchors without long-range s KV / t window (block-causal).

Sequence layout matches AR2 (single anchored clean stream, intra-block causal
AR). Differences:

Attention visibility:
  (a) only the *current* block's anchors are visible (no historical s);
  (b) any query sees *all* previous t (no window);
  (c) own-block t causal (same as AR2).

Prediction targets are identical to AR2 (exact NLL): the last anchor of each
block predicts the block's first token (skipped for block 0 under fix_bos),
t at offset j < B-1 predicts offset j+1.

Inference: anchor step samples the block's first token; after each block the
s KV is discarded and the block's t KV is appended to an unbounded cache_t.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ar1_5.config import FL_AR15Config
from models.model import FL_PreTrainedModel, ensure_token_layout, split_model_cfg
from models.rope import RotaryEmbedding
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False

ROLE_S = 0
ROLE_T = 1

_flex_attention_compiled = None


def fused_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask=None,
) -> torch.Tensor:
    """BD3LM-style entry: outer ``torch.compile(model)`` fuses this under Dynamo.

    Eager (fast / no whole-model compile): wrap once with ``torch.compile`` so
    FlexAttention does **not** fall back to the unfused path that materializes
    the full ``L×L`` score matrix.
    Never pass a ``score_mod`` that indexes data tensors by ``q_idx``/``kv_idx``
    — that commonly breaks fusion and triggers the same OOM.
    """
    if torch.compiler.is_dynamo_compiling():
        return flex_attention(q, k, v, block_mask=block_mask)
    global _flex_attention_compiled
    if _flex_attention_compiled is None:
        _flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_attention_compiled(q, k, v, block_mask=block_mask)


def make_ar15_mask_mod(
    *,
    block_size: int,
    num_anchors: int,
):
    """Visibility predicate for AR1.5 over the single [s.. t..] * K layout."""
    bs = block_size
    ns = num_anchors

    def mask_mod(b, h, q_idx, kv_idx):
        del b, h
        q_off = q_idx % (bs + ns)
        q_blk = q_idx // (bs + ns)
        q_is_s = q_off < ns

        kv_off = kv_idx % (bs + ns)
        kv_blk = kv_idx // (bs + ns)
        kv_is_s = kv_off < ns

        # (a) current-block anchors only (historical s never visible)
        see_s = kv_is_s & (kv_blk == q_blk)
        # (b) all previous t (no window)
        see_prev_t = ~kv_is_s & (kv_blk <= q_blk - 1)
        # (c) own-block t, causal, t queries only (anchors must not leak)
        same_blk_causal = (
            ~q_is_s & ~kv_is_s & (kv_blk == q_blk) & (kv_off <= q_off)
        )
        return see_s | see_prev_t | same_blk_causal

    return mask_mod


class Ar2Attention(nn.Module):
    """Self-attention with FlexAttention (training) and KV-cache SDPA (inference)."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        attn_type_bias: bool,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)
        # Fusion-safe role signal: add per-role vectors to Q/K (not a score_mod
        # table lookup, which breaks FlexAttention fusion and OOMs).
        self.role_q = nn.Embedding(2, n_embd) if attn_type_bias else None
        self.role_k = nn.Embedding(2, n_embd) if attn_type_bias else None
        if self.role_q is not None:
            nn.init.zeros_(self.role_q.weight)
            nn.init.zeros_(self.role_k.weight)

    def _project_qkv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rho: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        if self.role_q is not None and rho is not None:
            q = q + self.role_q(rho)
            k = k + self.role_k(rho)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        q, k = self.rope.apply_qk(q, k, positions)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        block_mask,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Training path: fused FlexAttention (no score_mod)."""
        q, k, v = self._project_qkv(x, positions, rho)
        y = fused_flex_attention(q, k, v, block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.n_embd)
        return self.resid_dropout(self.c_proj(y))

    def forward_infer(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        ctx_k: torch.Tensor | None,
        ctx_v: torch.Tensor | None,
        rho: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference: queries attend over [context KV; own KV], all visible.

        Returns (output, self_k, self_v); self K/V already carry RoPE + role.
        """
        q, k_self, v_self = self._project_qkv(x, positions, rho)
        if ctx_k is None:
            k, v = k_self, v_self
        else:
            k = torch.cat([ctx_k, k_self], dim=2)
            v = torch.cat([ctx_v, v_self], dim=2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.n_embd)
        return self.c_proj(y), k_self, v_self


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
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        attn_type_bias: bool,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = Ar2Attention(n_embd, n_head, dropout, attn_type_bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        block_mask,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), positions, block_mask, rho)
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_infer(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        ctx_k: torch.Tensor | None,
        ctx_v: torch.Tensor | None,
        rho: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, k_self, v_self = self.attn.forward_infer(
            self.ln_1(x), positions, ctx_k, ctx_v, rho,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, k_self, v_self


class _AR15Backbone(nn.Module):
    """AR1.5 backbone: current-block anchors + unbounded previous-t context."""

    full_sequence_training = True

    def __init__(
        self,
        token_layout: FL_TokenLayout,
        max_seq_len: int = 8192,
        block_size: int = 16,
        num_anchors: int = 1,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 672,
        dropout: float = 0.1,
        attn_backend: str = "flex",
        attn_type_bias: bool = True,
        fix_bos: bool = True,
    ) -> None:
        super().__init__()
        if attn_backend != "flex":
            raise ValueError("AR1.5 only implements attn_backend=flex")
        if not FLEX_ATTN_AVAILABLE:
            raise RuntimeError("AR1.5 requires PyTorch FlexAttention (torch >= 2.5)")
        if block_size < 2:
            raise ValueError("block_size must be >= 2")

        self.token_layout = token_layout
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.n_head = n_head
        self.fix_bos = fix_bos

        # Vocab layout: [tokenizer vocab | <s_0>..<s_{ns-1}>]
        self.vocab_size = token_layout.vocab_size
        self.anchor_index0 = token_layout.vocab_size
        self.model_vocab_size = token_layout.vocab_size + num_anchors

        self.wte = nn.Embedding(self.model_vocab_size, n_embd)
        self.type_emb = nn.Embedding(2, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList(
            Block(n_embd, n_head, dropout, attn_type_bias) for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.model_vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        nn.init.zeros_(self.type_emb.weight)

        self._block_mask_cache: dict[tuple[int, torch.device], object] = {}
        self._layout_cache: dict[tuple[int, torch.device], dict[str, torch.Tensor]] = {}

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -------------------------------------------------------------------------
    # Training layout helpers
    # -------------------------------------------------------------------------

    def _validate_seq_len(self, seq_len: int) -> None:
        if seq_len % self.block_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by block_size "
                f"({self.block_size})"
            )
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

    def _get_layout(self, n: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Static per-(seq_len) tensors: ids template, roles, positions, loss map."""
        key = (n, device)
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached

        bs, ns = self.block_size, self.num_anchors
        k = n // bs
        idx = torch.arange(k * (bs + ns), device=device)
        off = idx % (bs + ns)
        blk = idx // (bs + ns)
        is_anchor = off < ns
        # anchors take the block-start original index; t tokens their own index
        positions = torch.where(is_anchor, blk * bs, blk * bs + off - ns).long()
        # gather index from x0 for t positions (dummy 0 at anchors)
        gather_t = (blk * bs + (off - ns).clamp(min=0)).long()
        anchor_ids = self.anchor_index0 + torch.where(
            is_anchor, off, torch.zeros_like(off),
        )
        rho = torch.where(
            is_anchor,
            torch.full_like(off, ROLE_S),
            torch.full_like(off, ROLE_T),
        ).long()

        # Loss positions:
        #   last anchor slot of block k -> target x0[k*bs] (skip k=0 if fix_bos);
        #   t offset j in [0, bs-2]     -> target x0[k*bs + j + 1].
        is_last_anchor = off == (ns - 1)
        is_pred_t = (~is_anchor) & (off - ns < bs - 1)
        if self.fix_bos:
            is_last_anchor = is_last_anchor & (blk > 0)
        loss_mask = is_last_anchor | is_pred_t
        loss_pos = torch.nonzero(loss_mask, as_tuple=False).squeeze(-1)
        tgt_gather = torch.where(
            is_anchor, blk * bs, blk * bs + off - ns + 1,
        )[loss_pos].long()

        cached = {
            "is_anchor": is_anchor,
            "gather_t": gather_t,
            "anchor_ids": anchor_ids.long(),
            "rho": rho,
            "positions": positions,
            "loss_pos": loss_pos.long(),
            "tgt_gather": tgt_gather,
        }
        if len(self._layout_cache) >= 8:
            self._layout_cache.pop(next(iter(self._layout_cache)))
        self._layout_cache[key] = cached
        return cached

    def _get_block_mask(self, n: int, device: torch.device):
        key = (n, device)
        cached = self._block_mask_cache.get(key)
        if cached is None:
            bs, ns = self.block_size, self.num_anchors
            total = (n // bs) * (bs + ns)
            mask_mod = make_ar15_mask_mod(
                block_size=bs,
                num_anchors=ns,
            )
            cached = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device,
            )
            if len(self._block_mask_cache) >= 8:
                self._block_mask_cache.pop(next(iter(self._block_mask_cache)))
            self._block_mask_cache[key] = cached
        return cached

    # -------------------------------------------------------------------------
    # Training forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Exact-NLL training loss over the anchored clean stream.

        Args:
          idx: (bt, N) clean token chunk; N divisible by block_size.
          targets: ignored (train.py full-sequence interface).
        """
        del targets
        x0 = idx
        bt, n = x0.shape
        self._validate_seq_len(n)
        device = x0.device

        layout = self._get_layout(n, device)
        block_mask = self._get_block_mask(n, device)

        input_ids = torch.where(
            layout["is_anchor"].unsqueeze(0),
            layout["anchor_ids"].unsqueeze(0),
            x0[:, layout["gather_t"]],
        )
        rho = layout["rho"].unsqueeze(0).expand(bt, -1)

        x = self.drop(self.wte(input_ids) + self.type_emb(rho))
        positions = layout["positions"]
        for block in self.h:
            x = block(x, positions, block_mask, rho)
        x = self.ln_f(x)

        h_sel = x[:, layout["loss_pos"], :]
        tgt_sel = x0[:, layout["tgt_gather"]]

        logits = self.lm_head(h_sel)
        # fp32 for the wide-vocab CE; anchor tokens are never valid targets.
        with torch.amp.autocast("cuda", enabled=False):
            logits = logits.float()
            logits[..., self.vocab_size :] = float("-inf")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt_sel.reshape(-1),
            )
        return torch.empty(0), loss

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def _infer_pass(
        self,
        ids: torch.Tensor,
        rho_q: torch.Tensor,
        positions: torch.Tensor,
        caches: list[list[tuple[torch.Tensor, torch.Tensor]] | None],
        *,
        need_logits: bool,
    ) -> tuple[torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        """One forward over a short segment with concatenated cache context.

        Returns (logits or None, per-layer self K/V of this segment).
        """
        bt = ids.size(0)
        if rho_q.dim() == 1:
            rho_q_b = rho_q.unsqueeze(0).expand(bt, -1)
        else:
            rho_q_b = rho_q
        x = self.wte(ids) + self.type_emb(rho_q_b)
        self_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for li, block in enumerate(self.h):
            parts_k = []
            parts_v = []
            for cache in caches:
                if cache is not None and cache[li][0].size(2) > 0:
                    parts_k.append(cache[li][0])
                    parts_v.append(cache[li][1])
            ctx_k = torch.cat(parts_k, dim=2) if parts_k else None
            ctx_v = torch.cat(parts_v, dim=2) if parts_v else None
            x, k_self, v_self = block.forward_infer(
                x, positions, ctx_k, ctx_v, rho_q_b,
            )
            self_kv.append((k_self, v_self))
        if not need_logits:
            return None, self_kv
        x = self.ln_f(x)
        logits = self.lm_head(x).float()
        logits[..., self.vocab_size :] = float("-inf")
        return logits, self_kv

    def _empty_kv(
        self, bt: int, device: torch.device, dtype: torch.dtype,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.empty(bt, self.n_head, 0, self.h[0].attn.head_dim, device=device, dtype=dtype),
                torch.empty(bt, self.n_head, 0, self.h[0].attn.head_dim, device=device, dtype=dtype),
            )
            for _ in self.h
        ]

    @staticmethod
    def _append_kv(
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        new: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.cat([cache[li][0], new[li][0]], dim=2),
             torch.cat([cache[li][1], new[li][1]], dim=2))
            for li in range(len(cache))
        ]

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k,
    ) -> torch.Tensor:
        """Sample from (bt, V) logits; temperature <= 0 means argmax."""
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        scaled = logits / temperature
        if top_k is not None and int(top_k) > 0:
            kk = min(int(top_k), scaled.size(-1))
            vals, _ = torch.topk(scaled, kk)
            scaled = scaled.masked_fill(scaled < vals[..., -1, None], float("-inf"))
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        seqlen: int | None = None,
        *,
        bos_token_id: int | None = None,
        sampling_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Blockwise generation: discard s KV after each block; keep all t.

        Returns (tokens (num_samples, seqlen), nfe).
        """
        cfg = sampling_cfg or {}
        temperature = float(cfg.get("temperature", 1.0))
        top_k = cfg.get("top_k")

        if seqlen is None:
            raise ValueError("generate requires an explicit seqlen")
        self._validate_seq_len(seqlen)
        bos = bos_token_id if bos_token_id is not None else self.token_layout.bos_token_id

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        bt = num_samples
        bs, ns = self.block_size, self.num_anchors
        n_blocks = seqlen // bs
        nfe = 0

        cache_t = self._empty_kv(bt, device, dtype)   # all previous t (unbounded)

        anchor_ids = (
            torch.arange(ns, device=device) + self.anchor_index0
        ).unsqueeze(0).expand(bt, -1)
        rho_anchor = torch.full((ns,), ROLE_S, device=device, dtype=torch.long)
        rho_t = torch.full((1,), ROLE_T, device=device, dtype=torch.long)

        out = torch.empty(bt, seqlen, dtype=torch.long, device=device)

        for g in range(n_blocks):
            # ---- anchor step (context = all previous t; no historical s) -----
            pos_anchor = torch.full((ns,), g * bs, device=device, dtype=torch.long)
            logits, kv_anchor = self._infer_pass(
                anchor_ids, rho_anchor, pos_anchor,
                [cache_t], need_logits=True,
            )
            nfe += 1
            # Current-block s only (discarded after the block finishes).
            cache_s = kv_anchor

            if g == 0 and self.fix_bos:
                cur = torch.full((bt,), bos, dtype=torch.long, device=device)
            else:
                cur = self._sample_token(
                    logits[:, -1, :], temperature=temperature, top_k=top_k,
                )

            # ---- intra-block token-by-token decoding -------------------------
            blk_kv = self._empty_kv(bt, device, dtype)
            for j in range(bs):
                out[:, g * bs + j] = cur
                pos_j = torch.full((1,), g * bs + j, device=device, dtype=torch.long)
                # Last token's forward only realizes its KV for cache_t.
                need = j < bs - 1
                logits, kv_j = self._infer_pass(
                    cur.unsqueeze(1), rho_t, pos_j,
                    [cache_s, cache_t, blk_kv], need_logits=need,
                )
                nfe += 1
                blk_kv = self._append_kv(blk_kv, kv_j)
                if need:
                    cur = self._sample_token(
                        logits[:, -1, :], temperature=temperature, top_k=top_k,
                    )

            # ---- append clean t KV; drop current-block s KV -------------------
            cache_t = self._append_kv(cache_t, blk_kv)
            del cache_s

        return out, nfe


class FL_AR15Model(FL_PreTrainedModel):
    config_class = FL_AR15Config

    def __init__(self, config: FL_AR15Config) -> None:
        super().__init__(config)
        self.backbone = _AR15Backbone(**config.backbone_kwargs())


def build_model_from_config(config: FL_AR15Config) -> FL_AR15Model:
    ensure_token_layout(config)
    return FL_AR15Model(config)


def build_model(cfg: dict) -> FL_AR15Model:
    data, sampling = split_model_cfg(cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_AR15Config(**data)
    config.tokenizer = cfg.get("tokenizer", "gpt2")
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
