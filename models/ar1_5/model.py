"""AR1.5: AR2-like anchors without long-range s KV / t window.

Sequence layout matches AR2 (clean A + M noise B copies). Differences:

Attention visibility:
  (a) only the *current* block's anchors are visible (no historical s);
  (b) any query sees *all* previous clean t (copy A; no window);
  (c) own-block t bidirectional (same as AR2).

Inference: after each block, discard s KV; append clean t into an unbounded
cache_t. Anchor step / block refine only see current s + all previous t.
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
ROLE_T_CLEAN = 1
ROLE_T_MASK = 2

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
    the full ``L×L`` score matrix (OOM at AR1.5 seq len ≈ 3k).
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
    num_blocks: int,
    num_noise_copies: int,
):
    """Visibility predicate for AR1.5 over [A | B_0..B_{M-1}]."""
    bs = block_size
    ns = num_anchors
    la = num_blocks * (bs + ns)
    kb = num_blocks * bs

    def mask_mod(b, h, q_idx, kv_idx):
        del b, h
        q_in_a = q_idx < la
        kv_in_a = kv_idx < la

        q_off_a = q_idx % (bs + ns)
        q_blk_a = q_idx // (bs + ns)
        q_is_s = q_in_a & (q_off_a < ns)

        kv_off_a = kv_idx % (bs + ns)
        kv_blk_a = kv_idx // (bs + ns)
        kv_is_s = kv_in_a & (kv_off_a < ns)

        qb = q_idx - la
        q_copy = qb // kb
        q_blk_b = (qb % kb) // bs
        kvb = kv_idx - la
        kv_copy = kvb // kb
        kv_blk_b = (kvb % kb) // bs

        q_blk = torch.where(q_in_a, q_blk_a, q_blk_b)
        kv_blk = torch.where(kv_in_a, kv_blk_a, kv_blk_b)

        # (a) current-block anchors only (historical s never visible)
        see_s = kv_is_s & (kv_blk == q_blk)
        # (b) all previous clean t from copy A (no window)
        kv_clean_t = kv_in_a & ~kv_is_s
        see_prev_t = kv_clean_t & (kv_blk <= q_blk - 1)
        # (c) own block, bidirectional, t queries only
        same_blk_a = q_in_a & ~q_is_s & kv_clean_t & (kv_blk_a == q_blk_a)
        same_blk_b = (
            (~q_in_a) & (~kv_in_a) & (q_copy == kv_copy) & (q_blk_b == kv_blk_b)
        )
        return see_s | see_prev_t | same_blk_a | same_blk_b

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
        self.role_q = nn.Embedding(3, n_embd) if attn_type_bias else None
        self.role_k = nn.Embedding(3, n_embd) if attn_type_bias else None
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
        mask_ratio_min: float = 0.05,
        num_noise_copies: int = 2,
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
        self.mask_ratio_min = mask_ratio_min
        self.num_noise_copies = num_noise_copies
        self.fix_bos = fix_bos

        # Vocab layout: [tokenizer vocab | [MASK] | <s_0>..<s_{ns-1}>]
        self.vocab_size = token_layout.vocab_size
        self.mask_index = token_layout.vocab_size
        self.anchor_index0 = token_layout.vocab_size + 1
        self.model_vocab_size = token_layout.vocab_size + 1 + num_anchors

        self.wte = nn.Embedding(self.model_vocab_size, n_embd)
        self.type_emb = nn.Embedding(3, n_embd)
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
        """Static per-(seq_len) tensors: positions, copy-A template/roles."""
        key = (n, device)
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached

        bs, ns, m = self.block_size, self.num_anchors, self.num_noise_copies
        k = n // bs
        # Copy A: per block [anchors | tokens]
        idx_a = torch.arange(k * (bs + ns), device=device)
        off_a = idx_a % (bs + ns)
        blk_a = idx_a // (bs + ns)
        a_is_anchor = off_a < ns
        # anchors take the block-start original index; t tokens their own index
        pos_a = torch.where(a_is_anchor, blk_a * bs, blk_a * bs + off_a - ns)
        # gather index from x0 for copy-A t positions (dummy 0 at anchors)
        gather_a = (blk_a * bs + (off_a - ns).clamp(min=0)).long()
        anchor_ids = self.anchor_index0 + torch.where(
            a_is_anchor, off_a, torch.zeros_like(off_a),
        )
        rho_a = torch.where(
            a_is_anchor,
            torch.full_like(off_a, ROLE_S),
            torch.full_like(off_a, ROLE_T_CLEAN),
        )
        pos_b = torch.arange(n, device=device).repeat(m)
        positions = torch.cat([pos_a, pos_b]).long()

        cached = {
            "a_is_anchor": a_is_anchor,
            "gather_a": gather_a,
            "anchor_ids": anchor_ids.long(),
            "rho_a": rho_a.long(),
            "positions": positions,
        }
        if len(self._layout_cache) >= 8:
            self._layout_cache.pop(next(iter(self._layout_cache)))
        self._layout_cache[key] = cached
        return cached

    def _get_block_mask(self, n: int, device: torch.device):
        key = (n, device)
        cached = self._block_mask_cache.get(key)
        if cached is None:
            bs, ns, m = self.block_size, self.num_anchors, self.num_noise_copies
            k = n // bs
            total = k * (bs + ns) + m * n
            mask_mod = make_ar15_mask_mod(
                block_size=bs,
                num_anchors=ns,
                num_blocks=k,
                num_noise_copies=m,
            )
            cached = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device,
            )
            if len(self._block_mask_cache) >= 8:
                self._block_mask_cache.pop(next(iter(self._block_mask_cache)))
            self._block_mask_cache[key] = cached
        return cached

    def _sample_block_masks(
        self, x0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample per-block mask ratios and Bernoulli masks for the M noise copies.

        Returns:
          mask: (bt, M, N) bool — True where the position is masked
          t_blk: (bt, M, K) float — per-block mask ratio (NELBO weight is 1/t)
        """
        bt, n = x0.shape
        bs = self.block_size
        k = n // bs
        m = self.num_noise_copies
        device = x0.device

        t_blk = torch.rand(bt, m, k, device=device)
        t_blk = self.mask_ratio_min + t_blk * (1.0 - self.mask_ratio_min)
        mask = (
            torch.rand(bt, m, k, bs, device=device) < t_blk.unsqueeze(-1)
        )

        # Force >= 1 masked position per block (loss signal for every block).
        forced = torch.randint(0, bs, (bt, m, k), device=device)
        empty = ~mask.any(dim=-1)
        mask.scatter_(
            -1, forced.unsqueeze(-1), (empty | mask.gather(-1, forced.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1),
        )

        if self.fix_bos:
            # Never mask the BOS at position 0; re-force block 0 if it went empty.
            mask[:, :, 0, 0] = False
            empty0 = ~mask[:, :, 0, :].any(dim=-1)
            forced0 = torch.randint(1, bs, (bt, m), device=device)
            mask[:, :, 0, :].scatter_(
                -1, forced0.unsqueeze(-1), (empty0 | mask[:, :, 0, :].gather(-1, forced0.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1),
            )

        return mask.view(bt, m, n), t_blk

    # -------------------------------------------------------------------------
    # Training forward
    # -------------------------------------------------------------------------

    def _train_hidden(
        self, x0: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Hidden states of the noise copies for a given mask.

        Args:
          x0: (bt, N) clean tokens; mask: (bt, M, N) bool.
        Returns:
          (bt, M, N, d) post-ln_f hidden states of copies B_0..B_{M-1}.
        """
        bt, n = x0.shape
        device = x0.device
        bs, ns, m = self.block_size, self.num_anchors, self.num_noise_copies

        layout = self._get_layout(n, device)
        block_mask = self._get_block_mask(n, device)

        # Copy A ids: anchors + clean tokens
        ids_a = torch.where(
            layout["a_is_anchor"].unsqueeze(0),
            layout["anchor_ids"].unsqueeze(0),
            x0[:, layout["gather_a"]],
        )
        # Noise copies: masked positions replaced by [MASK]
        ids_b = torch.where(
            mask, self.mask_index, x0.unsqueeze(1).expand(bt, m, n),
        ).reshape(bt, m * n)
        input_ids = torch.cat([ids_a, ids_b], dim=1)

        rho_b = torch.where(mask, ROLE_T_MASK, ROLE_T_CLEAN).reshape(bt, m * n)
        rho = torch.cat(
            [layout["rho_a"].unsqueeze(0).expand(bt, -1), rho_b], dim=1,
        ).long()

        x = self.drop(self.wte(input_ids) + self.type_emb(rho))
        positions = layout["positions"]
        for block in self.h:
            x = block(x, positions, block_mask, rho)
        x = self.ln_f(x)

        la = (n // bs) * (bs + ns)
        return x[:, la:, :].reshape(bt, m, n, -1)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """NELBO training loss on the [A | B_0..B_{M-1}] concatenated sequence.

        Args:
          idx: (bt, N) clean token chunk; N divisible by block_size.
          targets: ignored (train.py full-sequence interface).
        """
        del targets
        x0 = idx
        bt, n = x0.shape
        self._validate_seq_len(n)
        bs, m = self.block_size, self.num_noise_copies

        mask, t_blk = self._sample_block_masks(x0)
        h_b = self._train_hidden(x0, mask)

        x0_rep = x0.unsqueeze(1).expand(bt, m, n)
        mask_flat = mask.reshape(-1)
        h_sel = h_b.reshape(-1, h_b.size(-1))[mask_flat]
        tgt_sel = x0_rep.reshape(-1)[mask_flat]
        w_sel = (
            (1.0 / t_blk)
            .unsqueeze(-1)
            .expand(bt, m, n // bs, bs)
            .reshape(-1)[mask_flat]
        )

        logits = self.lm_head(h_sel)
        # fp32 for the wide-vocab CE; special tokens are never valid targets.
        with torch.amp.autocast("cuda", enabled=False):
            logits = logits.float()
            logits[:, self.vocab_size :] = float("-inf")
            ce = F.cross_entropy(logits, tgt_sel, reduction="none")
            loss = (ce * w_sel.float()).sum() / (bt * m * n)
        return torch.empty(0), loss

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def _infer_pass(
        self,
        ids: torch.Tensor,
        rho_q: torch.Tensor,
        positions: torch.Tensor,
        cache_s: list[tuple[torch.Tensor, torch.Tensor]] | None,
        cache_t: list[tuple[torch.Tensor, torch.Tensor]] | None,
        rho_ctx: torch.Tensor | None,
        *,
        need_logits: bool,
    ) -> tuple[torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        """One forward over a short segment with [cache_s | cache_t] context.

        Returns (logits or None, per-layer self K/V of this segment).
        ``rho_ctx`` is unused (role is baked into cached K via role_k).
        """
        del rho_ctx
        bt = ids.size(0)
        if rho_q.dim() == 1:
            rho_q_b = rho_q.unsqueeze(0).expand(bt, -1)
        else:
            rho_q_b = rho_q
        x = self.wte(ids) + self.type_emb(rho_q_b)
        self_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for li, block in enumerate(self.h):
            ctx_k = ctx_v = None
            parts_k = []
            parts_v = []
            if cache_s is not None and cache_s[li][0].size(2) > 0:
                parts_k.append(cache_s[li][0])
                parts_v.append(cache_s[li][1])
            if cache_t is not None and cache_t[li][0].size(2) > 0:
                parts_k.append(cache_t[li][0])
                parts_v.append(cache_t[li][1])
            if parts_k:
                ctx_k = torch.cat(parts_k, dim=2)
                ctx_v = torch.cat(parts_v, dim=2)
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
        commit_threshold = float(cfg.get("commit_threshold", 0.9))
        max_refine_iters = int(cfg.get("max_refine_iters", self.block_size))

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

        def empty_kv() -> list[tuple[torch.Tensor, torch.Tensor]]:
            return [
                (
                    torch.empty(bt, self.n_head, 0, self.h[0].attn.head_dim, device=device, dtype=dtype),
                    torch.empty(bt, self.n_head, 0, self.h[0].attn.head_dim, device=device, dtype=dtype),
                )
                for _ in self.h
            ]

        cache_t = empty_kv()   # all previous clean t (unbounded)
        rho_t_len = 0

        anchor_ids = (
            torch.arange(ns, device=device) + self.anchor_index0
        ).unsqueeze(0).expand(bt, -1)
        rho_anchor = torch.full((ns,), ROLE_S, device=device, dtype=torch.long)

        out = torch.empty(bt, seqlen, dtype=torch.long, device=device)

        for g in range(n_blocks):
            # ---- anchor step (context = all previous t only; no historical s) ---
            pos_anchor = torch.full((ns,), g * bs, device=device, dtype=torch.long)
            rho_ctx_t = torch.full(
                (rho_t_len,), ROLE_T_CLEAN, device=device, dtype=torch.long,
            )
            _, kv_anchor = self._infer_pass(
                anchor_ids, rho_anchor, pos_anchor,
                None, cache_t, rho_ctx_t, need_logits=False,
            )
            nfe += 1
            # Current-block s only (discarded after the block finishes).
            cache_s = kv_anchor
            rho_ctx = torch.cat([
                torch.full((ns,), ROLE_S, device=device, dtype=torch.long),
                rho_ctx_t,
            ])

            # ---- iterative block infill ----------------------------------------
            pos_block = g * bs + torch.arange(bs, device=device, dtype=torch.long)
            y = torch.full((bt, bs), self.mask_index, dtype=torch.long, device=device)
            committed = torch.zeros(bt, bs, dtype=torch.bool, device=device)
            if g == 0 and self.fix_bos:
                y[:, 0] = bos
                committed[:, 0] = True
            cand = y.clone()

            for _ in range(max_refine_iters):
                if bool(committed.all()):
                    break
                rho_q = torch.where(committed, ROLE_T_CLEAN, ROLE_T_MASK).long()
                logits, _ = self._infer_pass(
                    y, rho_q, pos_block, cache_s, cache_t, rho_ctx, need_logits=True,
                )
                nfe += 1
                if temperature <= 0.0:
                    probs = F.softmax(logits, dim=-1)
                    conf, cand = probs.max(dim=-1)
                else:
                    scaled = logits / temperature
                    if top_k is not None and int(top_k) > 0:
                        kk = min(int(top_k), scaled.size(-1))
                        vals, _ = torch.topk(scaled, kk)
                        scaled = scaled.masked_fill(
                            scaled < vals[..., -1, None], float("-inf"),
                        )
                    probs = F.softmax(scaled, dim=-1)
                    cand = torch.multinomial(
                        probs.view(-1, probs.size(-1)), 1,
                    ).view(bt, bs)
                    conf = probs.gather(-1, cand.unsqueeze(-1)).squeeze(-1)

                conf = conf.masked_fill(committed, float("-inf"))
                commit = (conf >= commit_threshold) & ~committed
                # Guarantee progress: commit at least the most confident position.
                none_new = ~commit.any(dim=-1) & ~committed.all(dim=-1)
                best = conf.argmax(dim=-1)
                commit[none_new, best[none_new]] = True

                y = torch.where(commit, cand, y)
                committed |= commit

            if not bool(committed.all()):
                # R_max fallback: commit remaining positions from the last proposal.
                y = torch.where(committed, y, cand)

            out[:, g * bs : (g + 1) * bs] = y

            # ---- finalize: push clean t KV; discard current-block s KV ----------
            rho_clean = torch.full((bs,), ROLE_T_CLEAN, device=device, dtype=torch.long)
            _, kv_clean = self._infer_pass(
                y, rho_clean, pos_block, cache_s, cache_t, rho_ctx, need_logits=False,
            )
            nfe += 1
            cache_t = [
                (
                    torch.cat([cache_t[li][0], kv_clean[li][0]], dim=2),
                    torch.cat([cache_t[li][1], kv_clean[li][1]], dim=2),
                )
                for li in range(len(self.h))
            ]
            rho_t_len = cache_t[0][0].size(2)
            # Explicitly drop s so it cannot leak into the next block.
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
