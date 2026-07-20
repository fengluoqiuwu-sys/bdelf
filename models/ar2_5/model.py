"""AR2.5: block-end generated <s> + bridge predicts block start.

Spec: temp/ar2_5.md

Layout (ns=1): (X_1, s_1, ..., X_{K-1}, s_{K-1}, X_K), length N+(K-1).

Teacher-forcing inputs / targets (block k, 0-based):
  - Block 0: standard next-token; last t of non-final -> <s>.
  - Block k>0: t_off=0 (bridge) input=<s> target=tok[0];
    t_off=1..B-2 input=tok[t_off-1] target=tok[t_off];
    t_off=B-1 input=tok[B-1] target=<s> (non-final) or none (final).
  - Anchor slots: no CE.

Inference matches: width-2 boundary writes s KV + bridge KV (like t_off=0);
then decode with input=previous token at subsequent offsets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.ar2_5.config import FL_AR25Config
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


def fused_flex_attention(q, k, v, block_mask=None):
    if torch.compiler.is_dynamo_compiling():
        return flex_attention(q, k, v, block_mask=block_mask)
    global _flex_attention_compiled
    if _flex_attention_compiled is None:
        _flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_attention_compiled(q, k, v, block_mask=block_mask)


def make_ar25_mask_mod(*, block_size: int, t_window: int, n_blocks: int):
    bs = block_size
    period = bs + 1
    w = t_window
    last_start = (n_blocks - 1) * period

    def _decode(idx):
        in_last = idx >= last_start
        blk = torch.where(in_last, n_blocks - 1, idx // period)
        off = torch.where(in_last, idx - last_start, idx % period)
        is_s = (~in_last) & (off == bs)
        t_off = torch.where(is_s, torch.zeros_like(off), off)
        return blk, t_off, is_s

    def mask_mod(b, h, q_idx, kv_idx):
        del b, h
        q_blk, q_toff, q_is_s = _decode(q_idx)
        kv_blk, kv_toff, kv_is_s = _decode(kv_idx)
        see_s = kv_is_s & (kv_blk <= q_blk)
        see_window = ~kv_is_s & (kv_blk >= q_blk - w) & (kv_blk <= q_blk - 1)
        same_blk_causal = (
            ~q_is_s & ~kv_is_s & (kv_blk == q_blk) & (kv_toff <= q_toff)
        )
        s_see_own_t = q_is_s & ~kv_is_s & (kv_blk == q_blk)
        return see_s | see_window | same_blk_causal | s_see_own_t

    return mask_mod


class Ar25Attention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, attn_type_bias):
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
        self.role_q = nn.Embedding(2, n_embd) if attn_type_bias else None
        self.role_k = nn.Embedding(2, n_embd) if attn_type_bias else None
        if self.role_q is not None:
            nn.init.zeros_(self.role_q.weight)
            nn.init.zeros_(self.role_k.weight)

    def _project_qkv(self, x, positions, rho=None):
        bsz, seq_len, _ = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        if self.role_q is not None and rho is not None:
            q = q + self.role_q(rho)
            k = k + self.role_k(rho)

        def heads(t):
            return t.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        q, k = self.rope.apply_qk(q, k, positions)
        return q, k, v

    def forward(self, x, positions, block_mask, rho):
        q, k, v = self._project_qkv(x, positions, rho)
        y = fused_flex_attention(q, k, v, block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.n_embd)
        return self.resid_dropout(self.c_proj(y)), q, k, v

    def forward_infer(self, x, positions, ctx_k, ctx_v, rho=None, causal_self=False):
        q, k_self, v_self = self._project_qkv(x, positions, rho)
        q_len = q.size(2)
        if ctx_k is None:
            k, v = k_self, v_self
            ctx_len = 0
        else:
            k = torch.cat([ctx_k, k_self], dim=2)
            v = torch.cat([ctx_v, v_self], dim=2)
            ctx_len = ctx_k.size(2)
        attn_mask = None
        if causal_self and q_len > 1:
            total = ctx_len + q_len
            attn_mask = q.new_zeros(q_len, total)
            rows = torch.arange(q_len, device=q.device).unsqueeze(1)
            cols = torch.arange(q_len, device=q.device).unsqueeze(0)
            attn_mask[:, ctx_len:] = torch.where(
                cols > rows, q.new_full((), float("-inf")), q.new_zeros(()),
            )
            attn_mask = attn_mask.view(1, 1, q_len, total)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.n_embd)
        return self.c_proj(y), k_self, v_self


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, attn_type_bias):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = Ar25Attention(n_embd, n_head, dropout, attn_type_bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, positions, block_mask, rho):
        attn_out, q, k, v = self.attn(self.ln_1(x), positions, block_mask, rho)
        x = x + attn_out
        return x + self.mlp(self.ln_2(x)), q, k, v

    def forward_infer(self, x, positions, ctx_k, ctx_v, rho=None, causal_self=False):
        attn_out, k_self, v_self = self.attn.forward_infer(
            self.ln_1(x), positions, ctx_k, ctx_v, rho, causal_self=causal_self,
        )
        x = x + attn_out
        return x + self.mlp(self.ln_2(x)), k_self, v_self


class _AR25Backbone(nn.Module):
    full_sequence_training = True

    def __init__(
        self,
        token_layout: FL_TokenLayout,
        max_seq_len: int = 8192,
        block_size: int = 16,
        num_anchors: int = 1,
        t_window: int = 4,
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
            raise ValueError("AR2.5 only implements attn_backend=flex")
        if not FLEX_ATTN_AVAILABLE:
            raise RuntimeError("AR2.5 requires PyTorch FlexAttention")
        if block_size < 2:
            raise ValueError("block_size must be >= 2")
        if num_anchors != 1:
            raise ValueError("AR2.5 requires num_anchors=1")

        self.token_layout = token_layout
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.num_anchors = 1
        self.t_window = t_window
        self.n_head = n_head
        self.fix_bos = fix_bos
        # AR3 sets True so checkpoint outputs retain QKV for oracle align.
        self.collect_qkv = False
        self.vocab_size = token_layout.vocab_size
        self.anchor_index0 = token_layout.vocab_size
        self.model_vocab_size = token_layout.vocab_size + 1

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
        self._block_mask_cache: dict = {}
        self._layout_cache: dict = {}
        self.last_metrics: dict[str, float] = {}

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _validate_seq_len(self, seq_len: int) -> None:
        if seq_len % self.block_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by block_size "
                f"({self.block_size})"
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

    def _ext_len(self, n: int) -> int:
        return n + max(n // self.block_size - 1, 0)

    def _get_layout(self, n: int, device: torch.device) -> dict[str, torch.Tensor]:
        key = (n, device)
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached

        bs = self.block_size
        n_blocks = n // bs
        total = self._ext_len(n)
        period = bs + 1
        last_start = (n_blocks - 1) * period

        idx = torch.arange(total, device=device)
        in_last = idx >= last_start
        blk = torch.where(in_last, n_blocks - 1, idx // period)
        off = torch.where(in_last, idx - last_start, idx % period)
        is_s = (~in_last) & (off == bs)
        t_off = torch.where(is_s, torch.zeros_like(off), off)
        is_bridge = (~is_s) & (t_off == 0) & (blk > 0)

        positions = torch.where(
            is_s, blk * bs + (bs - 1), blk * bs + t_off,
        ).long()
        rho = torch.where(
            is_s,
            torch.full_like(off, ROLE_S),
            torch.full_like(off, ROLE_T),
        ).long()

        # Content gather for input ids (before bridge override):
        # blk0: token at t_off; blk>0 & t_off>=1: token at t_off-1 (shift AR)
        tok_at = (blk * bs + t_off).long()
        tok_prev = (blk * bs + (t_off - 1).clamp(min=0)).long()
        is_shift = (blk > 0) & (~is_s) & (t_off >= 1)
        gather_input = torch.where(is_shift, tok_prev, tok_at)

        # Targets (spec §4). Block has B t-slots:
        #   blk0: t_off=0..B-2 → next content; non-final t_off=B-1 → <s>
        #   blk>0: bridge → tok[0]; t_off=1..B-2 → tok[t_off]; non-final B-1 → <s>
        # Middle-block tok[B-1] has no CE slot (B slots cannot hold B content + <s>).
        is_bridge_loss = is_bridge
        is_blk0_next = (blk == 0) & (~is_s) & (t_off < bs - 1)
        is_last_t_nonfinal = (~is_s) & (t_off == bs - 1) & (~in_last)
        is_shift_next = is_shift & (~is_last_t_nonfinal)
        gather_input = torch.where(is_last_t_nonfinal, tok_at, gather_input)
        is_pred_s = is_last_t_nonfinal

        is_content_loss = is_bridge_loss | is_blk0_next | is_shift_next
        loss_mask = is_content_loss | is_pred_s
        loss_pos = torch.nonzero(loss_mask, as_tuple=False).squeeze(-1)

        tgt_x0 = torch.where(
            is_bridge,
            tok_at,
            torch.where(is_blk0_next, tok_at + 1, tok_at),
        ).long()

        token_loss_pos = torch.nonzero(is_content_loss, as_tuple=False).squeeze(-1)

        cached = {
            "is_s": is_s,
            "is_bridge": is_bridge,
            "gather_input": gather_input,
            "rho": rho,
            "positions": positions,
            "loss_pos": loss_pos.long(),
            "tgt_x0": tgt_x0,
            "tgt_is_anchor": is_pred_s,
            "token_loss_pos": token_loss_pos.long(),
        }
        if len(self._layout_cache) >= 8:
            self._layout_cache.pop(next(iter(self._layout_cache)))
        self._layout_cache[key] = cached
        return cached

    def _get_block_mask(self, n: int, device: torch.device):
        key = (n, device)
        cached = self._block_mask_cache.get(key)
        if cached is None:
            total = self._ext_len(n)
            mask_mod = make_ar25_mask_mod(
                block_size=self.block_size,
                t_window=self.t_window,
                n_blocks=n // self.block_size,
            )
            cached = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device,
            )
            if len(self._block_mask_cache) >= 8:
                self._block_mask_cache.pop(next(iter(self._block_mask_cache)))
            self._block_mask_cache[key] = cached
        return cached

    def forward(self, idx, targets=None):
        del targets
        x0 = idx
        bt, n = x0.shape
        self._validate_seq_len(n)
        device = x0.device
        layout = self._get_layout(n, device)
        block_mask = self._get_block_mask(n, device)

        content = x0[:, layout["gather_input"]]
        use_anchor = layout["is_s"] | layout["is_bridge"]
        anchor_row = torch.full(
            (1, layout["gather_input"].numel()),
            self.anchor_index0, device=device, dtype=torch.long,
        )
        input_ids = torch.where(use_anchor.unsqueeze(0), anchor_row, content)
        rho = layout["rho"].unsqueeze(0).expand(bt, -1)
        x = self.drop(self.wte(input_ids) + self.type_emb(rho))
        positions = layout["positions"]

        layer_qkv: list = []
        collect_qkv = self.collect_qkv and self.training
        for block in self.h:
            if self.training:
                if collect_qkv:
                    x, q, k, v = checkpoint(
                        block, x, positions, block_mask, rho, use_reentrant=False,
                    )
                    layer_qkv.append((q, k, v))
                else:
                    # Drop QKV so checkpoint does not retain them for backward.
                    x = checkpoint(
                        lambda *args: block(*args)[0],
                        x, positions, block_mask, rho, use_reentrant=False,
                    )
            else:
                out = block(x, positions, block_mask, rho)
                x = out[0]
                if collect_qkv:
                    layer_qkv.append(out[1:])
        if collect_qkv:
            self._last_layer_qkv = layer_qkv
            self._last_layout = layout
            self._last_n = n
        else:
            self._last_layer_qkv = None
            self._last_n = n
        x = self.ln_f(x)

        loss_pos = layout["loss_pos"]
        h_sel = x[:, loss_pos, :]
        tgt_is_anchor = layout["tgt_is_anchor"][loss_pos]
        tgt_sel = torch.where(
            tgt_is_anchor.unsqueeze(0).expand(bt, -1),
            torch.full((bt, loss_pos.numel()), self.anchor_index0, device=device),
            x0[:, layout["tgt_x0"][loss_pos]],
        )

        logits = self.lm_head(h_sel)
        with torch.amp.autocast("cuda", enabled=False):
            logits = logits.float()
            masked = logits.clone()
            masked[..., self.vocab_size :] = float("-inf")
            ai = tgt_is_anchor.unsqueeze(0).unsqueeze(-1).expand_as(logits)
            logits = torch.where(ai, logits, masked)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt_sel.reshape(-1),
            )

            tpos = layout["token_loss_pos"]
            t_logits = self.lm_head(x[:, tpos, :]).float()
            t_logits[..., self.vocab_size :] = float("-inf")
            t_tgt = x0[:, layout["tgt_x0"][tpos]]
            loss_token = F.cross_entropy(
                t_logits.reshape(-1, t_logits.size(-1)), t_tgt.reshape(-1),
            )

        self.last_metrics = {
            "ppl_incl_s": float(loss.detach().exp().item()),
            "ppl_token": float(loss_token.detach().exp().item()),
            "loss_incl_s": float(loss.detach().item()),
            "loss_token": float(loss_token.detach().item()),
        }
        return torch.empty(0), loss

    def _infer_pass(
        self, ids, rho_q, positions, caches, *,
        need_logits: bool, causal_self: bool = False, allow_anchor_logits: bool = False,
    ):
        bt = ids.size(0)
        rho_b = rho_q.unsqueeze(0).expand(bt, -1) if rho_q.dim() == 1 else rho_q
        x = self.wte(ids) + self.type_emb(rho_b)
        self_kv = []
        for li, block in enumerate(self.h):
            parts_k, parts_v = [], []
            for cache in caches:
                if cache is not None and cache[li][0].size(2) > 0:
                    parts_k.append(cache[li][0])
                    parts_v.append(cache[li][1])
            ctx_k = torch.cat(parts_k, dim=2) if parts_k else None
            ctx_v = torch.cat(parts_v, dim=2) if parts_v else None
            x, k_self, v_self = block.forward_infer(
                x, positions, ctx_k, ctx_v, rho_b, causal_self=causal_self,
            )
            self_kv.append((k_self, v_self))
        if not need_logits:
            return None, self_kv
        logits = self.lm_head(self.ln_f(x)).float()
        if not allow_anchor_logits:
            logits[..., self.vocab_size :] = float("-inf")
        return logits, self_kv

    def _empty_kv(self, bt, device, dtype):
        hd = self.h[0].attn.head_dim
        z = torch.empty(bt, self.n_head, 0, hd, device=device, dtype=dtype)
        return [(z, z.clone()) for _ in self.h]

    @staticmethod
    def _append_kv(cache, new):
        return [
            (torch.cat([cache[i][0], new[i][0]], dim=2),
             torch.cat([cache[i][1], new[i][1]], dim=2))
            for i in range(len(cache))
        ]

    @staticmethod
    def _slice_kv(kv, start, end):
        return [(kv[i][0][:, :, start:end], kv[i][1][:, :, start:end]) for i in range(len(kv))]

    @staticmethod
    def _sample_token(logits, *, temperature: float):
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(-1)

    def _slide_cache_t(self, cache_t, blk_kv):
        keep = (self.t_window - 1) * self.block_size
        return [
            (
                torch.cat([cache_t[i][0][:, :, -keep:], blk_kv[i][0]], dim=2)
                if keep > 0 else blk_kv[i][0],
                torch.cat([cache_t[i][1][:, :, -keep:], blk_kv[i][1]], dim=2)
                if keep > 0 else blk_kv[i][1],
            )
            for i in range(len(self.h))
        ]

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        seqlen: int | None = None,
        *,
        bos_token_id: int | None = None,
        prefix_tokens: torch.Tensor | None = None,
        sampling_cfg: dict | None = None,
    ):
        cfg = sampling_cfg or {}
        temperature = float(cfg.get("temperature", 1.0))
        if seqlen is None:
            raise ValueError("generate requires an explicit seqlen")
        self._validate_seq_len(seqlen)
        bos = bos_token_id if bos_token_id is not None else self.token_layout.bos_token_id
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        bt = num_samples
        bs = self.block_size
        n_blocks = seqlen // bs
        nfe = 0

        prefix_len = 0
        if prefix_tokens is not None:
            if prefix_tokens.dim() != 2 or prefix_tokens.size(0) != num_samples:
                raise ValueError("prefix_tokens shape must be (num_samples, prefix_len)")
            prefix_len = int(prefix_tokens.size(1))
            if not (1 <= prefix_len < seqlen):
                raise ValueError("prefix length must be in [1, seqlen)")
            prefix_tokens = prefix_tokens.to(device=device, dtype=torch.long)

        def _at(pos, sampled):
            return prefix_tokens[:, pos] if pos < prefix_len else sampled

        cache_s = self._empty_kv(bt, device, dtype)
        cache_t = self._empty_kv(bt, device, dtype)
        out = torch.empty(bt, seqlen, dtype=torch.long, device=device)
        aid = torch.full((bt, 1), self.anchor_index0, device=device, dtype=torch.long)
        rho_t = torch.tensor([ROLE_T], device=device, dtype=torch.long)

        cur = _at(0, torch.full((bt,), bos, device=device, dtype=torch.long))
        bridge_kv = None

        for g in range(n_blocks):
            blk_kv = self._empty_kv(bt, device, dtype)
            if g == 0:
                # Standard AR within block 0
                for j in range(bs):
                    out[:, j] = cur
                    pos = torch.tensor([j], device=device, dtype=torch.long)
                    need = j < bs - 1
                    logits, kv = self._infer_pass(
                        cur.unsqueeze(1), rho_t, pos,
                        [cache_s, cache_t, blk_kv], need_logits=need,
                    )
                    nfe += 1
                    blk_kv = self._append_kv(blk_kv, kv)
                    if need:
                        cur = _at(j + 1, self._sample_token(logits[:, -1], temperature=temperature))
            else:
                # First slot from bridge; cur = block-start token.
                out[:, g * bs] = cur
                blk_kv = self._append_kv(blk_kv, bridge_kv)
                bridge_kv = None
                prev = cur  # tok[0]
                # t_off=1..B-2: in=tok[j-1] -> tok[j]
                for j in range(1, bs - 1):
                    pos = torch.tensor([g * bs + j], device=device, dtype=torch.long)
                    logits, kv = self._infer_pass(
                        prev.unsqueeze(1), rho_t, pos,
                        [cache_s, cache_t, blk_kv], need_logits=True,
                    )
                    nfe += 1
                    blk_kv = self._append_kv(blk_kv, kv)
                    nxt = _at(
                        g * bs + j,
                        self._sample_token(logits[:, -1], temperature=temperature),
                    )
                    out[:, g * bs + j] = nxt
                    prev = nxt
                # t_off=B-1
                pos_last = torch.tensor([g * bs + bs - 1], device=device, dtype=torch.long)
                if g == n_blocks - 1:
                    # final block: in=tok[B-2] -> tok[B-1]
                    logits, kv = self._infer_pass(
                        prev.unsqueeze(1), rho_t, pos_last,
                        [cache_s, cache_t, blk_kv], need_logits=True,
                    )
                    nfe += 1
                    blk_kv = self._append_kv(blk_kv, kv)
                    last = _at(
                        g * bs + bs - 1,
                        self._sample_token(logits[:, -1], temperature=temperature),
                    )
                    out[:, g * bs + bs - 1] = last
                    cur = last
                else:
                    # non-final: train last slot is in=tok[B-1] -> <s>
                    # First sample tok[B-1] (not directly supervised), then TF last slot.
                    logits, _ = self._infer_pass(
                        prev.unsqueeze(1), rho_t, pos_last,
                        [cache_s, cache_t, blk_kv], need_logits=True,
                    )
                    nfe += 1
                    last = _at(
                        g * bs + bs - 1,
                        self._sample_token(logits[:, -1], temperature=temperature),
                    )
                    out[:, g * bs + bs - 1] = last
                    _, kv = self._infer_pass(
                        last.unsqueeze(1), rho_t, pos_last,
                        [cache_s, cache_t, blk_kv], need_logits=False,
                    )
                    nfe += 1
                    blk_kv = self._append_kv(blk_kv, kv)
                    cur = last

            cache_t = self._slide_cache_t(cache_t, blk_kv)
            if g == n_blocks - 1:
                break

            # width-2: [s | bridge]
            pos = torch.tensor([g * bs + bs - 1, (g + 1) * bs], device=device, dtype=torch.long)
            logits, kv_b = self._infer_pass(
                torch.cat([aid, aid], dim=1),
                torch.tensor([ROLE_S, ROLE_T], device=device, dtype=torch.long),
                pos, [cache_s, cache_t], need_logits=True, causal_self=True,
            )
            nfe += 1
            cache_s = self._append_kv(cache_s, self._slice_kv(kv_b, 0, 1))
            bridge_kv = self._slice_kv(kv_b, 1, 2)
            cur = _at(
                (g + 1) * bs,
                self._sample_token(logits[:, 1], temperature=temperature),
            )

        return out, nfe


class FL_AR25Model(FL_PreTrainedModel):
    config_class = FL_AR25Config

    def __init__(self, config: FL_AR25Config) -> None:
        super().__init__(config)
        self.backbone = _AR25Backbone(**config.backbone_kwargs())


def build_model_from_config(config: FL_AR25Config) -> FL_AR25Model:
    ensure_token_layout(config)
    return FL_AR25Model(config)


def build_model(cfg: dict) -> FL_AR25Model:
    data, sampling = split_model_cfg(cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_AR25Config(**data)
    config.tokenizer = cfg.get("tokenizer", "gpt2")
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
