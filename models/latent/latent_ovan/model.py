"""对称块因果 Text VAE（csh BlockVAE 结构 + 本机 gpt2 词表）。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.latent.encdec.readout import kl_gaussian
from models.latent.latent_ovan.config import FL_LatentOvanConfig
from models.latent.latent_ovan.layers import CausalBlock, VaeHeads, xavier_linear
from models.model import (
    FL_PreTrainedModel,
    ensure_token_layout,
    sample_from_logits,
    split_model_cfg,
)
from models.rope import RotaryEmbedding
from models.tokens import FL_TokenLayout, apply_token_layout_to_config, token_layout_from_cfg


def slotwise_noise(
    z0: torch.Tensor,
    t: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """z = t·z₀ + (1−t)·ε，ε ~ N(0, σ²I)。"""
    eps = torch.randn_like(z0) * sigma
    return t.unsqueeze(-1) * z0 + (1.0 - t.unsqueeze(-1)) * eps


class _LatentOvanBackbone(nn.Module):
    full_sequence_training = True
    supports_prefix = True

    def __init__(
        self,
        token_layout: FL_TokenLayout,
        max_seq_len: int = 4096,
        n_layer_enc: int = 6,
        n_layer_dec: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        d_kv: int = 64,
        mlp_mult: int = 4,
        latent_dim: int = 16,
        dropout: float = 0.1,
        beta_kl: float = 0.05,
        lambda_mask: float = 1.0,
        mask_ratio: float = 0.1,
        mask_rand_frac: float = 1.0,
        z_zero_ratio: float = 0.1,
        recon_t_mean: float = 0.5,
        recon_t_std: float = 1.0,
        noise_sigma: float = 1.0,
        sample_vocab_size: int = 50257,
        qk_norm: bool = True,
        tie_wte_normalize: bool = True,
        use_flash: bool = True,
        attn_backend: str = "sdpa",
        block_size: int = 16,
    ) -> None:
        super().__init__()
        if block_size > 1 and max_seq_len % block_size != 0:
            raise ValueError(
                f"max_seq_len={max_seq_len} 须能被 block_size={block_size} 整除"
            )
        self.token_layout = token_layout
        self.vocab_size = token_layout.vocab_size
        self.mask_token_id = self.vocab_size
        self.max_seq_len = max_seq_len
        self.n_embd = n_embd
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.lambda_mask = lambda_mask
        self.mask_ratio = mask_ratio
        self.mask_rand_frac = float(mask_rand_frac)
        self.z_zero_ratio = float(z_zero_ratio)
        self.recon_t_mean = float(recon_t_mean)
        self.recon_t_std = float(recon_t_std)
        self.noise_sigma = float(noise_sigma)
        hi = int(sample_vocab_size)
        self.rand_vocab_size = hi if hi > 0 else int(token_layout.pad_token_id)
        self.tie_wte_normalize = bool(tie_wte_normalize)
        self.block_size = block_size
        self.enc_drop = float(dropout)

        self.wte = nn.Embedding(self.vocab_size + 1, n_embd)
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        self.rope = RotaryEmbedding(d_kv, max_seq_len=max_seq_len)
        block_kw = dict(
            n_embd=n_embd,
            n_head=n_head,
            d_kv=d_kv,
            dropout=dropout,
            mlp_mult=mlp_mult,
            qk_norm=qk_norm,
            use_flash=use_flash,
            attn_backend=attn_backend,
            block_size=block_size,
        )
        self.enc_blocks = nn.ModuleList([
            CausalBlock(**block_kw) for _ in range(n_layer_enc)
        ])
        self.enc_ln = nn.LayerNorm(n_embd, bias=False)
        self.vae_heads = VaeHeads(n_embd, latent_dim)
        self.from_latent = nn.Linear(latent_dim, n_embd)
        xavier_linear(self.from_latent)
        self.dec_blocks = nn.ModuleList([
            CausalBlock(**block_kw) for _ in range(n_layer_dec)
        ])
        self.dec_ln = nn.LayerNorm(n_embd, bias=False)

        self.last_ce_loss = float("nan")
        self.last_kl_loss = float("nan")
        self.last_mask_loss = float("nan")
        self.last_token_acc = float("nan")
        self.last_mask_acc = float("nan")

    def _pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens == self.token_layout.pad_token_id

    def _attn_pad_mask(self, tokens: torch.Tensor) -> torch.Tensor | None:
        """逐 token 因果不加 pad mask（以便 Flash）；块因果才传 bool pad。"""
        if self.block_size <= 1:
            return None
        return self._pad_mask(tokens)

    def _loss_targets(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.masked_fill(
            tokens == self.token_layout.pad_token_id,
            self.token_layout.ignore_index,
        )

    def _ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.token_layout.ignore_index,
        )

    def _ce_on(self, logits: torch.Tensor, tokens: torch.Tensor, take: torch.Tensor) -> torch.Tensor:
        ignore = self.token_layout.ignore_index
        raw = self._ce(logits, tokens.masked_fill(~take, ignore))
        n = take.to(dtype=raw.dtype).sum()
        return torch.where(
            n > 0, raw, torch.zeros((), device=tokens.device, dtype=raw.dtype),
        )

    def _token_acc(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        pred = logits.argmax(dim=-1)
        n = valid.float().sum()
        acc = ((pred == tokens) & valid).float().sum() / n.clamp_min(1.0)
        return torch.where(
            n > 0, acc, torch.full((), float("nan"), device=tokens.device),
        )

    def _run_blocks(
        self,
        x: torch.Tensor,
        blocks: nn.ModuleList,
        ln_f: nn.Module,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device, dtype=torch.long)
        for blk in blocks:
            x = blk(
                x,
                key_padding_mask=key_padding_mask,
                positions=positions,
                rope=self.rope,
            )
        return ln_f(x)

    def encode(
        self,
        tokens: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad = self._attn_pad_mask(tokens)
        x = self.wte(tokens)
        if self.training and self.enc_drop > 0.0:
            x = F.dropout(x, p=self.enc_drop, training=True)
        h = self._run_blocks(x, self.enc_blocks, self.enc_ln, pad)
        return self.vae_heads(h, sample=sample)

    def _logits_from_h(self, h: torch.Tensor) -> torch.Tensor:
        h = F.gelu(h, approximate="tanh")
        wte = self.wte.weight[: self.vocab_size]
        if self.tie_wte_normalize:
            scale = wte.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
            wte = wte / scale.to(dtype=wte.dtype)
        return F.linear(h, wte)

    def decode_logits(
        self,
        z: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        last_n: int | None = None,
    ) -> torch.Tensor:
        x = self.from_latent(z)
        if self.training and self.enc_drop > 0.0:
            x = F.dropout(x, p=self.enc_drop, training=True)
        h = self._run_blocks(x, self.dec_blocks, self.dec_ln, key_padding_mask)
        if last_n is not None:
            n = int(last_n)
            if n < 0:
                raise ValueError(f"last_n 须非负，收到 {n}")
            if n < int(h.size(1)):
                h = h[:, -n:]
        return self._logits_from_h(h)

    def _sample_token_mask(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self.mask_ratio <= 0:
            return torch.zeros_like(valid)
        mask = torch.rand(tokens.shape, device=tokens.device) < self.mask_ratio
        mask = mask & valid
        mask[:, 0] = False
        return mask

    def _encoder_input(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """抽中位写入 encoder：frac=1 全随机词，0 全 [MASK]。"""
        frac = float(self.mask_rand_frac)
        if frac >= 1.0:
            rnd = torch.randint(
                0, max(int(self.rand_vocab_size), 1),
                tokens.shape, device=tokens.device, dtype=tokens.dtype,
            )
            return torch.where(mask, rnd, tokens)
        enc = torch.where(
            mask, torch.full_like(tokens, self.mask_token_id), tokens,
        )
        if frac <= 0.0:
            return enc
        rnd_pos = mask & (torch.rand(tokens.shape, device=tokens.device) < frac)
        rnd = torch.randint(
            0, max(int(self.rand_vocab_size), 1),
            tokens.shape, device=tokens.device, dtype=tokens.dtype,
        )
        return torch.where(rnd_pos, rnd, enc)

    def _apply_z_rand(self, z: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.z_zero_ratio > 0.0):
            return z
        drop = torch.rand(z.shape, device=z.device, dtype=z.dtype) < self.z_zero_ratio
        return torch.where(drop, torch.randn_like(z), z)

    def _apply_recon_noise(self, z: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.recon_t_std > 0.0):
            return z
        u = torch.randn(tokens.shape, device=z.device, dtype=z.dtype)
        t = torch.sigmoid(self.recon_t_mean + self.recon_t_std * u)
        return slotwise_noise(z, t, self.noise_sigma)

    def bert_mask_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """联合训 / 评测脚本：掩码 encode、无 recon 噪声、只算掩码 CE。"""
        valid = ~self._pad_mask(tokens)
        mask = self._sample_token_mask(tokens, valid)
        enc_in = self._encoder_input(tokens, mask)
        z_m, _, _ = self.encode(enc_in, sample=self.training)
        logits_m = self.decode_logits(
            z_m, key_padding_mask=self._attn_pad_mask(tokens),
        )
        with torch.no_grad():
            self.last_mask_acc = self._token_acc(logits_m, tokens, mask).detach()
        return self._ce_on(logits_m, tokens, mask)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets, kwargs
        tokens = idx
        valid = ~self._pad_mask(tokens)
        do_mask = self.lambda_mask > 0 and self.mask_ratio > 0
        mask = self._sample_token_mask(tokens, valid) if do_mask else torch.zeros_like(valid)
        enc_in = self._encoder_input(tokens, mask) if do_mask else tokens
        z, mu, logvar = self.encode(enc_in, sample=self.training)
        z = self._apply_z_rand(z)
        z_dec = self._apply_recon_noise(z, tokens)
        pad_enc = self._attn_pad_mask(tokens)
        logits = self.decode_logits(z_dec, key_padding_mask=pad_enc)
        vis = valid & ~mask
        recon = self._ce_on(logits, tokens, vis)
        mask_loss = self._ce_on(logits, tokens, mask)
        kl = kl_gaussian(mu, logvar, mask=valid)
        self.last_ce_loss = recon.detach()
        self.last_kl_loss = kl.detach()
        self.last_mask_loss = mask_loss.detach()
        with torch.no_grad():
            self.last_token_acc = self._token_acc(logits, tokens, vis).detach()
            self.last_mask_acc = self._token_acc(logits, tokens, mask).detach()
        if not self.training:
            return logits, recon
        loss = recon + self.beta_kl * kl + self.lambda_mask * mask_loss
        return logits, loss

    def train_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for attr, key in (
            ("last_ce_loss", "recon_ce"),
            ("last_kl_loss", "kl"),
            ("last_mask_loss", "mask"),
            ("last_token_acc", "token_acc"),
            ("last_mask_acc", "mask_acc"),
        ):
            val = getattr(self, attr, None)
            if val is None:
                continue
            if hasattr(val, "item"):
                val = val.item()
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if fval == fval:
                out[key] = fval
        out["beta_kl"] = float(self.beta_kl)
        out["lambda_mask"] = float(self.lambda_mask)
        return out

    def online_eval_components(self) -> list:
        return []

    @torch.compiler.disable
    @torch.no_grad()
    def reconstruct(
        self,
        tokens: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        z, _, _ = self.encode(tokens, sample=False)
        logits = self.decode_logits(
            z, key_padding_mask=self._attn_pad_mask(tokens),
        )
        return sample_from_logits(logits, temperature=temperature, top_k=top_k)

    @torch.compiler.disable
    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        seqlen: int | None = None,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        bos_token_id: int | None = None,
        prefix_tokens: torch.Tensor | None = None,
        sampling_cfg: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        del bos_token_id
        cfg = sampling_cfg or {}
        temperature = float(cfg.get("temperature", temperature))
        top_k = cfg.get("top_k", top_k)
        if top_k is not None:
            top_k = int(top_k)
        seqlen = int(seqlen or self.max_seq_len)
        device = next(self.parameters()).device
        if prefix_tokens is not None:
            prefix = prefix_tokens.to(device=device, dtype=torch.long)
            if prefix.size(0) != num_samples:
                raise ValueError("prefix_tokens batch must match num_samples")
            prefix_len = prefix.size(1)
            if prefix_len >= seqlen:
                return prefix[:, :seqlen], 0
            pad = torch.full(
                (num_samples, seqlen - prefix_len),
                self.token_layout.pad_token_id,
                device=device,
                dtype=torch.long,
            )
            enc_tokens = torch.cat([prefix, pad], dim=1)
            z, _, _ = self.encode(enc_tokens, sample=False)
            logits = self.decode_logits(
                z, key_padding_mask=self._attn_pad_mask(enc_tokens),
            )
            rest = sample_from_logits(
                logits[:, prefix_len:, :], temperature=temperature, top_k=top_k,
            )
            return torch.cat([prefix, rest], dim=1), 1
        z = torch.randn(num_samples, seqlen, self.latent_dim, device=device)
        logits = self.decode_logits(z)
        out = sample_from_logits(logits, temperature=temperature, top_k=top_k)
        return out, 1


class FL_LatentOvanModel(FL_PreTrainedModel):
    config_class = FL_LatentOvanConfig

    def __init__(self, config: FL_LatentOvanConfig) -> None:
        super().__init__(config)
        self.backbone = _LatentOvanBackbone(**config.backbone_kwargs())
        self.post_init()


def build_model_from_config(config: FL_LatentOvanConfig) -> FL_LatentOvanModel:
    ensure_token_layout(config)
    return FL_LatentOvanModel(config)


def build_model(model_cfg: dict) -> FL_LatentOvanModel:
    data, sampling = split_model_cfg(model_cfg)
    layout = token_layout_from_cfg(data)
    data.pop("tokenizer", None)
    for key in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        data.pop(key, None)
    config = FL_LatentOvanConfig(**data)
    apply_token_layout_to_config(config, layout)
    if sampling is not None:
        config.sampling = sampling
    return build_model_from_config(config)
