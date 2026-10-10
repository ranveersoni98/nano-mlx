"""Decoder-only transformer in MLX (M4): manual attention, kept permanently.

Architecture (AGENTS.md §7.1): token embedding -> N x pre-norm block ->
final RMSNorm -> LM head. Attention is multi-head with optional GQA, RoPE on
Q/K, causal masking, no absolute positions. MLP is SwiGLU. No biases by
default. Weight tying is a config flag (default on for small presets).

Precision: compute dtype follows the parameters (usually bfloat16); norms,
softmax inputs, and the loss stay float32 with explicit casts (M4 tests run
float32 throughout for determinism).

The `mx.fast` variants land in M7 behind `attention_impl="fast"`; this
manual path stays as the numerical reference.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from nanomlx.config import ModelConfig


class RMSNorm(nn.Module):
    """RMSNorm without bias. Manual path computes in float32 explicitly;
    `fast` delegates to `mx.fast.rms_norm`, `metal` to the M9 fused kernel
    (all equivalence-tested)."""

    def __init__(self, dims: int, eps: float = 1e-5, impl: str = "manual"):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        assert impl in ("manual", "fast", "metal")
        self.impl = impl

    def __call__(self, x: mx.array) -> mx.array:
        if self.impl == "fast":
            return mx.fast.rms_norm(x, self.weight, self.eps)
        if self.impl == "metal":
            from nanomlx.metal import fused_rms_norm

            return fused_rms_norm(x, self.weight, self.eps)
        dtype = x.dtype
        xf = x.astype(mx.float32)
        var = mx.mean(mx.square(xf), axis=-1, keepdims=True)
        normed = xf * mx.rsqrt(var + self.eps)
        return (normed * self.weight.astype(mx.float32)).astype(dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float = 10000.0, offset: int = 0
) -> tuple[mx.array, mx.array]:
    """(cos, sin) tables of shape (seq_len, head_dim // 2) for positions
    [offset, offset + seq_len). head_dim must be even."""
    assert head_dim % 2 == 0, f"RoPE needs even head_dim, got {head_dim}"
    half = head_dim // 2
    freqs = 1.0 / (theta ** (mx.arange(half, dtype=mx.float32) / half))
    positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
    angles = positions[:, None] * freqs[None, :]
    mx.eval(angles)
    return mx.cos(angles), mx.sin(angles)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate (..., T, head_dim). cos/sin: (T, head_dim // 2), broadcast
    over batch and heads."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # (1, 1, T, half) so batch/head dims broadcast.
    c = cos[None, None, :, :].astype(x.dtype)
    s = sin[None, None, :, :].astype(x.dtype)
    rot_first = x1 * c - x2 * s
    rot_second = x2 * c + x1 * s
    out = mx.empty_like(x)
    out[..., 0::2] = rot_first
    out[..., 1::2] = rot_second
    return out


class KVCache:
    """Hand-written per-layer K/V store (M6).

    Post-RoPE K and V, shape per layer (B, n_kv_heads, max_len, head_dim) -
    GQA-aware (stores kv heads, expands at attention time). Preallocated to
    `max_len` and written in place via slice updates (no per-step concat).
    `seq_len` tracks filled positions; positions are absolute (cache always
    starts at 0, so RoPE offset == row index).
    """

    def __init__(self, cfg: ModelConfig, batch_size: int = 1, max_len: int | None = None):
        self.cfg = cfg
        self.max_len = max_len or cfg.context
        kv, hd = cfg.n_kv_heads, cfg.head_dim
        self.layers = [
            {
                "k": mx.zeros((batch_size, kv, self.max_len, hd)),
                "v": mx.zeros((batch_size, kv, self.max_len, hd)),
            }
            for _ in range(cfg.n_layers)
        ]
        self.seq_len = 0


class CausalSelfAttention(nn.Module):
    """Manual scaled dot-product attention with GQA + RoPE (M4 path)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, h, kv, hd = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(d, h * hd, bias=cfg.bias)
        self.k_proj = nn.Linear(d, kv * hd, bias=cfg.bias)
        self.v_proj = nn.Linear(d, kv * hd, bias=cfg.bias)
        self.o_proj = nn.Linear(h * hd, d, bias=cfg.bias)
        self.n_rep = h // kv

    def __call__(self, x: mx.array, offset: int = 0, cache: dict | None = None) -> mx.array:
        B, T, _ = x.shape
        cfg = self.cfg
        h, kv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        # Project + split heads.
        q = self.q_proj(x).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, kv, hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, kv, hd).transpose(0, 2, 1, 3)
        # RoPE at absolute positions (offset>0 is the KV-cache decode path).
        if cfg.rope_impl == "fast":
            # traditional=True matches our interleaved-pairs layout (probed).
            q = mx.fast.rope(q, hd, traditional=True, base=cfg.rope_theta, scale=1.0, offset=offset)
            k = mx.fast.rope(k, hd, traditional=True, base=cfg.rope_theta, scale=1.0, offset=offset)
        else:
            cos, sin = build_rope_cache(T, hd, cfg.rope_theta, offset)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            # In-place slice update; attend over the filled prefix.
            cache["k"][:, :, offset : offset + T, :] = k
            cache["v"][:, :, offset : offset + T, :] = v
            S = offset + T
            k = cache["k"][:, :, :S, :]
            v = cache["v"][:, :, :S, :]
        else:
            S = T
        scale = 1.0 / math.sqrt(hd)
        q_pos = mx.arange(offset, offset + T)[:, None]
        k_pos = (
            mx.arange(S)[None, :] if cache is not None else mx.arange(offset, offset + T)[None, :]
        )
        if cfg.attention_impl == "fast":
            # Native GQA (no head repeat). Square no-offset prefill takes the
            # fused causal kernel; every other shape (cached decode, offset
            # forwards) uses an explicit boolean keep-mask (True = attend).
            # Both are equivalence-tested against the manual path.
            if cache is None and offset == 0:
                attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
            else:
                keep = (q_pos >= k_pos)[None, :, :]
                attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=keep)
            out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, h * hd)
        else:
            # GQA: repeat KV heads to match Q heads.
            if self.n_rep > 1:
                k = mx.repeat(k, self.n_rep, axis=1)
                v = mx.repeat(v, self.n_rep, axis=1)
            # Causal scores in float32.
            scores = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * scale
            scores = mx.where(q_pos < k_pos, mx.array(-1e30, mx.float32), scores)
            weights = mx.softmax(scores, axis=-1).astype(x.dtype)
            out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, h * hd)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """down(silu(gate(x)) * up(x)). gate/up: d->f, down: f->d."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        f = cfg.ffn_hidden
        assert f is not None, "ffn_hidden must be set by ModelConfig.__post_init__"
        self.gate = nn.Linear(cfg.d_model, f, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, f, bias=cfg.bias)
        self.down = nn.Linear(f, cfg.d_model, bias=cfg.bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm residual block: x += Attn(Norm(x)); x += MLP(Norm(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps, impl=cfg.norm_impl)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_eps, impl=cfg.norm_impl)
        self.mlp = SwiGLU(cfg)

    def __call__(self, x: mx.array, offset: int = 0, cache: dict | None = None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), offset=offset, cache=cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    """The model. `forward(idx)` -> float32 logits (B, T, vocab_size)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps, impl=cfg.norm_impl)
        self.lm_head = (
            None if cfg.tie_weights else nn.Linear(cfg.d_model, cfg.vocab_size, bias=cfg.bias)
        )
        self._init_weights(cfg)

    def _init_weights(self, cfg: ModelConfig) -> None:
        # Embeddings + all linears: N(0, 0.02); residual-output projections
        # (attn o_proj, MLP down) scaled by 1/sqrt(2*n_layers) (GPT-2 style).
        import mlx.core as _mx  # local: keeps module top-level tidy

        res_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        self.tok_emb.weight = (
            _mx.random.normal((cfg.vocab_size, cfg.d_model), dtype=_mx.float32) * 0.02
        )
        for blk in self.blocks:
            for proj, scale in [
                (blk.attn.q_proj, 0.02),
                (blk.attn.k_proj, 0.02),
                (blk.attn.v_proj, 0.02),
                (blk.attn.o_proj, 0.02 * res_scale),
                (blk.mlp.gate, 0.02),
                (blk.mlp.up, 0.02),
                (blk.mlp.down, 0.02 * res_scale),
            ]:
                w = _mx.random.normal(proj.weight.shape, dtype=_mx.float32) * scale
                proj.weight = w
                if cfg.bias and proj.bias is not None:
                    proj.bias = _mx.zeros(proj.bias.shape, dtype=_mx.float32)
        if self.lm_head is not None:
            self.lm_head.weight = (
                _mx.random.normal((cfg.vocab_size, cfg.d_model), dtype=_mx.float32) * 0.02
            )

    def forward(self, idx: mx.array, offset: int = 0, caches: list[dict] | None = None) -> mx.array:
        """idx: (B, T) int tokens -> (B, T, V) float32 logits.

        `caches`: one per-layer dict from a [`KVCache`], or None for a full
        forward. With caches, only `idx`'s tokens are processed (prefill the
        prompt once, then decode one token at a time at offset=cached len).
        """
        x = self.tok_emb(idx)
        for i, blk in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x = blk(x, offset=offset, cache=cache)
        x = self.final_norm(x).astype(mx.float32)
        if self.lm_head is None:
            head_w = self.tok_emb.weight.astype(mx.float32)
        else:
            head_w = self.lm_head.weight.astype(mx.float32)
        return x @ head_w.T

    def __call__(
        self, idx: mx.array, offset: int = 0, caches: list[dict] | None = None
    ) -> mx.array:
        return self.forward(idx, offset=offset, caches=caches)

    def count_params(self) -> tuple[int, int]:
        """(total, without embeddings). Tests assert == config formula."""
        total = sum(int(v.size) for _, v in tree_flatten(dict(self.parameters())))
        emb = int(self.tok_emb.weight.size)
        head = 0 if self.lm_head is None else int(self.lm_head.weight.size)
        return total, total - emb - head


def tree_flatten(tree, _path=()):
    """Yield (dotted.path, array) leaves of an MLX param tree."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from tree_flatten(v, _path + (str(k),))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from tree_flatten(v, _path + (str(i),))
    else:
        yield (".".join(_path), tree)
