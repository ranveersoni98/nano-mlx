"""Model + training configs.

This is where shapes, presets, and param counts live. The tokenizer
owns the vocab, the model just gets told `vocab_size`. Tests use the
tiny 320-id fixture vocab, real training uses the 32k one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Our decoder-only transformer setup. Llama-ish."""

    vocab_size: int = 32000
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 6  # has to divide n_heads for GQA
    context: int = 256
    ffn_hidden: int | None = None  # None means 8/3 * d_model, rounded up to a mult of 64
    rms_eps: float = 1e-5
    tie_weights: bool = True  # reuse embeddings as the output head
    bias: bool = False  # we just don't use biases
    dropout: float = 0.0
    attention_impl: str = "manual"  # keep "manual" around; "fast" uses mx.fast
    norm_impl: str = "manual"  # "manual" | "fast" (mx.fast.rms_norm) | "metal" (our fused kernel)
    rope_impl: str = "manual"  # "manual" | "fast" (mx.fast.rope)
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        for name in ["attention_impl"]:
            assert getattr(self, name) in ("manual", "fast"), f"{name} must be manual|fast"
        assert self.norm_impl in ("manual", "fast", "metal"), "norm_impl must be manual|fast|metal"
        for name in ["rope_impl"]:
            assert getattr(self, name) in ("manual", "fast"), f"{name} must be manual|fast"
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be a multiple of n_kv_heads ({self.n_kv_heads})"
        )
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        if self.ffn_hidden is None:
            hidden = int(8 * self.d_model / 3)
            self.ffn_hidden = ((hidden + 63) // 64) * 64

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def param_count(self, with_embeddings: bool = True) -> int:
        """How many params. We report both numbers since embeddings
        dominate the small models. Tests check this against the real model."""
        f = self.ffn_hidden
        assert f is not None, "ffn_hidden must be set by __post_init__"
        d, h, kv, L, V = (
            self.d_model,
            self.n_heads,
            self.n_kv_heads,
            self.n_layers,
            self.vocab_size,
        )
        hd = d // h
        # Per layer: q (d*d) + k/v (d * kv*hd each) + o (d*d) + 2 norms (d each)
        # + SwiGLU gate/up (d*f each) + down (f*d).
        attn = d * d + 2 * d * kv * hd + d * d
        norms = 2 * d
        mlp = 3 * d * f
        per_layer = attn + norms + mlp
        core = L * per_layer + d  # + final RMSNorm weight
        if not with_embeddings:
            return core
        emb = V * d
        head = 0 if self.tie_weights else V * d
        return core + emb + head


@dataclass
class TrainConfig:
    """Knobs for the training loop."""

    preset: str = "tiny"
    peak_lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    total_steps: int = 1000
    batch_size: int = 8
    grad_accum: int = 1
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    eval_interval: int = 100
    eval_iters: int = 20
    seed: int = 1337
    dtype: str = "bfloat16"
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8


# Starting sizes. Tune to whatever RAM you've got. Real counts come
# from ModelConfig.param_count(), not the comments here.
PRESETS: dict[str, dict] = {
    "debug": {"n_layers": 2, "d_model": 64, "n_heads": 4, "n_kv_heads": 4, "context": 128},
    "tiny": {"n_layers": 6, "d_model": 384, "n_heads": 6, "n_kv_heads": 6, "context": 256},
    "small": {"n_layers": 8, "d_model": 512, "n_heads": 8, "n_kv_heads": 4, "context": 512},
    "base": {"n_layers": 12, "d_model": 768, "n_heads": 12, "n_kv_heads": 4, "context": 1024},
}

# Rough LR starting points per size.
PRESET_LR: dict[str, float] = {
    "debug": 1e-3,
    "tiny": 1e-3,
    "small": 6e-4,
    "base": 3e-4,
}


def preset(name: str, vocab_size: int = 32000, **overrides) -> ModelConfig:
    """Grab a named preset as a ModelConfig."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = dict(PRESETS[name])
    cfg.update(overrides)
    return ModelConfig(vocab_size=vocab_size, **cfg)


def train_preset(name: str, total_steps: int = 1000, **overrides) -> TrainConfig:
    warmup = min(500, max(100, total_steps // 50))
    cfg = {
        "preset": name,
        "peak_lr": PRESET_LR[name],
        "warmup_steps": warmup,
        "total_steps": total_steps,
    }
    cfg.update(overrides)
    return TrainConfig(**cfg)
