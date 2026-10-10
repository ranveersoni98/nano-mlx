"""M9 stretch: custom Metal fused RMSNorm vs built-ins.

Correctness across shapes/dtypes (incl. bf16 rounding bound) plus a
full-model smoke run with norm_impl="metal".
"""

import mlx.core as mx
import pytest

from nanomlx.config import ModelConfig
from nanomlx.metal import fused_rms_norm
from nanomlx.model import GPT, RMSNorm

VOCAB = 320


def _maxdiff(a: mx.array, b: mx.array) -> float:
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


@pytest.mark.parametrize("shape", [(4, 64), (2048, 384), (512, 512), (1, 768)])
def test_metal_matches_manual_fp32(shape):
    mx.random.seed(0)
    x = mx.random.normal(shape) * 3.0
    w = mx.random.normal((shape[-1],)) + 2.0
    mx.eval(x, w)
    m = RMSNorm(shape[-1])
    m.weight = w
    d = _maxdiff(m(x), fused_rms_norm(x, w, 1e-5))
    assert d < 1e-5, f"{shape}: {d}"


def test_metal_bf16_rounding_bound():
    """bf16 agrees to 1 ulp (0.03125 at these magnitudes) - rounding, not a bug."""
    mx.random.seed(0)
    xb = (mx.random.normal((256, 384)) * 3.0).astype(mx.bfloat16)
    wb = mx.random.normal((384,)) + 2.0
    mx.eval(xb, wb)
    m = RMSNorm(384)
    m.weight = wb
    out = fused_rms_norm(xb, wb, 1e-5)
    assert out.dtype == mx.bfloat16
    assert _maxdiff(m(xb), out) <= 0.03125


def test_metal_full_model_runs():
    mx.random.seed(0)
    base = {
        "vocab_size": VOCAB,
        "n_layers": 2,
        "d_model": 64,
        "n_heads": 4,
        "n_kv_heads": 4,
        "context": 64,
    }
    manual = GPT(ModelConfig(**base))
    metal = GPT(ModelConfig(**base, norm_impl="metal"))
    metal.update(dict(manual.parameters()))
    x = mx.random.randint(0, VOCAB, (2, 16))
    mx.eval(x)
    d = _maxdiff(manual(x), metal(x))
    print(f"\nfull-model manual-vs-metal fp32: {d:.2e}")
    assert d < 1e-3
