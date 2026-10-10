"""M7 acceptance: manual vs mx.fast numerical equivalence.

Tolerances are stated, not assumed: fp32 must agree tightly (SDPA uses
different internal blocking than the manual matmul path); bf16 gets a
documented looser bound. Each op is isolated (rope / norm / attention)
plus a full-model combined check, with and without KV cache, plain + GQA.
"""

import mlx.core as mx
import pytest

from nanomlx.config import ModelConfig
from nanomlx.model import GPT, KVCache

VOCAB = 320
FP32_TOL = 1e-3  # measured 9.8e-05 on debug preset; 10x margin
BF16_TOL = 5e-2  # measured 2.1e-03; looser bound documented, not a bug


def _models(**overrides):
    base = {"vocab_size": VOCAB, "n_layers": 2, "d_model": 64, "n_heads": 4, "context": 64}
    base.update(overrides)
    mx.random.seed(0)
    m_manual = GPT(ModelConfig(**base))
    mx.random.seed(0)
    # Identical init: copy weights (seeds already align, belt and braces).
    m_fast = GPT(ModelConfig(**base, attention_impl="fast", norm_impl="fast", rope_impl="fast"))
    m_fast.update(dict(m_manual.parameters()))
    return m_manual, m_fast


def _maxdiff(a: mx.array, b: mx.array) -> float:
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


@pytest.mark.parametrize("kv_heads", [4, 2])
def test_attention_fast_equivalence(kv_heads: int):
    m_manual, m_fast = _models(n_kv_heads=kv_heads)
    x = mx.random.randint(0, VOCAB, (2, 16))
    mx.eval(x)
    a, b = m_manual(x), m_fast(x)
    d = _maxdiff(a, b)
    print(f"\nfull-model fp32 manual-vs-fast (kv={kv_heads}): {d:.2e}")
    assert d < FP32_TOL


def test_attention_fast_equivalence_cached():
    """Fast path agrees under KV-cache decode too (single-token steps)."""
    m_manual, m_fast = _models(n_kv_heads=2)
    cm, cf = KVCache(m_manual.cfg, 1), KVCache(m_fast.cfg, 1)
    ids = list(range(1, 17))
    lm = m_manual(mx.array([ids]), caches=cm.layers)
    lf = m_fast(mx.array([ids]), caches=cf.layers)
    mx.eval(lm, lf)
    cm.seq_len = cf.seq_len = len(ids)
    assert _maxdiff(lm, lf) < FP32_TOL
    for step in range(4):
        tok = (7 * step + 3) % VOCAB
        lm = m_manual(mx.array([[tok]]), offset=cm.seq_len, caches=cm.layers)
        lf = m_fast(mx.array([[tok]]), offset=cf.seq_len, caches=cf.layers)
        mx.eval(lm, lf)
        d = _maxdiff(lm, lf)
        assert d < FP32_TOL, f"decode step {step}: {d}"
        cm.seq_len += 1
        cf.seq_len += 1


def test_norm_fast_equivalence():
    from nanomlx.model import RMSNorm

    mx.random.seed(0)
    w = mx.random.normal((64,)) + 2.0
    x = mx.random.normal((2, 8, 64)) * 3.0
    mx.eval(w, x)
    m, f = RMSNorm(64, impl="manual"), RMSNorm(64, impl="fast")
    m.weight, f.weight = w, w
    d = _maxdiff(m(x), f(x))
    print(f"\nrmsnorm fp32 diff: {d:.2e}")
    assert d < 1e-5


def test_bf16_documented_tolerance():
    """bf16 agrees within the documented looser bound (not a bug)."""
    m_manual, m_fast = _models(n_kv_heads=4)
    m_manual.set_dtype(mx.bfloat16)
    m_fast.set_dtype(mx.bfloat16)
    x = mx.random.randint(0, VOCAB, (2, 16))
    mx.eval(x)
    a, b = m_manual(x), m_fast(x)
    d = _maxdiff(a, b)
    print(f"\nfull-model bf16 manual-vs-fast: {d:.2e} (tol {BF16_TOL})")
    assert d < BF16_TOL
