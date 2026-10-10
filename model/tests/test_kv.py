"""KV cache has to match no-cache greedy, token for token.

Prefill eats the whole prompt in one pass, decode goes one token at a
time with RoPE offset = how much is cached. We cover plain and GQA
configs and a few prompts (short, unicode, etc. Left-truncation truncates
the same on both sides, tested separately).
"""

import mlx.core as mx
import pytest

from nanomlx.config import ModelConfig, preset
from nanomlx.generate import generate
from nanomlx.model import GPT

VOCAB = 320


def _codec():
    enc = lambda s: [(ord(c) * 7 + 3) % VOCAB for c in s]
    dec = lambda ids: "".join(chr(i) for i in ids)
    return enc, dec


PROMPTS = [
    "hi",
    "hello world, this is a longer prompt with many tokens in it",
    "unicode 🎉 mixed 你好 text 123 !!!",
    "a",
]


@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_cache_equivalence_greedy(gqa: bool, prompt: str):
    mx.random.seed(11)
    if gqa:
        cfg = ModelConfig(
            vocab_size=VOCAB,
            n_layers=2,
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            context=64,
        )
    else:
        cfg = preset("debug", vocab_size=VOCAB, context=64)
    model = GPT(cfg)
    enc, dec = _codec()
    cached = generate(model, enc, dec, prompt, max_new_tokens=20, temperature=0.0, seed=0)
    nocache = generate(
        model,
        enc,
        dec,
        prompt,
        max_new_tokens=20,
        temperature=0.0,
        seed=0,
        use_cache=False,
    )
    assert cached == nocache, f"diverged on {prompt!r}: {cached} vs {nocache}"
    # Both sides hit the same context cap (no eos_id here): 64 minus prompt len.
    assert len(cached) == min(20, 64 - len(enc(prompt)))


def test_cache_sampling_deterministic():
    """Same seed plus temperature: cached runs agree with each other exactly.

    (Cached vs uncached *sampling* isn't checked: float adding happens in a
    different order between prefix-sliced and full attention, so draws can
    differ. Greedy matching above is the real check.)
    """
    mx.random.seed(12)
    model = GPT(preset("debug", vocab_size=VOCAB, context=64))
    enc, dec = _codec()
    kwargs = {"max_new_tokens": 15, "temperature": 0.9, "top_k": 20, "top_p": 0.9, "seed": 3}
    assert generate(model, enc, dec, "sampling test", **kwargs) == generate(
        model, enc, dec, "sampling test", **kwargs
    )


def test_prefill_single_pass_shapes():
    """Prefill takes the whole prompt at once (one forward, full width)."""
    from nanomlx.model import KVCache

    mx.random.seed(13)
    cfg = preset("debug", vocab_size=VOCAB, context=64)
    model = GPT(cfg)
    cache = KVCache(cfg, batch_size=1)
    ids = list(range(1, 25))
    logits = model(mx.array([ids]), offset=0, caches=cache.layers)
    mx.eval(logits)
    cache.seq_len = len(ids)
    assert logits.shape == (1, 24, VOCAB)
    assert cache.seq_len == 24
    # One decode step is just a single spot.
    logits = model(mx.array([[ids[-1]]]), offset=cache.seq_len, caches=cache.layers)
    mx.eval(logits)
    assert logits.shape == (1, 1, VOCAB)


def test_first_token_comes_from_prefill():
    """Off-by-one guard: token N has to equal argmax of the full forward
    at the last prompt spot (not some re-fed dupe)."""
    mx.random.seed(14)
    model = GPT(preset("debug", vocab_size=VOCAB, context=64))
    enc, dec = _codec()
    prompt = "the first generated token is decided here"
    ids = enc(prompt)
    full = model(mx.array([ids]))
    mx.eval(full)
    expected = int(mx.argmax(full[0, -1]))
    got = generate(model, enc, dec, prompt, max_new_tokens=1, temperature=0.0, seed=0)
    assert got == [expected], f"{got} != {[expected]}"
