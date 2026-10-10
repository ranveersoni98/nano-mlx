"""Generation tests (M5, no-cache): sampling filters + determinism."""

import mlx.core as mx
import numpy as np

from nanomlx.config import preset
from nanomlx.generate import generate, sample_next
from nanomlx.model import GPT

VOCAB = 64


def _model():
    mx.random.seed(0)
    return GPT(preset("debug", vocab_size=VOCAB, context=32))


def _codec():
    return lambda s: [ord(c) % VOCAB for c in s], lambda ids: "".join(chr(i % 1114112) for i in ids)


def test_topk1_equals_greedy():
    rng = np.random.default_rng(0)
    logits = np.random.default_rng(1).normal(size=VOCAB)
    greedy = sample_next(logits, 0.0, 0, 1.0, 1.0, [], rng)
    top1 = sample_next(logits, 1.0, 1, 1.0, 1.0, [], np.random.default_rng(0))
    assert greedy == top1 == int(np.argmax(logits))


def test_topp10_pure_temperature():
    logits = np.random.default_rng(2).normal(size=VOCAB)
    a = sample_next(logits, 0.7, 0, 1.0, 1.0, [], np.random.default_rng(9))
    b = sample_next(logits, 0.7, 0, 1.0, 1.0, [], np.random.default_rng(9))
    assert a == b  # seeded reproducible


def test_topp_one_equals_pure_temperature():
    """top-p=1.0 with top-k disabled == pure temperature sampling (§9/M6):
    checked against an independent inline implementation."""
    logits = np.random.default_rng(4).normal(size=VOCAB)
    temp = 0.8
    l = logits / temp
    probs = np.exp(l - l.max())
    probs /= probs.sum()
    rng = np.random.default_rng(5)
    got = [sample_next(logits, temp, 0, 1.0, 1.0, [], rng) for _ in range(5)]
    rng2 = np.random.default_rng(5)
    expected = [rng2.choice(VOCAB, p=probs) for _ in range(5)]
    assert got == expected


def test_temperature_zero_deterministic():
    logits = np.random.default_rng(3).normal(size=VOCAB)
    a = sample_next(logits, 0.0, 50, 0.9, 1.0, [1, 2], np.random.default_rng(0))
    b = sample_next(logits, 0.0, 0, 1.0, 2.0, [1, 2], np.random.default_rng(99))
    assert a == b == int(np.argmax(logits))


def test_generate_stops_and_reproducible():
    model = _model()
    enc, dec = _codec()
    a = generate(model, enc, dec, "hi", max_new_tokens=12, temperature=0.0, eos_id=None, seed=1)
    b = generate(model, enc, dec, "hi", max_new_tokens=12, temperature=0.0, eos_id=None, seed=1)
    assert a == b and len(a) == 12
    c = generate(model, enc, dec, "hi", max_new_tokens=12, temperature=1.0, eos_id=None, seed=2)
    assert len(c) == 12


def test_generate_eos_stop():
    model = _model()
    enc, dec = _codec()
    # eos_id = first generated id under greedy: run once, then demand stop.
    first = generate(model, enc, dec, "hi", max_new_tokens=1, temperature=0.0, seed=0)
    stopped = generate(
        model, enc, dec, "hi", max_new_tokens=50, temperature=0.0, eos_id=first[0], seed=0
    )
    assert stopped == first  # stopped immediately at EOS


def test_streamer_prints_deltas_only(capsys):
    """_Streamer emits each character once; a transient trailing U+FFFD
    (token split mid-UTF-8) is held back until it resolves."""
    from nanomlx.generate import _Streamer

    script = {1: "ab", 2: "ab�", 3: "ab🎉c", 4: "ab🎉cd"}
    s = _Streamer(lambda ids: script[len(ids)])
    for n in [1, 2, 3, 4]:
        s.show(list(range(n)))
    out = capsys.readouterr().out
    assert out == "ab🎉cd", repr(out)


def test_streamer_recovery_resyncs(capsys):
    from nanomlx.generate import _Streamer

    s = _Streamer(lambda ids: "zzz" if len(ids) == 1 else "zzz!")
    s.show([0])
    s.show([0, 1])
    assert capsys.readouterr().out == "zzz!"
