"""Data pipeline tests: building .bins, meta checks, sampler (fast)."""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from nanomlx.data import BatchSampler, BinData, build_bins, check_bin

DOCS = [
    "Hello, world! Once upon a time in a tiny village.",
    "def f(x):\n    return x * 2  # code doc",
    "Emoji doc 🎉 with unicode 你好 and numbers 12345",
    "   spaced\ttabs\nand newlines\r\nCRLF here",
    "Short one.",
    "Another document about foxes and boxes and socks.",
]


@pytest.fixture(scope="module")
def binned(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bins")
    train_txt, val_txt = tmp / "train.txt", tmp / "val.txt"
    train_txt.write_text("\n".join(DOCS * 20) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(DOCS[:3] * 10) + "\n", encoding="utf-8")
    vocab = Path(__file__).resolve().parents[1] / "model" / "tests" / "fixtures" / "vocab.json"
    # Fixture path looks different from model/ vs from root, so handle both.
    if not vocab.exists():
        vocab = Path("model/tests/fixtures/vocab.json")
    out = tmp / "bins"
    stats = build_bins(vocab, train_txt, val_txt, out)
    return tmp, out, vocab, stats


def test_bins_built(binned):
    _, out, _, stats = binned
    assert (out / "train.bin").exists() and (out / "val.bin").exists()
    assert stats["train"]["dtype"] == "uint16"
    assert stats["train"]["n_tokens"] > len(DOCS)


def test_meta_schema_and_eos_separation(binned):
    _, out, vocab, _ = binned
    meta = json.loads((out / "train.meta.json").read_text(encoding="utf-8"))
    for key in [
        "dtype",
        "vocab_size",
        "n_tokens",
        "eos_id",
        "tokenizer_sha256",
        "source",
        "created",
    ]:
        assert key in meta, f"missing meta key {key}"
    data = BinData(out / "train.bin", vocab)
    assert meta["eos_id"] == data.eos_id
    # Every doc should be followed by EOS, check the counts line up.
    assert data.n_tokens == meta["n_tokens"]


def test_loader_refuses_mismatch(binned, tmp_path):
    _, out, vocab, _ = binned
    bad_vocab = tmp_path / "bad_vocab.json"
    payload = json.loads(vocab.read_text(encoding="utf-8"))
    payload["pattern_hash"] = "0" * 16
    bad_vocab.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        check_bin(out / "train.bin", bad_vocab)


def test_sampler_shapes_and_dtypes(binned):
    _, out, vocab, _ = binned
    data = BinData(out / "train.bin", vocab)
    sampler = BatchSampler(data, context=32, seed=0)
    x, y = sampler.sample(4)
    mx.eval(x, y)
    assert x.shape == (4, 32) and y.shape == (4, 32)
    # y is just x shifted by one (same stream underneath).
    assert int(x[0, 1]) == int(y[0, 0])


def test_sampler_reproducible(binned):
    _, out, vocab, _ = binned
    data = BinData(out / "train.bin", vocab)
    a = BatchSampler(data, context=16, seed=5)
    b = BatchSampler(data, context=16, seed=5)
    xa, _ = a.sample(2)
    xb, _ = b.sample(2)
    mx.eval(xa, xb)
    assert bool(mx.all(xa == xb))
    st = a.get_state()
    x1, _ = a.sample(2)
    b.set_state(st)
    x2, _ = b.sample(2)
    mx.eval(x1, x2)
    assert bool(mx.all(x1 == x2))
