"""Rust training output should match the Python version exactly.

We train both on a fixed 1MB-plus mixed corpus (with CRLF in it, since
newline translation bit us once) and check the merge lists are identical.
Slow on purpose (the Python side recounts pairs every merge). Run it
directly when you want the full check.
"""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model" / "src"))

from nanomlx.reference_bpe import Tokenizer as RefTokenizer

nanomlx_tokenizer = pytest.importorskip("nanomlx_tokenizer")

VOCAB_SIZE = 320
SEED = 20261010


def make_corpus_mb1(tmp_path: Path) -> Path:
    """Make a fixed mixed corpus, 1MB-plus, with raw newlines kept.

    Written in binary so \\r\\n stays as-is (both sides train byte-exact).
    """
    rng = random.Random(SEED)
    words = [
        "hello",
        "world",
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "and",
        "with",
        "string",
        "theory",
        "tokens",
        "merge",
        "cache",
        "encode",
        "decode",
        "train",
        "model",
        "data",
        "batch",
        "attention",
        "norm",
        "def",
        "return",
        "import",
        "fn",
        "let",
        "mut",
        "struct",
        "impl",
        "for",
        "while",
        "if",
        "else",
        "print",
    ]
    code = [
        "def f(x):\n    return x * 2\n",
        '{"a": [1, 2, {"b": null}]}\n',
        'fn main() { println!("hi {}"); }\n',
        "for i in range(100):\n    x += f(i)\n",
    ]
    uni = [
        "🎉",
        "👨\u200d👩\u200d👧\u200d👦",
        "你好世界",
        "नमस्ते",
        "مرحبا",
        "café",
        "€",
        "₹",
    ]
    parts: list[str] = []
    n_bytes = 0
    while n_bytes < 1_100_000:
        r = rng.random()
        if r < 0.5:
            s = " ".join(rng.sample(words, rng.randint(5, 30)))
        elif r < 0.65:
            s = rng.choice(code) * rng.randint(1, 3)
        elif r < 0.8:
            s = "".join(rng.sample(uni, rng.randint(1, 4)))
        else:
            s = " " * rng.randint(1, 10)
        if rng.random() < 0.3:
            s += rng.choice(["\n", "\n\n", "\r\n", "\t", ". ", "! "])
        parts.append(s)
        n_bytes += len(s.encode("utf-8"))
    path = tmp_path / "parity_corpus.txt"
    path.write_bytes("".join(parts).encode("utf-8"))
    assert path.stat().st_size >= 1_000_000
    return path


@pytest.mark.slow
def test_training_parity_1mb(tmp_path):
    corpus = make_corpus_mb1(tmp_path)
    ref = RefTokenizer.train([corpus], vocab_size=VOCAB_SIZE)
    rust = nanomlx_tokenizer.Tokenizer.train([str(corpus)], VOCAB_SIZE, None)
    rust_merges = [tuple(m) for m in rust.merges()]
    assert rust_merges == ref.merges, "merge lists differ: " + str(
        [(i, a, b) for i, (a, b) in enumerate(zip(ref.merges, rust_merges)) if a != b][
            :5
        ]
    )
    assert rust.special_tokens() == ref.special_tokens
    assert len(ref.merges) > 0
    print(f"\n1MB training check: {len(ref.merges)} merges, no diffs")


def test_training_parity_small(tmp_path):
    """Quicker training check on the fixture corpus (runs by default)."""
    corpus = ROOT / "model" / "tests" / "fixtures" / "tiny_corpus.txt"
    ref = RefTokenizer.train([corpus], vocab_size=VOCAB_SIZE)
    rust = nanomlx_tokenizer.Tokenizer.train([str(corpus)], VOCAB_SIZE, None)
    assert [tuple(m) for m in rust.merges()] == ref.merges
    assert rust.special_tokens() == ref.special_tokens
