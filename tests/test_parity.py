"""Rust (`nanomlx_tokenizer`) vs Python reference: ids should match.

We check the saved fixtures AND a randomized corpus, zero mismatches.
Run it in the model env (`uv run --project model pytest tests/`).

The 10 MB randomized corpus is the big one. It's marked slow so the
normal run stays quick. Run it directly when you want the full check.
"""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model" / "src"))

from nanomlx.reference_bpe import Tokenizer as RefTokenizer

nanomlx_tokenizer = pytest.importorskip("nanomlx_tokenizer")

FIX = ROOT / "model" / "tests" / "fixtures"

WORD_POOL = [
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
    "tokens",
    "merge",
    "cache",
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
    "sed",
    "do",
    "eiusmod",
    "123",
    "4567",
    "19.99",
    "1000000",
    "don't",
    "can't",
    "won't",
    "it's",
]
CODE_POOL = [
    "def f(x):\n    return x * 2\n",
    '{"a": [1, 2, {"b": null}]}\n',
    'fn main() { println!("hi"); }\n',
    "for i in range(100):\n    x += f(i)\n",
    "   indented   \t spaced\n\n",
]
UNI_POOL = [
    "🎉",
    "👨\u200d👩\u200d👧\u200d👦",
    "你好",
    "नमस्ते",
    "مرحبا",
    "café",
    "naïve",
    "🇺🇸",
]


@pytest.fixture(scope="module")
def pair():
    payload = json.loads((FIX / "vocab.json").read_text(encoding="utf-8"))
    merges = [tuple(m) for m in payload["merges"]]
    specials = payload["special_tokens"]
    ref = RefTokenizer(merges, specials)
    rust = nanomlx_tokenizer.Tokenizer(merges, specials)
    return ref, rust


def test_fixture_probes_identical(pair):
    ref, rust = pair
    expected = json.loads((FIX / "expected_ids.json").read_text(encoding="utf-8"))
    for text, ids in expected["probes"].items():
        assert ref.encode(text) == ids  # python side reproduces the fixture
        assert rust.encode(text) == ids, f"mismatch for {text!r:.60}"
        assert rust.decode(ids) == text


def _random_doc(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(1, 12)):
        r = rng.random()
        if r < 0.55:
            parts.append(rng.choice(WORD_POOL))
        elif r < 0.7:
            parts.append(rng.choice(CODE_POOL))
        elif r < 0.85:
            parts.append(rng.choice(UNI_POOL))
        else:
            parts.append(" " * rng.randint(1, 8))
        if rng.random() < 0.25:
            parts.append(rng.choice(["\n", "\n\n", "\r\n", "\t", ". ", "! ", "... "]))
    return "".join(parts)


def _check_corpus(pair, n_docs: int, seed: int):
    ref, rust = pair
    rng = random.Random(seed)
    mismatches = 0
    total_ids = 0
    for _ in range(n_docs):
        doc = _random_doc(rng)
        a, b = ref.encode(doc), rust.encode(doc)
        total_ids += len(a)
        if a != b:
            mismatches += 1
            print(f"MISMATCH doc={doc!r:.120}\n ref={a[:20]}\n rust={b[:20]}")
        assert ref.decode(a) == doc
        assert rust.decode(b) == doc
    assert mismatches == 0
    return total_ids


def test_randomized_corpus_fast(pair):
    total = _check_corpus(pair, n_docs=1500, seed=1337)
    assert total > 10_000


def test_long_inputs_parity(pair):
    """Long-input chopping (>64K chars) has to agree exactly, 1MB runs too."""
    ref, rust = pair
    docs = [
        "a" * 200_000,
        " " * 100_000,
        "abあ🎉 " * 20_000,  # multibyte, crosses split points
        "word " * 50_000 + "\n" + "x" * 100_000,
    ]
    for doc in docs:
        assert rust.encode(doc) == ref.encode(doc)
        assert rust.decode(ref.encode(doc)) == doc


@pytest.mark.slow
def test_randomized_corpus_10mb_gate(pair):
    """Big check: 10 MB-plus of mixed text, zero mismatches."""
    ref, rust = pair
    rng = random.Random(20261010)
    n_bytes = 0
    n_docs = 0
    while n_bytes < 10 * 1024 * 1024:
        doc = _random_doc(rng)
        assert ref.encode(doc) == rust.encode(doc)
        n_bytes += len(doc.encode("utf-8"))
        n_docs += 1
    assert n_bytes >= 10 * 1024 * 1024
    print(f"\n10MB check: {n_docs} docs, {n_bytes} bytes, no mismatches")


def test_batch_matches_single(pair):
    ref, rust = pair
    rng = random.Random(99)
    texts = [_random_doc(rng) for _ in range(200)]
    assert rust.encode_batch(texts) == [rust.encode(t) for t in texts]
    assert rust.encode_batch(texts) == ref.encode_batch(texts)


def test_save_load_interop(pair, tmp_path):
    """A vocab saved by one implementation loads in the other."""
    ref, rust = pair
    path = tmp_path / "interop.json"
    ref.save(path)
    rust2 = nanomlx_tokenizer.Tokenizer.load(str(path))
    path2 = tmp_path / "interop2.json"
    rust.save(str(path2))
    from nanomlx.reference_bpe import Tokenizer as Ref2

    ref2 = Ref2.load(path2)
    probe = "hello 🎉 world 123"
    assert rust2.encode(probe) == ref.encode(probe) == ref2.encode(probe)
