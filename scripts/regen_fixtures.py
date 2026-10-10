#!/usr/bin/env python3
"""Regenerate golden fixtures for tokenizer parity tests.

Reads:  model/tests/fixtures/tiny_corpus.txt (committed, fixed)
Writes: model/tests/fixtures/vocab.json + expected_ids.json

Review the diff before committing - fixture changes mean the tokenizer
changed behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model" / "src"))

from nanomlx.reference_bpe import Tokenizer  # noqa: E402

FIX = ROOT / "model" / "tests" / "fixtures"
VOCAB_SIZE = 320

PROBES = [
    "Hello, world!",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    '{"hello": ["world", 123]}',
    "🎉👨‍👩‍👧‍👦🇺🇸",
    "你好世界日本語한국어",
    "नमस्ते दुनिया",
    "مرحبا بالعالم",
    "café Z͑ͫaͫlͨgͤoͭ",
    "   spaces\tand\nnewlines\r\n",
    "",
    "a",
    "don't stop 1234567 $19.99",
    "<|endoftext|> stays text by default",
]


def main() -> None:
    corpus = FIX / "tiny_corpus.txt"
    if not corpus.exists():
        raise SystemExit(f"missing {corpus} - it must be committed, not generated")
    tok = Tokenizer.train([corpus], vocab_size=VOCAB_SIZE)
    tok.save(FIX / "vocab.json")
    expected = {text: tok.encode(text) for text in PROBES}
    (FIX / "expected_ids.json").write_text(
        json.dumps({"vocab_size": VOCAB_SIZE, "probes": expected}, indent=2),
        encoding="utf-8",
    )
    print(f"vocab_size={tok.vocab_size} merges={len(tok.merges)} probes={len(expected)}")
    print(f"wrote {FIX / 'vocab.json'} and {FIX / 'expected_ids.json'}")


if __name__ == "__main__":
    main()
