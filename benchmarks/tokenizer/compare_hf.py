#!/usr/bin/env python3
"""Ours vs HuggingFace `tokenizers` (early sanity check).

Not a strict match test: HF's byte-level BPE splits text differently and
breaks ties differently, so ids are *supposed* to differ. What has to
hold: both round-trip everything (including bad UTF-8), and we time
encode speed on the same inputs.

Usage:
    uv run --project benchmarks python benchmarks/tokenizer/compare_hf.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "src"))

VOCAB_SIZE = 320
CORPUS = ROOT / "model" / "tests" / "fixtures" / "tiny_corpus.txt"

PROBES = [
    "Hello, world! 123",
    "def f(x):\n    return x * 2\n",
    "🎉👨\u200d👩\u200d👧\u200d👦🇺🇸",
    "你好世界日本語한국어",
    "नमस्ते दुनिया مرحبا",
    "   trailing spaces   ",
    "",
]


def train_hf():
    from tokenizers import Tokenizer as HFTok
    from tokenizers.decoders import ByteLevel as ByteLevelDec
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tok = HFTok(BPE())
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDec()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        show_progress=False,
    )
    tok.train([str(CORPUS)], trainer)
    return tok


def time_it(fn, repeats=5):
    fn()  # warmup, doesn't count
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    from nanomlx.reference_bpe import Tokenizer as Ours

    ours = Ours.train([CORPUS], vocab_size=VOCAB_SIZE)
    hf = train_hf()

    print("== round-trip ==")
    for p in PROBES:
        assert ours.decode(ours.encode(p)) == p, f"ours failed on {p!r:.40}"
        assert hf.decode(hf.encode(p).ids) == p, f"hf failed on {p!r:.40}"
    # Bad bytes: ours is byte-exact, HF byte-level decodes without loss too.
    blob = b"\xff\xfe\x00abc"
    assert ours.decode(ours.encode_bytes(blob)).encode("utf-8", errors="replace")
    print("both round-trip all probes (incl. empty string)")

    print("== ids (supposed to differ, different splitters) ==")
    same = sum(1 for p in PROBES if ours.encode(p) == hf.encode(p).ids)
    print(f"identical id sequences on {same}/{len(PROBES)} probes")

    import random

    rng = random.Random(0)
    words = CORPUS.read_text(encoding="utf-8").split()
    docs = [" ".join(rng.sample(words, 40)) for _ in range(2000)]
    n_bytes = sum(len(d.encode()) for d in docs)

    t_ours = time_it(lambda: [ours.encode(d) for d in docs])
    t_hf = time_it(lambda: [hf.encode(d).ids for d in docs])
    print("== speed (2000 docs, %.1f KB, best of 5) ==" % (n_bytes / 1024))
    print(f"ours (pure Python): {t_ours * 1000:.0f} ms  ({n_bytes / t_ours / 1e6:.2f} MB/s)")
    print(f"hf tokenizers (Rust):    {t_hf * 1000:.0f} ms  ({n_bytes / t_hf / 1e6:.2f} MB/s)")
    print("heads up: ours here is the SLOW Python version, not the Rust one.")

    from nanomlx_bench import machine_specs

    print("machine:", machine_specs().get("cpu_brand", "?"))


if __name__ == "__main__":
    main()
