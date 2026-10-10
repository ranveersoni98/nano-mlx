#!/usr/bin/env python3
"""How long does training take: ours (Rust) vs HF `tokenizers` BPE.

Same 20MB TinyStories slice, same target vocab (5000 keeps HF's time
sane). tiktoken doesn't ship a trainer so there's nothing to compare
there. Saves JSON.

Usage:
    uv run --project benchmarks python benchmarks/tokenizer/bench_train.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "src"))

import nanomlx_tokenizer as OURS

from nanomlx_bench import write_result

SLICE = ROOT / "data" / "tinystories" / "train_slice_20m.txt"
VOCAB = 5000


def main() -> None:
    assert SLICE.exists(), f"run bench_encode.py first ({SLICE})"
    n_bytes = SLICE.stat().st_size

    t0 = time.perf_counter()
    ours = OURS.Tokenizer.train([str(SLICE)], VOCAB, None)
    t_ours = time.perf_counter() - t0

    from tokenizers import Tokenizer as HFTok
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    hf = HFTok(BPE())
    hf.pre_tokenizer = ByteLevel(add_prefix_space=False)
    t0 = time.perf_counter()
    hf.train(
        [str(SLICE)],
        BpeTrainer(
            vocab_size=VOCAB,
            special_tokens=["<|endoftext|>", "<|pad|>"],
            show_progress=False,
            initial_alphabet=ByteLevel.alphabet(),
        ),
    )
    t_hf = time.perf_counter() - t0

    print(f"ours (Rust incremental): {t_ours:.1f}s for {len(ours.merges())} merges")
    print(f"hf (Rust BPE trainer):   {t_hf:.1f}s")
    print("tiktoken: N/A (ships no trainer)")

    from datetime import date

    path = write_result(
        f"{date.today().isoformat()}_tokenizer_train",
        {
            "corpus": {"file": str(SLICE), "bytes": n_bytes, "vocab_size": VOCAB},
            "caveat": "Same slice and size, different pretokenizers/merges. "
            "tiktoken doesn't ship a trainer.",
            "ours_rust_s": round(t_ours, 2),
            "ours_merges": len(ours.merges()),
            "hf_tokenizers_s": round(t_hf, 2),
        },
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
