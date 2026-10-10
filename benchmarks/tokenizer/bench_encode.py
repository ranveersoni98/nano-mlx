#!/usr/bin/env python3
"""Tokenizer encode showdown: ours (Rust) vs HF `tokenizers` vs `tiktoken`.

Fairness (§11.1): same inputs throughout. Vocabs necessarily differ (ours
and HF are both 32k BPE trained on TinyStories - different merges; tiktoken
uses published o200k_base), so ids are NOT compared across implementations;
each implementation's own round-trip is verified and results are reported
separately with the caveat stated. Release-built extension required.

Workloads: (a) 100MB large-file (+ full 1.9GB ours-only), (b) short-string
latency p50/p99, (c) code-heavy, (d) multilingual.
Method: warmup discarded, ≥3 timed repeats, median reported, wall +
tokens/sec + MB/sec + RSS-after. JSON -> benchmarks/results/.

Usage:
    uv run --project benchmarks python benchmarks/tokenizer/bench_encode.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "src"))

import nanomlx_tokenizer as OURS
import psutil
import tiktoken

from nanomlx_bench import write_result

OUR_VOCAB = ROOT / "data" / "tinystories" / "vocab-32k.json"
TRAIN_TXT = ROOT / "data" / "tinystories" / "train.txt"
HF_VOCAB_SIZE = 32000


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def train_hf_if_needed(slice_path: Path):
    """HF 32k BPE on the same training slice. Cached to data/."""
    from tokenizers import Tokenizer as HFTok
    from tokenizers.decoders import ByteLevel as ByteLevelDec
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    cache = ROOT / "data" / "tinystories" / "hf-32k.json"
    if cache.exists():
        return HFTok.from_file(str(cache))
    tok = HFTok(BPE())
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDec()
    trainer = BpeTrainer(
        vocab_size=HF_VOCAB_SIZE,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        show_progress=False,
        # Full 256-byte coverage: without this, bytes unseen in the slice
        # (e.g. emoji) are OOV and silently dropped on encode.
        initial_alphabet=ByteLevel.alphabet(),
    )
    t0 = time.perf_counter()
    tok.train([str(slice_path)], trainer)
    print(f"HF trained 32k in {time.perf_counter() - t0:.1f}s")
    tok.save(str(cache))
    return tok


def load_impls(train_slice: Path):
    ours = OURS.Tokenizer.load(str(OUR_VOCAB))
    hf = train_hf_if_needed(train_slice)
    tik = tiktoken.get_encoding("o200k_base")
    return {"ours-rust": ours, "hf-tokenizers": hf, "tiktoken-o200k": tik}


def encode_all(name: str, impl, texts: list[str]):
    if name == "ours-rust":
        return impl.encode_batch(texts)
    if name == "hf-tokenizers":
        return [e.ids for e in impl.encode_batch(texts)]
    return [impl.encode(t, disallowed_special=()) for t in texts]


def bench_workload(name, impl, texts, repeats, out: dict, notes: str = ""):
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    encode_all(name, impl, texts[:10])  # warmup, discarded
    walls, counts = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ids = encode_all(name, impl, texts)
        walls.append(time.perf_counter() - t0)
        counts.append(sum(len(x) for x in ids))
    wall = statistics.median(walls)
    n_tok = counts[0]
    out[name] = {
        "notes": notes,
        "wall_s": round(wall, 3),
        "tokens": n_tok,
        "tokens_per_s": round(n_tok / wall),
        "mb_per_s": round(n_bytes / wall / 1e6, 2),
        "rss_after_mb": round(rss_mb(), 1),
    }
    # Correctness: same ids across repeats (determinism) + round-trip on
    # probes and on the workload sample itself. (An earlier revision caught
    # HF dropping bytes unseen in training - fixed via initial_alphabet.)
    again = encode_all(name, impl, texts[:5])
    first = encode_all(name, impl, texts[:5])
    assert [list(x) for x in again] == [list(x) for x in first], f"{name} nondeterministic"
    probes = ["Hello, world! 123", "def f(x):\n    return x\n", "café 🎉", " "]
    for p in probes + texts[:1]:
        ids = encode_all(name, impl, [p])[0]
        back = impl.decode(list(ids))
        assert back == p, f"{name} round-trip failed on {p!r:.60}"


def bench_latency(name, impl, texts, out: dict):
    encode_all(name, impl, texts[:10])
    per_doc = []
    for t in texts:
        t0 = time.perf_counter()
        encode_all(name, impl, [t])
        per_doc.append((time.perf_counter() - t0) * 1e6)
    per_doc.sort()
    n = len(per_doc)
    out[name] = {
        "p50_us": round(per_doc[n // 2], 1),
        "p99_us": round(per_doc[int(n * 0.99)], 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="100 docs instead of full sets")
    args = ap.parse_args()
    assert OUR_VOCAB.exists(), f"train it first (see M3): {OUR_VOCAB}"
    assert TRAIN_TXT.exists(), f"missing {TRAIN_TXT}"

    # 20MB slice for HF training (same data ours saw, subset for speed).
    train_slice = ROOT / "data" / "tinystories" / "train_slice_20m.txt"
    if not train_slice.exists():
        with open(TRAIN_TXT, "rb") as f:
            blob = f.read(20_000_000)
        cut = blob.rfind(b"\n")
        train_slice.write_bytes(blob[:cut])
    impls = load_impls(train_slice)

    full = TRAIN_TXT.read_bytes()
    cut = full.find(b"\n", 100_000_000)
    blob100 = full[:cut].decode("utf-8")
    docs100 = blob100.split("\n")
    words = " ".join(docs100[:2000]).split()
    import random

    rng = random.Random(7)
    short = [" ".join(rng.sample(words, 40)) for _ in range(2000 if not args.quick else 100)]
    code = [
        "def f(x):\n    return x * 2\nimport os\n" * 4,
        '{"a": [1, 2, {"b": null}], "c": true}\n' * 6,
        'fn main() {\n    println!("hi");\n}\nlet x: Vec<u8> = vec![];\n' * 4,
    ] * (170 if not args.quick else 10)
    multi = ["🎉 你好 नमस्ते مرحبا café naïve " * 20] * (500 if not args.quick else 20)

    results: dict = {
        "vocabs": {
            "ours-rust": "32k BPE trained on full TinyStories train (this repo)",
            "hf-tokenizers": "32k BPE trained on 20MB TinyStories slice",
            "tiktoken-o200k": "published o200k_base (151k, own merges)",
        },
        "caveat": "Same inputs; different vocabs => ids not cross-compared. "
        "Round-trip verified per implementation.",
        "workloads": {},
    }
    wl = results["workloads"]

    print("== workload: 100MB large-file ==")
    wl["large_file_100mb"] = {}
    for name, impl in impls.items():
        bench_workload(name, impl, [blob100], 3, wl["large_file_100mb"])
        r = wl["large_file_100mb"][name]
        print(f"{name:15} {r['tokens_per_s']:>10,} tok/s  {r['mb_per_s']:>6} MB/s")

    print("== workload: short-string latency (2000 docs) ==")
    wl["short_latency"] = {}
    for name, impl in impls.items():
        bench_latency(name, impl, short, wl["short_latency"])
        r = wl["short_latency"][name]
        print(f"{name:15} p50 {r['p50_us']:>8} us  p99 {r['p99_us']:>8} us")

    print("== workload: code-heavy / multilingual ==")
    for key, docs in [("code_heavy", code), ("multilingual", multi)]:
        wl[key] = {}
        for name, impl in impls.items():
            bench_workload(name, impl, docs, 3, wl[key])
            r = wl[key][name]
            print(f"{key:13} {name:15} {r['tokens_per_s']:>10,} tok/s")

    if not args.quick:
        print("== workload: full 1.9GB file, ours only ==")
        ours = impls["ours-rust"]
        t0 = time.perf_counter()
        stats = ours.encode_file_to_bin(str(TRAIN_TXT), "/tmp/full_bench.bin")
        wall = time.perf_counter() - t0
        n_bytes = TRAIN_TXT.stat().st_size
        wl["full_file_ours_only"] = {
            "tokens": stats["n_tokens"],
            "wall_s": round(wall, 2),
            "tokens_per_s": round(stats["n_tokens"] / wall),
            "mb_per_s": round(n_bytes / wall / 1e6, 2),
            "dtype": stats["dtype"],
        }
        r = wl["full_file_ours_only"]
        print(f"ours-rust full file: {r['tokens_per_s']:,} tok/s  {r['mb_per_s']} MB/s")

    from datetime import date

    path = write_result(f"{date.today().isoformat()}_tokenizer_encode", results)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
