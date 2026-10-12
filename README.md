# Nano-MLX

<p align="center">
  <a href="https://support.apple.com/en-ca/116943"><img src="https://img.shields.io/badge/platform-Apple%20silicon-blue" alt="Apple silicon" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" /></a>
</p>

A GPT built from scratch, only runs on Apple silicon. No CUDA, no cloud GPUs, no shortcuts. Just a byte-level BPE tokenizer in Rust, a decoder-only transformer in MLX, and benchmarks to keep us honest.

## What lives here

1. **Tokenizer (Rust)** - byte-level BPE. 256 base bytes so anything round-trips, o200k-style regex splitting, 32k vocab by default, `<|endoftext|>` / `<|pad|>` specials. Fast bits: large inputs encode their 64K pre-tokenize pieces on the rayon pool (output identical at any thread count), per-chunk cache, linked-list plus heap merge loop with fast lanes for single bytes / tiny chunks / whole-chunk hits, `memchr`-based special scanning, parallel batch encode plus batch decode, parallel word counting in training (20MB slice to 5k vocab in ~1s, at parity with HF).
2. **Model (MLX)** - decoder-only transformer with RoPE, RMSNorm, SwiGLU, multi-head attention plus optional GQA, tied weights, no biases. AdamW with warmup and cosine decay, grad clipping, `mx.compile`, KV cache we wrote by hand, temperature / top-k / top-p sampling. Sizes from debug (under 1M params) to base (around 125M).
3. **Benchmarks** - our tokenizer vs `tiktoken` and HF `tokenizers`, plus model speed runs (hand-written vs `mx.fast`, compile on/off, cache on/off, fp32 vs bf16).

## Repo layout

```
nanoMLX/
  README.md / LICENSE
  tokenizer/             # Rust workspace: tokenizer-core + tokenizer-py (PyO3)
  model/                 # Python uv project: MLX model, data, training, inference
  benchmarks/            # Python uv project: bench scripts, results/*.json, charts/
  scripts/               # fixture regen, dataset download
  tests/                 # Rust vs Python cross-checks
  data/                  # gitignored: raw text, .bin files, checkpoints
```

## Quickstart

You need Rust stable, [`uv`](https://docs.astral.sh/uv/), and a Mac with Apple silicon.

```bash
git clone <repo> && cd nanoMLX
cargo test --workspace --manifest-path tokenizer/Cargo.toml   # 27 + 4 pass
uv sync --project model --group dev && uv sync --project benchmarks --group dev
uv run --project model maturin develop --release -m tokenizer/tokenizer-py/Cargo.toml
uv run --project model pytest model/tests/ tests/            # 70+ pass
# data + 100-step proof train + generate:
uv run --project model python scripts/download_tinystories.py
# train the 32k vocab with the Rust tokenizer (a few minutes on the full
# TinyStories text), then encode everything to .bin files:
uv run --project model python -c "import nanomlx_tokenizer as T; tok = T.Tokenizer.train(['data/tinystories/train.txt'], 32000, None); tok.save('data/tinystories/vocab-32k.json')"
uv run --project model python -c "from nanomlx.data import build_bins; build_bins('data/tinystories/vocab-32k.json', 'data/tinystories/train.txt', 'data/tinystories/validation.txt', 'data/tinystories/bins')"
uv run --project model python -m nanomlx.train --preset tiny --bin-dir data/tinystories/bins --ckpt-dir data/ckpts/proof --vocab data/tinystories/vocab-32k.json --steps 100 --batch 8
uv run --project model python -m nanomlx.generate --ckpt data/ckpts/proof --vocab data/tinystories/vocab-32k.json --prompt "Once upon a time" --max-new-tokens 120
```

Two things to know first:
- `uv sync` wipes the venv, so run `maturin develop` again after every sync.
- Always bench `--release` Rust builds. Debug builds are 10 to 100x slower and the numbers will lie to you.

Data flow is: `download TinyStories, encode to .bin files, load with memmap, train, generate`.

## Benchmarks

<!-- Made from benchmarks/results/*.json by benchmarks/tokenizer/make_readme_table.py, don't edit by hand. -->
_Machine: Apple M5, 23.3GB RAM, Python 3.14.8, rustc 1.99.0 (b940084d7 2026-09-28) (Homebrew). Release build. Median of repeats, warmup thrown away._

Tokenizer encode (same inputs, different vocabs, so ids aren't compared):

| Workload | ours (Rust) | HF tokenizers | tiktoken o200k |
|---|---|---|---|
| 100MB file | 3,580,190 tok/s (15.61 MB/s) | 839,038 tok/s (3.46 MB/s) | 12,516,930 tok/s (53.48 MB/s) |
| short-string latency p50/p99 | 13.8/24.0 µs | 30.8/47.1 µs | 4.2/7.0 µs |
| code-heavy | 21,862,668 tok/s | 9,386,520 tok/s | 10,340,038 tok/s |
| multilingual | 88,860,092 tok/s | 20,044,298 tok/s | 8,655,806 tok/s |
| full 1.9GB file (ours only) | 10,007,244 tok/s (43.59 MB/s) | - | - |

Tokenizer training (same 20MB slice, vocab 5000, tiktoken has no trainer):

| Implementation | Wall time |
|---|---|
| ours (Rust) | 1.31s (4742 merges) |
| HF tokenizers | 0.94s |

_Heads up: large inputs encode their 64K pre-tokenize pieces in parallel (same pieces either way, so output is identical at any thread count). tiktoken still leads big-file encode - its regex engine stays linear where our backtracking one slows down - but the gap is much smaller than it was._

Model speed (tiny preset, 32k vocab, random init, speed doesn't depend on weights):

| Train step (batch 4, ctx 256) | tok/s | Peak mem |
|---|---|---|
| manual attention, fp32, compiled | 11,173 | 1359.2 MB |
| mx.fast SDPA, fp32, compiled | 8,335 | 1346.2 MB |
| mx.fast, fp32, no compile | 7,662 | - |
| mx.fast, bf16, compiled | 10,603 | 1003.2 MB |

| Batch sweep (fast, fp32) | tok/s |
|---|---|
| batch 1 | 3,733 |
| batch 2 | 5,589 |
| batch 4 | 7,851 |
| batch 8 | 8,731 |

| Inference (128-token prompt, 64 new) | prefill tok/s | decode tok/s |
|---|---|---|
| manual + KV cache | 25,452 | 226 |
| manual, re-forward | 13,769 | 102 |
| mx.fast + KV cache | 39,127 | 503 |
| mx.fast, re-forward | 27,206 | 160 |

(Model table made from `benchmarks/results/2026-10-10_model.json` with the same script.)

## Samples

`--prompt "Once upon a time" --temperature 0.8 --top-k 50 --top-p 0.95 --seed 0`:
> Once upon a time, there was a little boy named Timmy. Every day, Sam and Lily were scared to play with a big box. One day, Lily's mom came to the park with a big dog with her little boy. Lily was very proud of her. Lily felt happy that her mom said, "I want to play with you. We wanted to go to the grass and share the room."

`--prompt "Tom was sad because" --temperature 0.7 --seed 7`:
> Tom was sad because she liked to play with her. One day, she saw a big dog named Sue. Lily liked to play with her and play with her toys. She said, "We are very very brave. It will play with you."

`--prompt "One sunny morning" --temperature 0.9 --top-k 40 --top-p 0.9 --seed 21`:
> One sunny morning, Lily would be scared because her mom gave her a special piece of people.

They start reading like real stories after ~300 steps. Train longer and they get better. Command is under **Train a model** below.

## Train a model

```bash
uv run --project model python -m nanomlx.train --preset small --bin-dir data/tinystories/bins --ckpt-dir data/ckpts/small --vocab data/tinystories/vocab-32k.json --steps 20000 --batch 16 --seed 0
# continue later from any checkpoint:
uv run --project model python -m nanomlx.train --preset small --bin-dir data/tinystories/bins --ckpt-dir data/ckpts/small --vocab data/tinystories/vocab-32k.json --steps 20000 --batch 16 --seed 0 --resume data/ckpts/small/step_002000
```

Presets: `debug` (sanity checks), `tiny` (around 10M params, quick tries), `small` (around 30M, the real TinyStories run), `base` (around 125M, only if you've got the RAM). Checkpoints keep the newest 3 plus the best-val weights, always with the full config and tokenizer hash.


## Known limitations

- tiktoken still beats us on big-file encode (its regex engine stays linear, ours backtracks). We win code-heavy, multilingual, and short-latency-vs-HF.
- `fancy-regex` has a 1M backtrack cap, so we chop at 64K chars (tests cover it, both implementations agree exactly).
- Short-string latency floor is ~10µs (regex + PyO3 overhead per call); batch your docs.
- Millisecond prefill timings wobble on laptops (worse on battery - sustained benches throttle). We report medians plus the relative takeaways.
- `meta.json` / `tokenizer_sha256` is FNV-1a for now. We'll swap in a real hash later.

## License

Code is MIT - see [LICENSE](LICENSE).

<p align="center">
  Built with <3 and a lot of bad decisions! by <a href="https://ranveersoni.me">Ranveer Soni</a>
</p>
