# Nano-MLX

<p align="center">
  <a href="https://support.apple.com/en-ca/116943"><img src="https://img.shields.io/badge/platform-Apple%20silicon-blue" alt="Apple silicon" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" /></a>
</p>

A GPT built entirely from scratch, running only on Apple silicon. No CUDA, no cloud GPUs, no shortcuts: a byte-level BPE tokenizer written in Rust, a decoder-only transformer written in MLX, and benchmarks!

## What lives here

1. **Tokenizer (Rust)** - byte-level BPE: 256 base bytes (no unknown tokens, any bytes round-trip), o200k-style regex pre-splitting, 32k default vocab, `<|endoftext|>` / `<|pad|>` special tokens. Performance work: per-chunk cache, linked-list plus binary-heap merge loop, `rayon` batch encoding, incremental pair counting in training.
2. **Model (MLX)** - decoder-only transformer: RoPE, RMSNorm, SwiGLU, multi-head attention with optional GQA, weight tying, no biases. AdamW with warmup plus cosine decay, gradient clipping, `mx.compile`, hand-written KV cache, temperature / top-k / top-p sampling. Presets from debug (under 1M parameters) to base (about 125M).
3. **Benchmarks** - our tokenizer against `tiktoken` and HuggingFace `tokenizers`, plus model throughput ablations (manual vs `mx.fast`, compile on/off, cache on/off, fp32 vs bf16).

## Repo layout

```
nanoMLX/
  README.md / LICENSE
  tokenizer/             # Rust workspace: tokenizer-core (pure) + tokenizer-py (PyO3)
  model/                 # Python uv project: MLX model, data, training, inference
  benchmarks/            # Python uv project: harnesses, results/*.json, charts/
  scripts/               # fixture regen, dataset download
  tests/                 # cross-implementation parity (Rust vs Python)
  data/                  # gitignored: raw text, .bin files, checkpoints
```

## Quickstart

You need Rust stable, [`uv`](https://docs.astral.sh/uv/), and a Mac with Apple silicon.

```bash
git clone <repo> && cd nanoMLX
cargo test --workspace --manifest-path tokenizer/Cargo.toml   # 22 + 4 pass
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

Two things worth knowing upfront:
- `uv sync` resets the virtualenv, so re-run `maturin develop` after any sync.
- Always benchmark `--release` Rust builds. Debug builds are 10 to 100 times slower and will mislead you.

Data flow: `download TinyStories -> encode_file_to_bin -> .bin (np.memmap) -> (x, y = x shifted) batches -> train -> generate`.

## Benchmarks

<!-- GENERATED from benchmarks/results/*.json by benchmarks/tokenizer/make_readme_table.py - do not hand-edit. -->
_Machine: Apple M5, 23.2GB RAM, Python 3.14.8, rustc 1.99.0 (b940084d7 2026-09-28) (Homebrew). Release build. Median of repeats; warmup discarded._

Tokenizer encode (same inputs; different vocabs - ids not cross-compared):

| Workload | ours (Rust) | HF tokenizers | tiktoken o200k |
|---|---|---|---|
| 100MB file | 1,834,746 tok/s (8.0 MB/s) | 454,892 tok/s (1.88 MB/s) | 6,554,700 tok/s (28.0 MB/s) |
| short-string latency p50/p99 | 26.5/59.1 µs | 61.7/147.6 µs | 8.8/35.7 µs |
| code-heavy | 16,342,003 tok/s | 6,827,342 tok/s | 5,509,937 tok/s |
| multilingual | 62,504,411 tok/s | 13,587,341 tok/s | 3,785,801 tok/s |
| full 1.9GB file (ours only) | 5,940,192 tok/s (25.87 MB/s) | - | - |

Tokenizer training (same 20MB slice, vocab 5000; tiktoken ships no trainer):

| Implementation | Wall time |
|---|---|
| ours (Rust incremental) | 2.5s (4742 merges) |
| HF tokenizers | 1.7s |

_Caveats: single-doc inputs use one thread in ours (file and batch paths parallelize); tiktoken wins large-file encode because its DFA regex stays linear where our backtracking one slows down._

Model throughput (tiny preset, 32k vocab, random init - throughput is weight-independent):

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

(Model table generated from `benchmarks/results/2026-10-10_model.json` - same generator.)

## Samples

`--prompt "Once upon a time" --temperature 0.8 --top-k 50 --top-p 0.95 --seed 0`:
> Once upon a time, there was a little boy named Timmy. Every day, Sam and Lily were scared to play with a big box. One day, Lily's mom came to the park with a big dog with her little boy. Lily was very proud of her. Lily felt happy that her mom said, "I want to play with you. We wanted to go to the grass and share the room."

`--prompt "Tom was sad because" --temperature 0.7 --seed 7`:
> Tom was sad because she liked to play with her. One day, she saw a big dog named Sue. Lily liked to play with her and play with her toys. She said, "We are very very brave. It will play with you."

`--prompt "One sunny morning" --temperature 0.9 --top-k 40 --top-p 0.9 --seed 21`:
> One sunny morning, Lily would be scared because her mom gave her a special piece of people.

They read like real stories after 300 steps. Train longer and they get better - the command is in **Train a model** below.

## Train a model

```bash
uv run --project model python -m nanomlx.train --preset small --bin-dir data/tinystories/bins --ckpt-dir data/ckpts/small --vocab data/tinystories/vocab-32k.json --steps 20000 --batch 16 --seed 0
# continue later from any checkpoint:
uv run --project model python -m nanomlx.train --preset small --bin-dir data/tinystories/bins --ckpt-dir data/ckpts/small --vocab data/tinystories/vocab-32k.json --steps 20000 --batch 16 --seed 0 --resume data/ckpts/small/step_002000
```

Presets: `debug` (sanity checks), `tiny` (about 10M parameters, quick experiments), `small` (about 30M, the real TinyStories run), `base` (about 125M, only if your RAM allows). Checkpoints keep the newest 3 plus best-val weights, always with the full config and the tokenizer hash.


## Known limitations

- Single-doc `encode()` uses one thread (file/batch paths parallelize with rayon).
- tiktoken beats us on large-file encode (DFA regex vs backtracking `fancy-regex`); we win cached/batch workloads.
- `fancy-regex` 1M backtrack cap leads to 64K-char force-splits (covered by parity tests).
- Few-ms prefill timings wobble on laptops; medians plus relative findings reported.
- `meta.json`/`tokenizer_sha256` uses FNV-1a for now; switching to a real hash later.

## License

Code is MIT - see [LICENSE](LICENSE).

<p align="center">
  Built with <3 and a lot of bad decisions! by <a href="https://ranveersoni.me">Ranveer Soni</a>
</p>
