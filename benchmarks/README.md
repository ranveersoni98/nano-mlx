# nanomlx-benchmarks

Benchmark scripts that save the machine info with every run. Root README has the tables these make.

## Layout

```
tokenizer/
  compare_hf.py        # early sanity check (ours vs HF)
  bench_encode.py      # big-file / latency / code / multilingual timings
  bench_train.py       # training time: ours vs HF (tiktoken has no trainer)
  make_readme_table.py # builds README tables from results/*.json (don't type them by hand)
model/
  bench_model.py       # train tok/s, prefill/decode, speed comparisons (needs mlx, run with model env)
  bench_metal.py       # our Metal RMSNorm vs built-ins
  plot_loss.py         # loss curves from metrics.jsonl
results/               # checked-in JSON, one file per run (machine info inside)
charts/                # made PNGs (loss curves etc.)
```

## How we bench

- Same inputs everywhere. Warmup doesn't count, median of repeats, `mx.eval` inside the timed bit.
- Release Rust builds only. We note power mode and battery every run.
- If two setups can't share a vocab, we say so and report them separately.
- Tables come from JSON via script, never typed by hand. Charts come from scripts, never edited.

## Commands (run from repo root)

```bash
uv sync --project benchmarks --group dev
uv run --project benchmarks maturin develop --release -m tokenizer/tokenizer-py/Cargo.toml
uv run --project benchmarks python benchmarks/tokenizer/bench_encode.py
uv run --project benchmarks python benchmarks/tokenizer/bench_train.py
uv run --project model python benchmarks/model/bench_model.py        # needs mlx
uv run --project model python benchmarks/model/bench_metal.py        # needs mlx
uv run --project benchmarks python benchmarks/tokenizer/make_readme_table.py
```
