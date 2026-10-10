# nanomlx-benchmarks

Honest harnesses, machine specs attached to every run. See the [root README](../README.md) for the tables these produce.

## Layout

```
tokenizer/
  compare_hf.py        # first-look correctness sketch (ours vs HF)
  bench_encode.py      # large-file / latency / code / multilingual showdown
  bench_train.py       # training wall-time: ours vs HF (tiktoken ships no trainer)
  make_readme_table.py # renders README tables from results/*.json (never hand-type!)
model/
  bench_model.py       # train tok/s, prefill/decode, ablations (needs mlx: run via model env)
  bench_metal.py       # custom Metal RMSNorm vs built-ins
  plot_loss.py         # loss curves from metrics.jsonl
results/               # committed JSON, one file per run (machine specs inside)
charts/                # generated PNGs referenced by the README
```

## Rules we hold to

- Same inputs everywhere; warmups discarded; median of repeats; `mx.eval` inside timed regions.
- Release Rust builds only. Note power mode + battery in every run.
- If implementations can't share a vocab, say so and report separately.
- Tables are generated from JSON, never hand-typed. Charts from scripts, never edited.

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
