# nanomlx-model

The Python side of nanoMLX: MLX transformer, data pipeline, training loop, and generation. See the [root README](../README.md) for the full story.

## Layout

```
src/nanomlx/
  reference_bpe.py  # pure-Python BPE - the source of truth for the Rust core
  config.py         # ModelConfig / TrainConfig + debug/tiny/small/base presets
  model.py          # GPT: manual attention, GQA, RoPE, RMSNorm, SwiGLU (+mx.fast flags)
  metal.py          # optional custom fused-RMSNorm Metal kernel
  data.py           # .bin building, memmap loader, batch sampler
  train.py          # python -m nanomlx.train
  generate.py       # python -m nanomlx.generate
  utils.py          # seeding, LR schedule, optimizer groups, checkpoints, logging
tests/              # per-module tests (fast by default; -m slow for the big gates)
fixtures live under tests/fixtures/ (tiny corpus + golden vocab + expected ids)
```

## Common commands (run from repo root)

```bash
uv sync --project model --group dev
uv run --project model maturin develop --release -m tokenizer/tokenizer-py/Cargo.toml  # after every sync!
uv run --project model pytest model/tests/ tests/            # everything, incl. parity
uv run --project model pytest model/tests/ -m "not slow"     # quick pass
uv run --project model ruff check model/ tests/ && uv run --project model ruff format --check model/ tests/
```

## Train / generate

```bash
uv run --project model python -m nanomlx.train --preset small --bin-dir data/tinystories/bins --ckpt-dir data/ckpts/small --vocab data/tinystories/vocab-32k.json --steps 20000 --batch 16 --seed 0
uv run --project model python -m nanomlx.generate --ckpt data/ckpts/small --vocab data/tinystories/vocab-32k.json --prompt "Once upon a time" --max-new-tokens 200
```
