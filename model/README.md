# nanomlx-model

The Python half of nanoMLX: MLX transformer, data loading, training loop, and generation. Root README has the full story.

## Layout

```
src/nanomlx/
  reference_bpe.py  # Python BPE. The Rust side has to match this.
  config.py         # ModelConfig / TrainConfig plus debug/tiny/small/base presets
  model.py          # GPT: hand-written attention, GQA, RoPE, RMSNorm, SwiGLU (plus mx.fast flags)
  metal.py          # our optional fused-RMSNorm Metal kernel
  data.py           # .bin building, memmap loading, batch sampling
  train.py          # python -m nanomlx.train
  generate.py       # python -m nanomlx.generate
  utils.py          # seeding, LR schedule, optimizer groups, checkpoints, logging
tests/              # tests per file (fast by default, -m slow for the big ones)
fixtures sit under tests/fixtures/ (tiny corpus plus saved vocab and expected ids)
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
