#!/usr/bin/env python3
"""Model speed: train tok/s, prefill/decode tok/s, peak memory.

We change one thing at a time: hand-written vs mx.fast, compile on/off,
cache vs no cache, fp32 vs bf16, batch size sweep. Warmup doesn't count,
mx.eval sits inside the timed bit, we take the median. Tiny preset with
random weights (speed doesn't care about the actual weights). JSON goes
in benchmarks/results/.

Needs mlx, so run it with the MODEL env:
    uv run --project model python benchmarks/model/bench_model.py [--quick]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "src"))

import mlx.core as mx
from mlx import nn
from nanomlx.config import preset
from nanomlx.model import GPT, KVCache

from nanomlx_bench import write_result

VOCAB = 32000
CTX = 256


def _model(fast: bool, dtype, batch_note: str = ""):
    mx.random.seed(0)
    cfg = preset(
        "tiny",
        vocab_size=VOCAB,
        attention_impl="fast" if fast else "manual",
        norm_impl="fast" if fast else "manual",
        rope_impl="fast" if fast else "manual",
    )
    m = GPT(cfg)
    if dtype != mx.float32:
        m.set_dtype(dtype)
    mx.eval(m.parameters())
    return m, cfg


def _batch(B: int, T: int, seed: int):
    mx.random.seed(seed)
    x = mx.random.randint(0, VOCAB, (B, T))
    y = mx.random.randint(0, VOCAB, (B, T))
    mx.eval(x, y)
    return x, y


def _ce(m, x, y):
    l = m(x).reshape(-1, VOCAB)
    return -mx.mean((l - mx.logsumexp(l, axis=-1, keepdims=True))[mx.arange(y.size), y.reshape(-1)])


def bench_train_step(fast: bool, dtype, batch: int, compiled: bool, repeats: int):
    m, _ = _model(fast, dtype)
    import mlx.optimizers as optim

    opt = optim.AdamW(learning_rate=1e-3)
    x, y = _batch(batch, CTX, 1)
    lag = nn.value_and_grad(m, _ce)

    def micro(x, y):
        return lag(m, x, y)

    fn = (
        mx.compile(micro, inputs=[m.state, opt.state], outputs=[m.state, opt.state])
        if compiled
        else micro
    )
    # Warmup (traces compile, doesn't count).
    for _ in range(3):
        l, g = fn(x, y)
        opt.update(m, g)
        mx.eval(l, m.parameters(), opt.state)
    walls = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        l, g = fn(x, y)
        opt.update(m, g)
        mx.eval(l, m.parameters(), opt.state)
        walls.append(time.perf_counter() - t0)
    wall = statistics.median(walls)
    mx.reset_peak_memory()
    l, g = fn(x, y)
    opt.update(m, g)
    mx.eval(l, m.parameters(), opt.state)
    peak = float(mx.get_peak_memory()) / 1e6
    return {
        "wall_s": round(wall, 4),
        "tokens_per_s": round(batch * CTX / wall),
        "peak_mem_mb": round(peak, 1),
    }


def bench_infer(fast: bool, dtype, cached: bool, n_new: int = 64, repeats: int = 3):
    m, cfg = _model(fast, dtype)
    prompt = list(range(1, 129))

    def prefill(ids, caches):
        logits = m(mx.array([ids]), caches=caches)
        mx.eval(logits)
        return logits

    def decode_step(tok, off, caches):
        logits = m(mx.array([[tok]]), offset=off, caches=caches)
        mx.eval(logits)
        return int(mx.argmax(logits[0, -1]))

    # Warmup (kernel tracing/compile, doesn't count).
    if cached:
        wc = KVCache(cfg, batch_size=1)
        prefill(prompt, wc.layers)
        wc.seq_len = len(prompt)
        for _ in range(4):
            decode_step(prompt[-1], wc.seq_len, wc.layers)
            wc.seq_len += 1
    else:
        prefill(prompt, None)
        w = list(prompt)
        for _ in range(4):
            logits = m(mx.array([w[-CTX:]]))
            mx.eval(logits)
            w.append(int(mx.argmax(logits[0, -1])))

    import statistics

    pres, decs = [], []
    for _ in range(repeats):
        if cached:
            cache = KVCache(cfg, batch_size=1)
            t0 = time.perf_counter()
            prefill(prompt, cache.layers)
            pres.append(time.perf_counter() - t0)
            cache.seq_len = len(prompt)
            t0 = time.perf_counter()
            out = list(prompt)
            for _ in range(n_new):
                out.append(decode_step(out[-1], cache.seq_len, cache.layers))
                cache.seq_len += 1
            decs.append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            prefill(prompt, None)
            pres.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            out = list(prompt)
            for _ in range(n_new):
                logits = m(mx.array([out[-CTX:]]))
                mx.eval(logits)
                out.append(int(mx.argmax(logits[0, -1])))
            decs.append(time.perf_counter() - t0)
    prefill_s = statistics.median(pres)
    decode_s = statistics.median(decs)
    return {
        "prefill_tok_s": round(len(prompt) / prefill_s),
        "decode_tok_s": round(n_new / decode_s),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    repeats = 2 if args.quick else 3

    # Warmup soak: let the GPU clocks/thermals settle before timing (laptops
    # swing ~2x across configs otherwise, and run order swamps everything).
    print("== warmup soak (10s matmuls, doesn't count) ==")
    soak_end = time.perf_counter() + 10
    while time.perf_counter() < soak_end:
        a = mx.random.normal((512, 512))
        b = mx.random.normal((512, 512))
        mx.eval(a @ b)
    mx.reset_peak_memory()

    results: dict = {"preset": "tiny-32k-random-init", "context": CTX, "ablations": {}}
    ab = results["ablations"]

    print("== train step: manual vs fast (fp32, batch 4, compiled) ==")
    ab["train_manual_fp32"] = bench_train_step(False, mx.float32, 4, True, repeats)
    ab["train_fast_fp32"] = bench_train_step(True, mx.float32, 4, True, repeats)
    for k in ["train_manual_fp32", "train_fast_fp32"]:
        print(f"{k}: {ab[k]['tokens_per_s']:,} tok/s  mem {ab[k]['peak_mem_mb']} MB")

    print("== train step: compile on/off (fast, fp32, batch 4) ==")
    ab["train_fast_nocompile"] = bench_train_step(True, mx.float32, 4, False, repeats)
    print(f"no-compile: {ab['train_fast_nocompile']['tokens_per_s']:,} tok/s")

    print("== train step: fp32 vs bf16 (fast, batch 4, compiled) ==")
    ab["train_fast_bf16"] = bench_train_step(True, mx.bfloat16, 4, True, repeats)
    print(f"bf16: {ab['train_fast_bf16']['tokens_per_s']:,} tok/s")

    print("== train step: batch sweep (fast, fp32, compiled) ==")
    ab["batch_sweep"] = {}
    for b in [1, 4] if args.quick else [1, 2, 4, 8]:
        ab["batch_sweep"][str(b)] = bench_train_step(True, mx.float32, b, True, repeats)
        print(f"batch {b}: {ab['batch_sweep'][str(b)]['tokens_per_s']:,} tok/s")

    print("== inference: cache vs no-cache, manual vs fast ==")
    ab["infer"] = {}
    for fast in [False, True]:
        for cached in [True, False]:
            key = f"{'fast' if fast else 'manual'}_{'cache' if cached else 'nocache'}"
            ab["infer"][key] = bench_infer(fast, mx.float32, cached)
            r = ab["infer"][key]
            print(
                f"{key}: prefill {r['prefill_tok_s']:,} tok/s  decode {r['decode_tok_s']:,} tok/s"
            )

    results["machine_note"] = "speed doesn't depend on weights, so random init is fine"
    from datetime import date

    path = write_result(f"{date.today().isoformat()}_model", results)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
