#!/usr/bin/env python3
"""Build the README benchmark tables from the saved results JSON (don't type them by hand).

Usage:
    uv run --project benchmarks python benchmarks/tokenizer/make_readme_table.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def latest(suffix: str) -> dict:
    paths = sorted(glob.glob(str(RESULTS / f"*_{suffix}.json")))
    if not paths:
        raise SystemExit(f"no results *_{suffix}.json - run the bench first")
    return json.loads(Path(paths[-1]).read_text(encoding="utf-8"))


def model_table() -> list[str]:
    m = latest("model")
    a = m["ablations"]
    mach = m["machine"]
    lines = [
        (
            "Model speed (tiny preset, 32k vocab, random init, "
            "speed doesn't depend on weights):"
        ),
        "",
        "| Train step (batch 4, ctx 256) | tok/s | Peak mem |",
        "|---|---|---|",
        f"| manual attention, fp32, compiled | {a['train_manual_fp32']['tokens_per_s']:,} | {a['train_manual_fp32']['peak_mem_mb']} MB |",
        f"| mx.fast SDPA, fp32, compiled | {a['train_fast_fp32']['tokens_per_s']:,} | {a['train_fast_fp32']['peak_mem_mb']} MB |",
        f"| mx.fast, fp32, no compile | {a['train_fast_nocompile']['tokens_per_s']:,} | - |",
        f"| mx.fast, bf16, compiled | {a['train_fast_bf16']['tokens_per_s']:,} | {a['train_fast_bf16']['peak_mem_mb']} MB |",
        "",
        "| Batch sweep (fast, fp32) | tok/s |",
        "|---|---|",
    ]
    for b, r in sorted(a["batch_sweep"].items(), key=lambda kv: int(kv[0])):
        lines.append(f"| batch {b} | {r['tokens_per_s']:,} |")
    lines += [
        "",
        "| Inference (128-token prompt, 64 new) | prefill tok/s | decode tok/s |",
        "|---|---|---|",
    ]
    for key, label in [
        ("manual_cache", "manual + KV cache"),
        ("manual_nocache", "manual, re-forward"),
        ("fast_cache", "mx.fast + KV cache"),
        ("fast_nocache", "mx.fast, re-forward"),
    ]:
        r = a["infer"][key]
        lines.append(f"| {label} | {r['prefill_tok_s']:,} | {r['decode_tok_s']:,} |")
    lines += [
        "",
        f"_Power: {mach.get('power_source', '?')} ({mach.get('battery_pct', '?')}%). "
        + "Warmup thrown away, median of repeats with mx.eval inside. "
        + "Few-ms prefill timings wobble on laptops, so take the relative "
        + "bits seriously (cache ~2x decode, fast ~1.5x decode, bf16 +~30% "
        + "train, compile ~+10%, batch goes sublinear), they hold up run to run._",
    ]
    return lines


def main() -> None:
    enc = latest("tokenizer_encode")
    tr = latest("tokenizer_train")
    mach = enc["machine"]
    cpu = mach.get("cpu_brand", mach.get("chip", "?"))
    w = enc["workloads"]

    lines = [
        (
            "<!-- Made from benchmarks/results/*.json by "
            "benchmarks/tokenizer/make_readme_table.py, don't edit by hand. -->"
        ),
        (
            f"_Machine: {cpu}, {mach.get('ram_gb', '?')}GB RAM, "
            f"Python {mach.get('python', '?')}, {mach.get('rust_version', '')}. "
            "Release build. Median of repeats, warmup thrown away._"
        ),
        "",
        "Tokenizer encode (same inputs, different vocabs, so ids aren't compared):",
        "",
        "| Workload | ours (Rust) | HF tokenizers | tiktoken o200k |",
        "|---|---|---|---|",
    ]
    lf = w["large_file_100mb"]
    lines.append(
        f"| 100MB file | {lf['ours-rust']['tokens_per_s']:,} tok/s "
        f"({lf['ours-rust']['mb_per_s']} MB/s) | "
        f"{lf['hf-tokenizers']['tokens_per_s']:,} tok/s "
        f"({lf['hf-tokenizers']['mb_per_s']} MB/s) | "
        f"{lf['tiktoken-o200k']['tokens_per_s']:,} tok/s "
        f"({lf['tiktoken-o200k']['mb_per_s']} MB/s) |"
    )
    lat = w["short_latency"]
    lines.append(
        f"| short-string latency p50/p99 | {lat['ours-rust']['p50_us']}/{lat['ours-rust']['p99_us']} µs | "
        f"{lat['hf-tokenizers']['p50_us']}/{lat['hf-tokenizers']['p99_us']} µs | "
        f"{lat['tiktoken-o200k']['p50_us']}/{lat['tiktoken-o200k']['p99_us']} µs |"
    )
    for key, label in [("code_heavy", "code-heavy"), ("multilingual", "multilingual")]:
        r = w[key]
        lines.append(
            f"| {label} | {r['ours-rust']['tokens_per_s']:,} tok/s | "
            f"{r['hf-tokenizers']['tokens_per_s']:,} tok/s | "
            f"{r['tiktoken-o200k']['tokens_per_s']:,} tok/s |"
        )
    full = w.get("full_file_ours_only")
    if full:
        lines.append(
            f"| full 1.9GB file (ours only) | {full['tokens_per_s']:,} tok/s "
            f"({full['mb_per_s']} MB/s) | - | - |"
        )
    lines += [
        "",
        "Tokenizer training (same 20MB slice, vocab 5000, tiktoken has no trainer):",
        "",
        "| Implementation | Wall time |",
        "|---|---|",
        f"| ours (Rust) | {tr['ours_rust_s']}s ({tr['ours_merges']} merges) |",
        f"| HF tokenizers | {tr['hf_tokenizers_s']}s |",
        "",
        (
            "_Heads up: single-doc inputs only use one thread on our side "
            "(file and batch paths do parallelize). tiktoken wins big-file "
            "encode because its regex stays linear where our backtracking "
            "one slows down._"
        ),
    ]
    print("\n".join(lines))
    print()
    print("\n".join(model_table()))


if __name__ == "__main__":
    main()
