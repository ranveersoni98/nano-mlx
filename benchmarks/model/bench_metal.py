#!/usr/bin/env python3
"""Our fused RMSNorm Metal kernel vs the built-ins.

Shapes match real model rows: (2048, 384) tiny-batch, (8192, 512) small,
(8192, 768) base-ish. We report GB/s over input+weight+output traffic.
Warmup (Metal compile) doesn't count, median of repeats, mx.eval inside.
Saves JSON.

Needs mlx, so run it with the MODEL env:
    uv run --project model python benchmarks/model/bench_metal.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "src"))

import mlx.core as mx
from nanomlx.metal import fused_rms_norm
from nanomlx.model import RMSNorm

from nanomlx_bench import write_result


def bench(fn, *args, repeats: int = 7):
    fn(*args)  # warmup (Metal compile for our kernel), doesn't count
    mx.eval(*[a for a in args if isinstance(a, mx.array)])
    walls = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        walls.append(time.perf_counter() - t0)
    return statistics.median(walls)


def main() -> None:
    mx.random.seed(0)
    results: dict = {"kernel": "fused RMSNorm (1 thread/row, fp32 accumulate)", "shapes": {}}
    for nrows, dim in [(2048, 384), (8192, 512), (8192, 768), (32768, 512)]:
        x = mx.random.normal((nrows, dim)) * 3.0
        w = mx.random.normal((dim,)) + 2.0
        mx.eval(x, w)
        m = RMSNorm(dim)
        m.weight = w
        traffic_gb = (nrows * dim * 4 * 2 + dim * 4 + nrows * dim * 4) / 1e9
        row = {}
        for name, fn in [
            ("manual", lambda m=m, x=x: m(x)),
            ("mx_fast", lambda x=x, w=w: mx.fast.rms_norm(x, w, 1e-5)),
            ("metal_fused", lambda x=x, w=w: fused_rms_norm(x, w, 1e-5)),
        ]:
            wall = bench(fn, repeats=7)
            row[name] = {
                "wall_ms": round(wall * 1000, 3),
                "gb_per_s": round(traffic_gb / wall, 1),
            }
        results["shapes"][f"{nrows}x{dim}"] = row
        r = row
        print(
            f"{nrows}x{dim}: manual {r['manual']['gb_per_s']} GB/s | "
            f"mx.fast {r['mx_fast']['gb_per_s']} GB/s | "
            f"metal {r['metal_fused']['gb_per_s']} GB/s"
        )
    from datetime import date

    path = write_result(f"{date.today().isoformat()}_metal_rmsnorm", results)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
