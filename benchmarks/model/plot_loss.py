#!/usr/bin/env python3
"""Plot training curves from a run's metrics.jsonl (script-made, not hand-drawn).

Usage:
    uv run --project benchmarks python benchmarks/model/plot_loss.py \
        --metrics data/ckpts/proof/logs/metrics.jsonl \
        --out benchmarks/charts/loss_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, val = {}, {}
    for line in Path(args.metrics).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        step = r["step"]
        if "train/loss" in r:
            train[step] = r["train/loss"]
        if "val/loss" in r:
            val[step] = r["val/loss"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sx, sy = zip(*sorted(train.items()))
    ax.plot(sx, sy, label="train loss", linewidth=1)
    if val:
        vx, vy = zip(*sorted(val.items()))
        ax.plot(vx, vy, "o-", label="val loss", markersize=4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("nanoMLX tiny preset, TinyStories (300 steps, proof run)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out} ({len(train)} train points, {len(val)} val points)")


if __name__ == "__main__":
    main()
