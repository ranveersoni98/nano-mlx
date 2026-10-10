#!/usr/bin/env python3
"""Download raw training text into data/ (gitignored).

M5 uses TinyStories train+validation; this script only fetches and stores
the raw text (tokenizing to .bin happens in M5 with the trained 32k vocab).
Re-running is idempotent: existing files are left alone.

Usage:
    uv run --project model python scripts/download_tinystories.py [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tinystories"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max docs per split (0 = all)")
    args = ap.parse_args()

    from datasets import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")
    for split in ["train", "validation"]:
        path = OUT / f"{split}.txt"
        if path.exists():
            print(f"exists, skipping: {path}")
            continue
        rows = ds[split]
        if args.limit:
            rows = rows.select(range(min(args.limit, len(rows))))
        n_chars = 0
        with open(path, "w", encoding="utf-8") as f:
            for ex in rows:
                text = ex["text"].strip()
                if not text:
                    continue
                f.write(text + "\n")
                n_chars += len(text) + 1
        mb = path.stat().st_size / 1e6
        print(f"wrote {path} ({mb:.1f} MB, ~{n_chars / 1e6:.1f}M chars)")


if __name__ == "__main__":
    main()
