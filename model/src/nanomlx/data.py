"""Loading `.bin` files with memmap.

How it works: raw TinyStories text gets EOS markers stuck between docs,
our Rust tokenizer (release build, please) encodes it to `.bin` plus a
`meta.json`, then we load that with `np.memmap` and sample random windows.

We use `nanomlx_tokenizer` (Rust). If it's not built we fall back to the
pure-Python one, which works but is slow. You'll get a warning.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

_DTYPE_MAP = {"uint16": np.uint16, "uint32": np.uint32}


def load_tokenizer(vocab_path: str | Path):
    """Load our Rust tokenizer. Falls back to the Python one if needed (slow)."""
    try:
        import nanomlx_tokenizer

        return nanomlx_tokenizer.Tokenizer.load(str(vocab_path))
    except ImportError:
        import warnings

        warnings.warn("nanomlx_tokenizer missing: reference fallback (slow)")
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from nanomlx.reference_bpe import Tokenizer

        return Tokenizer.load(vocab_path)


def eos_string(vocab_path: str | Path) -> str:
    payload = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    specials = payload.get("special_tokens", [])
    if "<|endoftext|>" in specials:
        return "<|endoftext|>"
    if specials:
        return specials[0]
    raise ValueError(f"no special tokens in {vocab_path}")


def build_bins(
    vocab_path: str | Path,
    train_txt: str | Path,
    val_txt: str | Path,
    out_dir: str | Path,
    tmp_suffix: str = ".eos_marked.tmp",
) -> dict:
    """Encode raw text files to EOS-separated train/val `.bin` plus meta.json.

    We join docs (= lines) with `<|endoftext|>` markers passed as allowed
    specials, so the `.bin` stream has EOS between docs. The source text
    can't contain the marker itself (we check, TinyStories has zero).
    """
    vocab_path, out_dir = Path(vocab_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eos = eos_string(vocab_path)
    tok = load_tokenizer(vocab_path)
    stats = {}
    for split, txt in [("train", Path(train_txt)), ("val", Path(val_txt))]:
        raw = txt.read_bytes()
        if eos.encode() in raw:
            raise ValueError(f"{txt} contains the EOS marker as text, bailing out")
        # Join docs with EOS markers in a temp file (a bit of extra disk,
        # no extra RAM, deleted after).
        tmp = out_dir / f"{split}{tmp_suffix}"
        marker = f"\n{eos}\n".encode()
        with open(txt, "rb") as fin, open(tmp, "wb") as fout:
            fout.writelines(line.rstrip(b"\r\n") + marker for line in fin)
        out_bin = out_dir / f"{split}.bin"
        if hasattr(tok, "encode_file_to_bin"):
            try:
                s = tok.encode_file_to_bin(str(tmp), str(out_bin), [eos])
            except TypeError:
                s = tok.encode_file_to_bin(str(tmp), str(out_bin))
        else:
            s = tok.encode_file_to_bin(str(tmp), str(out_bin))
        tmp.unlink(missing_ok=True)
        check_bin(out_bin, vocab_path)
        stats[split] = s
        print(f"{split}: {s['n_tokens']:,} tokens, {s['dtype']}, {s['seconds']:.1f}s")
    return stats


def check_bin(bin_path: str | Path, vocab_path: str | Path) -> dict:
    """Check a `.bin` against its `meta.json` and vocab. Bails on mismatch."""
    bin_path = Path(bin_path)
    meta_path = bin_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    assert meta["tokenizer_sha256"] == vocab["pattern_hash"], (
        f"tokenizer mismatch: .bin was encoded with pattern "
        f"{meta['tokenizer_sha256']}, vocab has {vocab['pattern_hash']}"
    )
    dtype = _DTYPE_MAP[meta["dtype"]]
    itemsize = np.dtype(dtype).itemsize
    assert bin_path.stat().st_size == meta["n_tokens"] * itemsize, "truncated .bin?"
    assert meta["vocab_size"] == 256 + len(vocab["merges"]) + len(vocab["special_tokens"])
    data = np.memmap(bin_path, dtype=dtype, mode="r")
    assert int(data.max()) < meta["vocab_size"], "id out of vocab range"
    return meta


class BinData:
    """A token stream sitting in a memmap file, plus its checked meta."""

    def __init__(self, bin_path: str | Path, vocab_path: str | Path):
        self.meta = check_bin(bin_path, vocab_path)
        self.data = np.memmap(bin_path, dtype=_DTYPE_MAP[self.meta["dtype"]], mode="r")
        self.n_tokens = int(self.meta["n_tokens"])
        self.eos_id = int(self.meta["eos_id"])

    def __len__(self) -> int:
        return self.n_tokens


class BatchSampler:
    """Random windows: x = d[i:i+T], y = d[i+1:i+T+1]."""

    def __init__(self, data: BinData, context: int, seed: int = 1337):
        assert data.n_tokens > context + 1, "not enough tokens for one window"
        self.data = data
        self.context = context
        self.rng = np.random.RandomState(seed)

    def sample(self, batch_size: int) -> tuple[mx.array, mx.array]:
        T = self.context
        lo = np.int64(self.data.n_tokens - T - 1)
        starts = self.rng.randint(0, lo, size=batch_size)
        x = np.stack([self.data.data[s : s + T] for s in starts]).astype(np.int64)
        y = np.stack([self.data.data[s + 1 : s + T + 1] for s in starts]).astype(np.int64)
        return mx.array(x), mx.array(y)

    def get_state(self) -> dict:
        return {"rng_state": self.rng.get_state()}

    def set_state(self, state: dict) -> None:
        self.rng.set_state(state["rng_state"])
