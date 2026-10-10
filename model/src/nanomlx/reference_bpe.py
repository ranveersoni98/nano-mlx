"""Python byte-level BPE.

The Rust core (`tokenizer-core`) must reproduce this module's ids exactly.
Frozen rules this file defines:
- §5.1 ID layout: 0-255 bytes, then merges in order, then specials.
- §5.4 deterministic tie-breaks: highest count, then smallest (left, right).
- §6 `.bin` / `meta.json` format and the u16-iff-vocab<64k dtype rule.

The pre-tokenization regex is loaded from
`tokenizer/patterns/o200k_style.txt` (never duplicated here).
Needs the third-party `regex` module (Unicode \\p{...} classes + lookahead).
"""

from __future__ import annotations

import datetime
import json
import struct
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

import regex as _re

SAVE_VERSION = 1
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]

#: Hard ceiling for one regex input, in chars. Identical rule to Rust
#: `pretokenize::MAX_SEGMENT_CHARS` (fancy-regex has a hardcoded 1M-entry
#: backtrack stack; 1 MB letter runs overflow it, 64K chars keeps ~8x
#: margin). Longer segments are force-split at char boundaries in both
#: implementations so parity holds; merges never cross a force-split.
MAX_SEGMENT_CHARS = 65_536


def _split_long_segment(text: str) -> list[str]:
    if len(text) <= MAX_SEGMENT_CHARS:
        return [text]
    return [text[i : i + MAX_SEGMENT_CHARS] for i in range(0, len(text), MAX_SEGMENT_CHARS)]


def _pattern_path() -> Path:
    """Locate the single-source-of-truth pattern file from any cwd."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        cand = parent / "tokenizer" / "patterns" / "o200k_style.txt"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"tokenizer/patterns/o200k_style.txt not found above {here} (searched parents)"
    )


def load_pattern() -> str:
    return _pattern_path().read_text(encoding="utf-8").strip("\r\n")


def fnv1a64(data: bytes) -> int:
    """Portable FNV-1a-64. Must match Rust `vocab::fnv1a64` exactly."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _split_utf8_runs(data: bytes):
    """Yield `(is_valid, chunk)` with maximal valid-UTF-8 runs.

    Invalid bytes surface one at a time; each becomes a single-byte token,
    mirroring Rust `str::utf8_chunks` handling on the bytes path. Exact run
    boundaries need not match Rust - only the emitted ids must.
    """

    def seq_len(d: bytes, i: int) -> int:
        n = len(d)
        b = d[i]
        if b < 0x80:
            return 1
        if 0xC2 <= b <= 0xDF:
            need = 1
        elif 0xE0 <= b <= 0xEF:
            need = 2
        elif 0xF0 <= b <= 0xF4:
            need = 3
        else:
            return 0
        if i + need >= n:
            return 0  # truncated at end
        cont = d[i + 1 : i + 1 + need]
        if any(c & 0xC0 != 0x80 for c in cont):
            return 0
        # Overlong / surrogate / out-of-range rejections (RFC 3629).
        if b == 0xE0 and cont[0] < 0xA0:
            return 0
        if b == 0xED and cont[0] > 0x9F:
            return 0
        if b == 0xF0 and cont[0] < 0x90:
            return 0
        if b == 0xF4 and cont[0] > 0x8F:
            return 0
        return 1 + need

    i, n = 0, len(data)
    while i < n:
        ln = seq_len(data, i)
        if ln == 0:
            yield False, data[i : i + 1]
            i += 1
        else:
            j = i + ln
            while j < n:
                ln2 = seq_len(data, j)
                if ln2 == 0:
                    break
                j += ln2
            yield True, data[i:j]
            i = j


class Tokenizer:
    """Byte-level BPE tokenizer (reference implementation)."""

    def __init__(
        self,
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        # id -> bytes for normal tokens.
        self.token_bytes: list[bytes] = [bytes([b]) for b in range(256)]
        for i, (left, right) in enumerate(self.merges):
            if not (left < 256 + i and right < 256 + i):
                raise ValueError(f"merge {i} references future token(s): ({left}, {right})")
            self.token_bytes.append(self.token_bytes[left] + self.token_bytes[right])
        self.bytes_to_id = {b: i for i, b in enumerate(self.token_bytes)}
        base = 256 + len(self.merges)
        self.special_to_id = {s: base + i for i, s in enumerate(self.special_tokens)}
        self._re = _re.compile(load_pattern())

    # -- properties ------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes) + len(self.special_tokens)

    def pattern_hash(self) -> int:
        return fnv1a64(load_pattern().encode())

    def checksum(self) -> int:
        buf = b"".join(struct.pack("<II", l, r) for l, r in self.merges)
        return fnv1a64(buf)

    # -- training ----------------------------------------------------------
    @classmethod
    def train(
        cls,
        files_or_texts: str | Path | list[str | Path],
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Train on files (paths that exist) or raw strings.

        Accepts a single path/str or a list mixing both: entries resolving
        to an existing file are read, anything else is literal text.
        """
        specials = list(special_tokens) if special_tokens else list(DEFAULT_SPECIAL_TOKENS)
        if isinstance(files_or_texts, (str, Path)):
            files_or_texts = [files_or_texts]
        texts: list[str] = []
        for item in files_or_texts:
            p = Path(item) if isinstance(item, str) else item
            if isinstance(p, Path) and p.exists() and p.is_file():
                # newline="": no universal-newline translation. Training must
                # see exact bytes (Rust reads raw); translating \r\n -> \n
                # silently changes merges (caught by training parity).
                with open(p, encoding="utf-8", errors="surrogateescape", newline="") as f:
                    texts.append(f.read())
            else:
                texts.append(item if isinstance(item, str) else str(item))
        # Surrogateescape keeps file bytes recoverable; lone surrogates are
        # treated as ordinary (non-merging) content during counting.
        pattern = _re.compile(load_pattern())
        # One pass: matches are words, and so are the gaps between them
        # (Rust parity - every byte belongs to exactly one chunk). Long
        # texts are force-split first, exactly like Rust pretokenize.
        words: Counter[str] = Counter()
        for text in texts:
            for piece in _split_long_segment(text):
                pos = 0
                for m in pattern.finditer(piece):
                    s, e = m.span()
                    if s > pos:
                        words[piece[pos:s]] += 1
                    if e > s and m.group():
                        words[m.group()] += 1
                        pos = e
                    elif e == s and pos < len(piece):
                        # Zero-width guard (pattern never emits these).
                        words[piece[pos : pos + 1]] += 1
                        pos += 1
                if pos < len(piece):
                    words[piece[pos:]] += 1

        target_merges = max(0, vocab_size - 256 - len(specials))
        word_tokens: dict[str, list[int]] = {
            w: list(w.encode("utf-8", errors="surrogateescape")) for w in words
        }
        merges: list[tuple[int, int]] = []
        while len(merges) < target_merges:
            counts: Counter[tuple[int, int]] = Counter()
            for w, ids in word_tokens.items():
                f = words[w]
                for a, b in pairwise(ids):
                    counts[(a, b)] += f
            if not counts:
                break
            top = max(counts.values())
            if top < 2:
                break
            # Frozen tie-break (§5.4): highest count, then smallest pair.
            best = min(p for p, c in counts.items() if c == top)
            new_id = 256 + len(merges)
            merges.append(best)
            left, right = best
            for w, ids in word_tokens.items():
                if left not in ids or right not in ids:
                    # Cheap prefilter; full check below.
                    pass
                merged: list[int] = []
                i = 0
                changed = False
                while i < len(ids):
                    if i + 1 < len(ids) and ids[i] == left and ids[i + 1] == right:
                        merged.append(new_id)
                        i += 2
                        changed = True
                    else:
                        merged.append(ids[i])
                        i += 1
                if changed:
                    word_tokens[w] = merged
        return cls(merges, specials)

    # -- encoding ----------------------------------------------------------
    def _split_special(self, text: str, allowed: set[str]) -> list[tuple[bool, str]]:
        if not allowed or not text:
            return [] if not text else [(False, text)]
        pats = sorted(allowed, key=len, reverse=True)
        out: list[tuple[bool, str]] = []
        pos, run = 0, 0
        while pos < len(text):
            hit = next((p for p in pats if text.startswith(p, pos)), None)
            if hit is not None:
                if pos > run:
                    out.append((False, text[run:pos]))
                out.append((True, hit))
                pos += len(hit)
                run = pos
            else:
                pos += 1
        if run < len(text):
            out.append((False, text[run:]))
        return out

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        if not chunk:
            return []
        if chunk in self.bytes_to_id:
            return [self.bytes_to_id[chunk]]
        ids = list(chunk)
        # Merge all non-overlapping occurrences of the lowest-rank pair per
        # pass, then rescan. Output-identical to merging one occurrence at a
        # time (ranks are unique, so the global minimum is unambiguous), but
        # passes are bounded by the number of firing merges instead of the
        # number of occurrences.
        while len(ids) >= 2:
            best_rank = None
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
            if best_rank is None:
                break
            merged_id = 256 + best_rank
            merged: list[int] = []
            i = 0
            while i < len(ids):
                pair = (ids[i], ids[i + 1]) if i + 1 < len(ids) else None
                if pair is not None and self.rank.get(pair) == best_rank:
                    merged.append(merged_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        return ids

    def encode(self, text: str, allowed_special: str | set[str] = "none") -> list[int]:
        """Encode text. `allowed_special`: "none" (default), "all", or a set."""
        if allowed_special == "none":
            allowed: set[str] = set()
        elif allowed_special == "all":
            allowed = set(self.special_tokens)
        else:
            allowed = set(allowed_special or [])
        unknown = allowed - set(self.special_tokens)
        if unknown:
            raise ValueError(f"unknown special token(s): {sorted(unknown)}")
        out: list[int] = []
        for is_special, seg in self._split_special(text, allowed):
            if is_special:
                out.append(self.special_to_id[seg])
                continue
            for piece in _split_long_segment(seg):
                pos = 0
                for m in self._re.finditer(piece):
                    s, e = m.span()
                    if s > pos:
                        out.extend(self._encode_chunk(piece[pos:s].encode("utf-8")))
                    if e > s and m.group():
                        out.extend(self._encode_chunk(m.group().encode("utf-8")))
                        pos = e
                    elif e == s:
                        # Zero-width guard (pattern never emits these).
                        out.extend(self._encode_chunk(piece[pos : pos + 1].encode("utf-8")))
                        pos += 1
                if pos < len(piece):
                    out.extend(self._encode_chunk(piece[pos:].encode("utf-8")))
        return out

    def encode_batch(
        self, texts: list[str], allowed_special: str | set[str] = "none"
    ) -> list[list[int]]:
        return [self.encode(t, allowed_special) for t in texts]

    def encode_bytes(self, data: bytes, allowed_special: str | set[str] = "none") -> list[int]:
        out: list[int] = []
        for valid, run in _split_utf8_runs(data):
            if valid:
                out.extend(self.encode(run.decode("utf-8"), allowed_special))
            else:
                out.extend(run)
        return out

    def encode_file_to_bin(
        self, in_path: str | Path, out_path: str | Path, allowed_special: str | set[str] = "none"
    ) -> dict:
        """Encode a file to `.bin` (+ `meta.json`); the MLX pipeline input."""
        started = time.perf_counter()
        data = Path(in_path).read_bytes()
        ids = self.encode_bytes(data, allowed_special)
        use_u16 = self.vocab_size < 65536  # frozen dtype rule
        dtype = "uint16" if use_u16 else "uint32"
        fmt = "<" + ("H" * len(ids) if use_u16 else "I" * len(ids))
        Path(out_path).write_bytes(struct.pack(fmt, *ids) if ids else b"")
        eos = self.special_to_id.get("<|endoftext|>")
        if eos is None:
            eos = next(iter(self.special_to_id.values()), 0)
        meta = {
            "dtype": dtype,
            "vocab_size": self.vocab_size,
            "n_tokens": len(ids),
            "eos_id": eos,
            # Fingerprint of the training pattern; the loader compares it
            # for equality. (Key name follows the frozen .bin schema; the
            # value is FNV-1a until a hashing dependency is approved.)
            "tokenizer_sha256": f"{self.pattern_hash():016x}",
            "source": str(in_path),
            "created": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        meta_path = Path(out_path).with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {
            "n_tokens": len(ids),
            "bytes_in": len(data),
            "seconds": time.perf_counter() - started,
            "dtype": dtype,
        }

    # -- decoding ----------------------------------------------------------
    def decode(self, ids: list[int], strict: bool = False) -> str:
        """Ids -> str. Replacement character by default; `strict` raises."""
        vocab = self.vocab_size
        specials_base = len(self.token_bytes)
        buf = bytearray()
        for i in ids:
            if not 0 <= i < vocab:
                raise ValueError(f"invalid token id: {i}")
            if i < specials_base:
                buf.extend(self.token_bytes[i])
            else:
                buf.extend(self.special_tokens[i - specials_base].encode("utf-8"))
        if strict:
            return bytes(buf).decode("utf-8")  # raises UnicodeDecodeError
        return bytes(buf).decode("utf-8", errors="replace")

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "version": SAVE_VERSION,
            "pattern_hash": f"{self.pattern_hash():016x}",
            "special_tokens": self.special_tokens,
            "merges": [[l, r] for l, r in self.merges],
            "checksum": f"{self.checksum():016x}",
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != SAVE_VERSION:
            raise ValueError(
                f"vocab version mismatch: file is v{payload.get('version')}, "
                f"this code reads v{SAVE_VERSION}"
            )
        tok = cls(
            [tuple(m) for m in payload["merges"]],
            payload.get("special_tokens", []),
        )
        if f"{tok.pattern_hash():016x}" != payload.get("pattern_hash"):
            raise ValueError(
                "pattern mismatch: vocab was trained with a different "
                "tokenizer/patterns/o200k_style.txt - re-train"
            )
        if f"{tok.checksum():016x}" != payload.get("checksum"):
            raise ValueError("vocab checksum mismatch (file is corrupted)")
        return tok
