"""Text generation (M6): KV-cache prefill/decode + sampling + CLI.

Prefill runs the whole prompt in one forward pass (filling the cache);
decode runs one token at a time. Sampling order is fixed (§9):
repetition penalty -> temperature -> top-k -> top-p -> softmax -> sample.
Stops on EOS, --max-new-tokens, or context overflow (left-truncate,
documented; never silently corrupt RoPE positions).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from nanomlx.config import ModelConfig
from nanomlx.model import GPT, KVCache
from nanomlx.train import _load_dict, dotted_key
from nanomlx.utils import unflatten_params


def load_model(ckpt_dir: str | Path, vocab_size: int) -> tuple[GPT, ModelConfig, dict]:
    """Load weights + config from a training checkpoint."""
    ckpt_dir = Path(ckpt_dir)
    metas = sorted(ckpt_dir.glob("step_*/meta.json"))
    if not metas:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    cfg = ModelConfig(**meta["model_cfg"])
    assert cfg.vocab_size == vocab_size, "vocab size mismatch with checkpoint"
    model = GPT(cfg)
    weights = _load_dict(metas[-1].parent / "weights.safetensors")
    model.update(unflatten_params({dotted_key(k): v for k, v in weights.items()}))
    mx.eval(model.parameters())
    return model, cfg, meta


def sample_next(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    recent: list[int],
    rng: np.random.Generator,
) -> int:
    """Single next-token sample. Fixed order: rep-penalty -> temp -> top-k
    -> top-p -> softmax -> sample. temperature 0 = greedy."""
    l = logits.astype(np.float64)
    if repetition_penalty != 1.0 and recent:
        for t in set(recent):
            if l[t] > 0:
                l[t] /= repetition_penalty
            else:
                l[t] *= repetition_penalty
    if temperature <= 0:
        return int(np.argmax(l))
    l = l / temperature
    if top_k > 0:
        cutoff = np.sort(l)[-min(top_k, len(l))]
        l = np.where(l < cutoff, -np.inf, l)
    if top_p < 1.0:
        order = np.argsort(-l)
        probs = np.exp(l[order] - np.max(l[order]))
        probs /= probs.sum()
        keep = np.cumsum(probs) <= top_p
        keep[0] = True  # always keep at least the best
        drop = np.zeros_like(keep, dtype=bool)
        drop[order] = ~keep
        l = np.where(drop, -np.inf, l)
    probs = np.exp(l - np.max(l))
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


class _Streamer:
    """Incremental terminal output: prints only text not shown before.

    Decodes the full id list each step but emits just the new suffix, so
    terminals show one growing line instead of N full reprints. A trailing
    replacement char (token split mid-UTF-8-sequence) is held back until it
    resolves - printing it would leave permanent junk. Invariant: `printed`
    only advances over exact prefix matches, so it stays a true prefix of
    every future decode (UTF-8 decoding is stateless per character).
    """

    def __init__(self, decode):
        self.decode = decode
        self.printed = ""

    def show(self, ids: list[int]) -> None:
        text = self.decode(ids)
        if not text.startswith(self.printed):
            # Recovery path (unreachable while the invariant holds): resync.
            print(flush=True)
            print(text, end="", flush=True)
            self.printed = text
            return
        delta = text[len(self.printed) :]
        delta = delta.removesuffix("�")
        if delta:
            print(delta, end="", flush=True)
            self.printed += delta


def generate(
    model: GPT,
    encode,
    decode,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    eos_id: int | None = None,
    seed: int = 1337,
    stream: bool = False,
    use_cache: bool = True,
) -> list[int]:
    """Generate with prefill + cached decode (M6). `use_cache=False` keeps
    the M5 full-re-forward path (equivalence-tested)."""
    rng = np.random.default_rng(seed)
    ids = list(encode(prompt))
    context = model.cfg.context
    if len(ids) > context:
        ids = ids[-context:]  # left-truncate (documented); never corrupt RoPE
    out = list(ids)
    if max_new_tokens <= 0:
        return []
    streamer = _Streamer(decode) if stream else None

    def step_logits(window: list[int], offset: int, caches) -> np.ndarray:
        logits = model(mx.array([window]), offset=offset, caches=caches)
        mx.eval(logits)
        return np.array(logits[0, -1])

    def take(logits: np.ndarray) -> int:
        nxt = sample_next(logits, temperature, top_k, top_p, repetition_penalty, out, rng)
        out.append(nxt)
        if streamer is not None:
            streamer.show(out)
        return nxt

    if use_cache:
        cache = KVCache(model.cfg, batch_size=1)
        # Prefill: whole prompt in one pass. Its LAST logits already predict
        # token N - sample it directly (re-feeding ids[-1] at offset N would
        # duplicate that token into the cache at the wrong position).
        logits = step_logits(ids, 0, cache.layers)
        mx.eval(*[c["k"] for c in cache.layers], *[c["v"] for c in cache.layers])
        cache.seq_len = len(ids)
        nxt = take(logits)
        if not (eos_id is not None and nxt == eos_id):
            # Guard on len(out): decode at offset=len(out) writes that row,
            # valid iff < context (seq_len tracks len(out) throughout).
            while len(out) - len(ids) < max_new_tokens and len(out) < context:
                nxt = take(step_logits([out[-1]], cache.seq_len, cache.layers))
                cache.seq_len += 1
                if eos_id is not None and nxt == eos_id:
                    break
    else:
        while len(out) - len(ids) < max_new_tokens and len(out) < context:
            nxt = take(step_logits(out[-context:], 0, None))
            if eos_id is not None and nxt == eos_id:
                break
    return out[len(ids) :]


def _call(tok, name: str):
    """Method-or-attribute access: Rust exposes merges()/special_tokens()
    as methods, the reference as attributes."""
    attr = getattr(tok, name)
    return attr() if callable(attr) else attr


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate with a nanoMLX checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import sys

    try:
        import nanomlx_tokenizer as RT

        tok = RT.Tokenizer.load(args.vocab)
        encode, decode = tok.encode, tok.decode
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from nanomlx.reference_bpe import Tokenizer

        tok = Tokenizer.load(args.vocab)
        encode, decode = tok.encode, tok.decode

    vocab_size = len(_call(tok, "merges")) + 256 + len(_call(tok, "special_tokens"))
    model, _, _ = load_model(args.ckpt, vocab_size)
    eos = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    eos_id = None
    if "<|endoftext|>" in eos.get("special_tokens", []):
        eos_id = 256 + len(_call(tok, "merges")) + eos["special_tokens"].index("<|endoftext|>")
    generate(
        model,
        encode,
        decode,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=eos_id,
        seed=args.seed,
        stream=True,
    )
    print()  # deltas already streamed above; just end the line


if __name__ == "__main__":
    main()
