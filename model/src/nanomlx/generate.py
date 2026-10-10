"""Text generation: KV-cache prefill/decode plus sampling plus CLI.

Prefill runs the whole prompt in one go (fills the cache), decode runs
one token at a time. Sampling always goes in this order: repetition
penalty, then temperature, then top-k, then top-p, then softmax, then
sample. We stop on EOS, on --max-new-tokens, or when we'd overflow
context (we left-truncate and say so, never silently mess up positions).
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
    """Load weights and config back from a training checkpoint."""
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
    """Pick the next token. Order is fixed: rep-penalty, temp, top-k,
    top-p, softmax, sample. Temperature 0 means greedy."""
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
    """Prints new text as it comes in, without reprinting everything.

    We decode the full id list each step but only print the new tail, so
    the terminal shows one growing line. A trailing replacement char means
    we split a UTF-8 sequence mid-token, so we hold it back until it
    resolves instead of printing junk. `printed` only moves forward on
    exact prefix matches, so it stays a true prefix of every future decode.
    """

    def __init__(self, decode):
        self.decode = decode
        self.printed = ""

    def show(self, ids: list[int]) -> None:
        text = self.decode(ids)
        if not text.startswith(self.printed):
            # Shouldn't happen while the prefix rule holds, but resync if it does.
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
    """Generate with prefill plus cached decode. `use_cache=False` keeps
    the old full-re-forward path (we test that both agree)."""
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
        # Prefill the whole prompt in one pass. Its LAST logits already
        # predict the next token, so sample that directly (feeding ids[-1]
        # back in at offset N would put a dupe token in the cache).
        logits = step_logits(ids, 0, cache.layers)
        mx.eval(*[c["k"] for c in cache.layers], *[c["v"] for c in cache.layers])
        cache.seq_len = len(ids)
        nxt = take(logits)
        if not (eos_id is not None and nxt == eos_id):
            # len(out) is the next offset to write. That's only valid while
            # it's still under context (seq_len tracks len(out) the whole time).
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
    """Get a method-or-attribute: the Rust tokenizer exposes merges() and
    special_tokens() as methods, the Python one as plain attributes."""
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
    print()  # the deltas above already streamed, this just ends the line


if __name__ == "__main__":
    main()
