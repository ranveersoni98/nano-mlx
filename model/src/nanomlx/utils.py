"""Seeding, LR schedule, optimizer groups, checkpoints, logging.

One gotcha: every entry point takes `--seed` (default 1337) and seeds
Python, NumPy, and MLX. MLX doesn't let you get/set its RNG state, so
resumed runs restore Python/NumPy/sampler state exactly and just keep
going on the MLX stream. Close enough, we note it and move on.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    mx.random.seed(seed)


def lr_at(step: int, peak: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Linear warmup up to peak, then cosine decay down to min_ratio * peak."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_ratio + (1.0 - min_ratio) * cosine)


def is_decay_param(key: str, param: mx.array) -> bool:
    """Weight decay only goes on 2-D matrix weights. No decay on norms
    (`.attn_norm.` / `.mlp_norm.` / `final_norm`), biases, embeddings
    (`tok_emb`), or the untied head (`lm_head`)."""
    return (
        key.endswith(".weight")
        and len(param.shape) == 2
        and "tok_emb" not in key
        and "lm_head" not in key
        and "norm" not in key
    )


def build_optimizer(lr: float, wd: float = 0.1) -> optim.MultiOptimizer:
    """AdamW where only the matrices get decay, via param groups."""
    return optim.MultiOptimizer(
        [
            optim.AdamW(learning_rate=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd),
            optim.AdamW(learning_rate=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        ],
        filters=[is_decay_param],
    )


def _map_tree(tree, fn):
    if isinstance(tree, dict):
        return {k: _map_tree(v, fn) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_map_tree(v, fn) for v in tree]
    return fn(tree)


def clip_grad_norm(grads: dict, max_norm: float = 1.0) -> tuple[dict, float]:
    """Clip by global norm. Returns (scaled grads, norm before clipping)."""
    from nanomlx.model import tree_flatten

    leaves = [v for _, v in tree_flatten(grads)]
    total = float(mx.sqrt(sum(mx.sum(mx.square(g.astype(mx.float32))) for g in leaves)))
    mx.eval(total)
    scale = min(1.0, max_norm / (total + 1e-12))
    clipped = _map_tree(grads, lambda g: g * scale if isinstance(g, mx.array) else g)
    return clipped, total


def flatten_params(tree) -> dict[str, mx.array]:
    from nanomlx.model import tree_flatten

    return {k: v for k, v in tree_flatten(tree)}


def unflatten_params(flat: dict[str, mx.array]) -> dict:
    """Undo dotted-path flattening (number bits mean list indices)."""
    root: dict = {}
    for key, value in flat.items():
        parts = key.split(".") if key else []
        node = root
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            if part.isdigit():
                idx = int(part)
                while len(node) <= idx:  # type: ignore[arg-type]
                    node.append({})  # type: ignore[attr-defined]
                if last:
                    node[idx] = value  # type: ignore[index]
                else:
                    if not isinstance(node[idx], (dict, list)):  # type: ignore[index]
                        nxt = parts[i + 1]
                        node[idx] = [] if nxt.isdigit() else {}  # type: ignore[index]
                    node = node[idx]  # type: ignore[index]
            else:
                if last:
                    node[part] = value
                else:
                    if part not in node or not isinstance(node[part], (dict, list)):
                        nxt = parts[i + 1]
                        node[part] = [] if nxt.isdigit() else {}
                    node = node[part]
    return root


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def peak_memory_mb() -> float:
    try:
        return float(mx.get_peak_memory()) / 1e6
    except Exception:  # noqa: BLE001 - just a probe, any backend hiccup means -1
        return -1.0


def _best_effort(fn) -> None:
    """Run logging/teardown stuff that should never crash training."""
    try:
        fn()
    except Exception:  # noqa: BLE001, S110 - logging should never crash a run
        pass


class RunLogger:
    """Always writes metrics.jsonl. TensorBoard too if we can import a writer.

    We don't depend on torch or tensorboardX (torch is out of scope for
    now), so TB is try-it-and-see: jsonl is the real record and you can
    read it without running any server.
    """

    def __init__(self, log_dir: str | Path):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # We hold this handle open (closed in close(), not a with-block).
        self.jsonl = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")  # noqa: SIM115
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(self.dir))
        except ImportError:
            try:
                from tensorboardX import SummaryWriter  # type: ignore[import]

                self.tb = SummaryWriter(str(self.dir))
            except ImportError:
                self.tb = None

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.jsonl.write(json.dumps({"step": step, tag: value}) + "\n")
        self.jsonl.flush()
        if self.tb is not None:
            _best_effort(lambda: self.tb.add_scalar(tag, value, step))

    def close(self) -> None:
        _best_effort(self.jsonl.close)
        if self.tb is not None:
            _best_effort(self.tb.close)
