"""Training loop: AdamW with warmup/cosine, clipping, grad accumulation,
evals, safetensors checkpoints with resume, jsonl logging (plus TensorBoard
if it's around).

A few things we always do: loss in float32, `mx.eval` every step so the
lazy graph doesn't grow, tokens/step = batch * context * grad_accum,
perplexity = exp(val). Optimizer state is stored with `|`-joined flat
paths (can't use `_` as the separator since `tok_emb` has underscores).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from nanomlx.config import ModelConfig, TrainConfig, preset, train_preset
from nanomlx.data import BatchSampler, BinData
from nanomlx.model import GPT
from nanomlx.utils import (
    RunLogger,
    build_optimizer,
    clip_grad_norm,
    flatten_params,
    git_commit,
    lr_at,
    peak_memory_mb,
    seed_all,
    unflatten_params,
)

SEP = "|"


def flat_key(dotted: str) -> str:
    return dotted.replace(".", SEP)


def dotted_key(flat: str) -> str:
    return flat.replace(SEP, ".")


def ce_loss(model: GPT, x: mx.array, y: mx.array) -> mx.array:
    """Plain next-token cross-entropy in float32, nothing masked out."""
    logits = model(x).reshape(-1, model.cfg.vocab_size)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.mean(logp[mx.arange(y.size), y.reshape(-1)])


def _sampler_state_json(sampler_state: dict) -> dict:
    rng = sampler_state.get("rng_state")
    if rng is None:
        return {}
    return {
        "rng_state": [rng[0], [a.tolist() for a in rng[1]], rng[2], rng[3], rng[4]],
    }


def _sampler_state_restore(d: dict) -> dict:
    rng = d.get("rng_state")
    if rng is None:
        return {}
    return {
        "rng_state": (
            rng[0],
            np.array(rng[1], dtype=np.uint32),
            int(rng[2]),
            int(rng[3]),
            float(rng[4]),
        ),
    }


def save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: GPT,
    opt,
    sampler: BatchSampler,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    vocab_path: Path,
    val_loss: float | None,
    keep_last: int = 3,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / f"step_{step:06d}"
    dest.mkdir(exist_ok=True)
    mx.save_safetensors(str(dest / "weights.safetensors"), flatten_params(dict(model.parameters())))
    mx.savez(
        str(dest / "optim.npz"),
        **{flat_key(k): v for k, v in flatten_params(opt.state).items()},
    )
    vocab_hash = json.loads(Path(vocab_path).read_text(encoding="utf-8"))["pattern_hash"]
    meta = {
        "step": step,
        "val_loss": val_loss,
        "sampler_state": _sampler_state_json(sampler.get_state()),
        "model_cfg": vars(model_cfg),
        "train_cfg": vars(train_cfg),
        "tokenizer_pattern_hash": vocab_hash,
        "git_commit": git_commit(),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    steps = sorted(int(p.name.split("_")[1]) for p in ckpt_dir.glob("step_*") if p.is_dir())
    for old in steps[:-keep_last]:
        shutil.rmtree(ckpt_dir / f"step_{old:06d}", ignore_errors=True)
    return dest


def _load_dict(path: str | Path) -> dict[str, mx.array]:
    """Load one of our checkpoints with mx.load. Has to be a flat dict of arrays."""
    data = mx.load(str(path))
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not load as a dict (got {type(data).__name__})")
    return data


def load_checkpoint(ckpt_step_dir: str | Path, model: GPT, opt, vocab_path: str | Path):
    """Bring back weights, optimizer, and sampler state. Returns (step, sampler_state)."""
    dest, vocab_path = Path(ckpt_step_dir), Path(vocab_path)
    weights = _load_dict(dest / "weights.safetensors")
    model.update(unflatten_params({dotted_key(k): v for k, v in weights.items()}))
    raw = _load_dict(dest / "optim.npz")
    # Assign, don't .update(): MultiOptimizer.state is a property that fans
    # out to the sub-optimizers on set. Updating the dict in place would
    # quietly do nothing.
    opt.state = unflatten_params({dotted_key(k): v for k, v in raw.items()})
    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    vocab_hash = json.loads(vocab_path.read_text(encoding="utf-8"))["pattern_hash"]
    if meta["tokenizer_pattern_hash"] != vocab_hash:
        raise ValueError("checkpoint tokenizer != current vocab - refusing to resume")
    mx.eval(model.parameters(), opt.state)
    return meta["step"], _sampler_state_restore(meta["sampler_state"])


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    bin_dir: str | Path,
    ckpt_dir: str | Path,
    vocab_path: str | Path,
    max_steps: int | None = None,
    resume: str | Path | None = None,
    keep_last: int = 3,
    log_every: int = 10,
) -> dict:
    """Run training. Returns a summary like {steps, best_val_loss, last_loss}."""
    seed_all(train_cfg.seed)
    bin_dir, ckpt_dir, vocab_path = Path(bin_dir), Path(ckpt_dir), Path(vocab_path)
    train_data = BinData(bin_dir / "train.bin", vocab_path)
    val_data = BinData(bin_dir / "val.bin", vocab_path)
    sampler = BatchSampler(train_data, model_cfg.context, seed=train_cfg.seed)
    val_sampler = BatchSampler(val_data, model_cfg.context, seed=train_cfg.seed + 1)

    model = GPT(model_cfg)
    opt = build_optimizer(train_cfg.peak_lr, train_cfg.weight_decay)
    logger = RunLogger(ckpt_dir / "logs")

    start_step, best_val, val_loss = 0, float("inf"), None
    if resume is not None:
        start_step, sampler_state = load_checkpoint(Path(resume), model, opt, vocab_path)
        sampler.set_state(sampler_state)
        print(f"resumed from {resume} at step {start_step}")

    total = train_cfg.total_steps
    # max_steps stops early but doesn't reshape the schedule: the LR horizon
    # stays at `total` so picking up from a checkpoint keeps the exact same
    # schedule (we test this).
    end = min(max_steps, total) if max_steps else total
    assert end > start_step, f"nothing to do: start {start_step} >= end {end}"
    mx.eval(model.parameters(), opt.state)

    loss_and_grad = nn.value_and_grad(model, ce_loss)

    def micro(x, y):
        # Closed over (model, opt): the compiled trace can't take the module
        # itself as an arg (it shows up as a param dict inside).
        return loss_and_grad(model, x, y)

    state = [model.state, opt.state]
    try:
        compiled = mx.compile(micro, inputs=state, outputs=state)
        # Smoke batch shouldn't move the sampler, so snapshot and restore it.
        snap = sampler.get_state()
        x0, y0 = sampler.sample(train_cfg.batch_size)
        l_ref, _ = micro(x0, y0)
        l_cmp, _ = compiled(x0, y0)
        sampler.set_state(snap)
        mx.eval(l_ref, l_cmp)
        assert abs(float(l_ref) - float(l_cmp)) < 1e-4, "compile diverged"
        step_fn = compiled
    except Exception as e:  # noqa: BLE001 - if compile blows up for any reason, eager is fine
        print(f"compile unavailable, running eager instead ({type(e).__name__})")
        step_fn = micro

    def set_lr(step: int) -> float:
        lr = lr_at(step, train_cfg.peak_lr, train_cfg.warmup_steps, total, train_cfg.min_lr_ratio)
        for sub in opt.optimizers:
            sub.learning_rate = lr
        return lr

    tokens_per_step = train_cfg.batch_size * model_cfg.context * train_cfg.grad_accum
    last_ckpt = None
    loss_val = math.nan  # gets overwritten below since end > start_step is asserted
    step = start_step
    try:
        for step in range(start_step, end):
            t0 = time.perf_counter()
            lr = set_lr(step)
            acc: dict | None = None
            loss_val = 0.0
            for _ in range(train_cfg.grad_accum):
                x, y = sampler.sample(train_cfg.batch_size)
                loss_m, grads_m = step_fn(x, y)
                mx.eval(loss_m, grads_m)
                loss_val += float(loss_m) / train_cfg.grad_accum
                flat_g = flatten_params(grads_m)
                if acc is None:
                    acc = {k: v / train_cfg.grad_accum for k, v in flat_g.items()}
                else:
                    for k, v in flat_g.items():
                        acc[k] = acc[k] + v / train_cfg.grad_accum
            assert acc is not None
            grads, gnorm = clip_grad_norm(unflatten_params(acc), train_cfg.grad_clip)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            dt = time.perf_counter() - t0

            if not math.isfinite(loss_val):
                raise RuntimeError(
                    f"loss is NaN/inf at step {step}; last good checkpoint: {last_ckpt}"
                )
            if step % log_every == 0 or step == end - 1:
                logger.scalar("train/loss", loss_val, step)
                logger.scalar("train/lr", lr, step)
                logger.scalar("train/grad_norm", gnorm, step)
                logger.scalar("train/tokens_per_s", tokens_per_step / dt, step)
                logger.scalar("train/step_ms", dt * 1000, step)
                logger.scalar("sys/peak_mem_mb", peak_memory_mb(), step)
                print(
                    f"step {step:5d} loss {loss_val:.4f} lr {lr:.2e} "
                    f"gnorm {gnorm:.2f} {tokens_per_step / dt:,.0f} tok/s"
                )
            if (step + 1) % train_cfg.eval_interval == 0 or step == end - 1:
                val_loss = evaluate(model, val_sampler, train_cfg.eval_iters)
                ppl = math.exp(min(val_loss, 20.0))
                logger.scalar("val/loss", val_loss, step)
                logger.scalar("val/ppl", ppl, step)
                print(f"  eval loss {val_loss:.4f} ppl {ppl:.1f}")
                last_ckpt = save_checkpoint(
                    ckpt_dir,
                    step + 1,
                    model,
                    opt,
                    sampler,
                    model_cfg,
                    train_cfg,
                    vocab_path,
                    val_loss,
                    keep_last,
                )
                if val_loss < best_val:
                    best_val = val_loss
                    mx.save_safetensors(
                        str(ckpt_dir / "best.safetensors"), flatten_params(dict(model.parameters()))
                    )
    except KeyboardInterrupt:
        # Ctrl-C: save what we've got so we can resume from here.
        # The batch in flight gets skipped on resume (tiny, we live with it).
        print(f"\ninterrupted at step {step}; saving resume checkpoint...")
        save_checkpoint(
            ckpt_dir,
            step,
            model,
            opt,
            sampler,
            model_cfg,
            train_cfg,
            vocab_path,
            None,
            keep_last,
        )
        logger.close()
        mx.eval(model.parameters())
        print(f"resume with: --resume {ckpt_dir}/step_{step:06d} (same preset/seed/batch)")
        return {"steps": step, "best_val_loss": best_val, "last_loss": loss_val}
    logger.close()
    mx.eval(model.parameters())
    return {"steps": end, "best_val_loss": best_val, "last_loss": loss_val}


def evaluate(model: GPT, val_sampler: BatchSampler, iters: int) -> float:
    losses = []
    for _ in range(iters):
        x, y = val_sampler.sample(8)
        loss = ce_loss(model, x, y)
        mx.eval(loss)
        losses.append(float(loss))
    return sum(losses) / len(losses)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train nanoMLX GPT on .bin data")
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    train(
        preset(args.preset, vocab_size=args.vocab_size),
        train_preset(args.preset, total_steps=args.steps, batch_size=args.batch, seed=args.seed),
        args.bin_dir,
        args.ckpt_dir,
        args.vocab,
        max_steps=args.steps,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
