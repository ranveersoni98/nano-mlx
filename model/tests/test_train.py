"""Training loop tests: schedule, decay groups, compile matching eager,
checkpoint resume (fast, debug preset and tiny bins)."""

import mlx.core as mx
import pytest
from mlx import nn

from nanomlx.config import TrainConfig, preset
from nanomlx.model import GPT, tree_flatten
from nanomlx.train import ce_loss, train
from nanomlx.utils import build_optimizer, is_decay_param, lr_at


def test_lr_schedule_values():
    peak, warmup, total = 1e-3, 100, 1000
    assert lr_at(0, peak, warmup, total) == pytest.approx(peak / warmup)
    assert lr_at(99, peak, warmup, total) == pytest.approx(peak)
    mid = lr_at(550, peak, warmup, total)
    assert peak * 0.1 < mid < peak
    assert lr_at(999, peak, warmup, total) == pytest.approx(peak * 0.1, rel=0.05)
    assert lr_at(5000, peak, warmup, total) == pytest.approx(peak * 0.1)


def test_decay_groups_correct():
    """Zero grads: decay matrices shrink by (1-lr*wd), everything else stays put."""
    mx.random.seed(0)
    cfg = preset("debug", vocab_size=320)
    model = GPT(cfg)
    before = {k: v for k, v in tree_flatten(dict(model.parameters()))}
    mx.eval(before)
    opt = build_optimizer(1e-2, 0.1)
    zero = {k: mx.zeros_like(v) for k, v in before.items()}
    opt.update(model, zero)
    mx.eval(model.parameters())
    after = dict(tree_flatten(dict(model.parameters())))
    for k, b in before.items():
        a = after[k]
        if is_decay_param(k, b):
            expected = b * (1 - 1e-2 * 0.1)
            assert bool(mx.allclose(a, expected, atol=1e-7)), f"{k} missing decay"
        else:
            assert bool(mx.allclose(a, b, atol=1e-9)), f"{k} wrongly decayed: {k}"
    # Spot-check the rule.
    assert is_decay_param("blocks.0.attn.q_proj.weight", before["blocks.0.attn.q_proj.weight"])
    assert not is_decay_param("tok_emb.weight", before["tok_emb.weight"])
    assert not is_decay_param("blocks.0.attn_norm.weight", before["blocks.0.attn_norm.weight"])


def test_compiled_uncompiled_equivalence():
    mx.random.seed(1)
    cfg = preset("debug", vocab_size=320)
    m1, m2 = GPT(cfg), GPT(cfg)
    # Same init: copy weights.
    m2.update(dict(m1.parameters()))
    opt = build_optimizer(1e-3)
    x = mx.random.randint(0, 320, (2, 8))
    y = mx.random.randint(0, 320, (2, 8))
    mx.eval(x, y)

    def run(model, compiled: bool):
        lag = nn.value_and_grad(model, ce_loss)

        def micro(x, y):
            return lag(model, x, y)

        fn = (
            mx.compile(micro, inputs=[model.state, opt.state], outputs=[model.state, opt.state])
            if compiled
            else micro
        )
        loss, grads = fn(x, y)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        return float(loss)

    l_eager = run(m1, False)
    # Fresh optimizer for the compiled side (staging order isn't the same).
    opt2 = build_optimizer(1e-3)
    lag = nn.value_and_grad(m2, ce_loss)

    def micro2(x, y):
        return lag(m2, x, y)

    fn = mx.compile(micro2, inputs=[m2.state, opt2.state], outputs=[m2.state, opt2.state])
    loss, grads = fn(x, y)
    opt2.update(m2, grads)
    mx.eval(loss, m2.parameters(), opt2.state)
    assert abs(l_eager - float(loss)) < 1e-4


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Tiny 4-step run with evals and checkpoints. The resume test reuses it."""
    from pathlib import Path

    from nanomlx.data import build_bins

    tmp = tmp_path_factory.mktemp("tiny_run")
    docs = ["hello world " * 20 + "\n"] * 60 + ["foo bar baz\n"] * 30
    (tmp / "train.txt").write_text("".join(docs), encoding="utf-8")
    (tmp / "val.txt").write_text("hello world test\n" * 30, encoding="utf-8")
    vocab = Path(__file__).resolve().parents[1] / "model" / "tests" / "fixtures" / "vocab.json"
    if not vocab.exists():
        vocab = Path("model/tests/fixtures/vocab.json")
    bins = tmp / "bins"
    build_bins(vocab, tmp / "train.txt", tmp / "val.txt", bins)
    mcfg = preset("debug", vocab_size=320, context=32)
    # NOTE: the 4-step bit uses the SAME 6-step schedule (total_steps=6) and
    # just stops early with max_steps. LR depends on total, so a plain
    # total_steps=4 run would use different LRs at steps 2-3 and could never
    # match the straight 6-step run.
    tcfg = TrainConfig(
        preset="debug",
        peak_lr=3e-3,
        warmup_steps=2,
        total_steps=6,
        batch_size=4,
        eval_interval=2,
        eval_iters=2,
        seed=0,
    )
    ckpt = tmp / "ckpt"
    summary = train(mcfg, tcfg, bins, ckpt, vocab, max_steps=4)
    return tmp, bins, vocab, mcfg, tcfg, ckpt, summary


def test_short_run_logs_and_checkpoints(tiny_run):
    _, _, _, _, _, ckpt, summary = tiny_run
    assert (ckpt / "step_000004").exists()
    assert (ckpt / "logs" / "metrics.jsonl").exists()
    assert summary["best_val_loss"] < float("inf")
    meta = (ckpt / "step_000004" / "meta.json").read_text(encoding="utf-8")
    assert "tokenizer_pattern_hash" in meta and "config" not in meta


def test_resume_matches_uninterrupted(tiny_run):
    """Train 4, resume to 6, should equal a straight 6 (weights allclose)."""
    import json

    tmp, bins, vocab, mcfg, tcfg, ckpt, _ = tiny_run
    straight_dir = tmp / "straight"
    train(mcfg, tcfg, bins, straight_dir, vocab, max_steps=6)
    w_straight = mx.load(str(straight_dir / "step_000006" / "weights.safetensors"))

    resume_dir = tmp / "resumed"
    train(
        mcfg,
        tcfg,
        bins,
        resume_dir,
        vocab,
        max_steps=6,
        resume=ckpt / "step_000004",
    )
    w_resumed = mx.load(str(resume_dir / "step_000006" / "weights.safetensors"))
    worst = 0.0
    for k in w_straight:
        d = float(mx.max(mx.abs(w_straight[k] - w_resumed[k])))
        worst = max(worst, d)
    print(f"\nresume max param drift: {worst:.2e}")
    assert worst < 1e-4, f"resumed run drifted: {worst}"
    assert json.loads((resume_dir / "step_000006" / "meta.json").read_text())["step"] == 6
