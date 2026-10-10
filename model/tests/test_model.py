"""M4 acceptance: MLX model sanity (AGENTS.md §4 M4 + §10 matrix).

Vocab comes from our own tokenizer: the 320-id fixture vocab grounds
`vocab_size` (frozen default 32k is used in real training, not tests).
"""

import mlx.core as mx
import mlx.optimizers as optim
import pytest
from mlx import nn

from nanomlx.config import ModelConfig, preset
from nanomlx.model import GPT, CausalSelfAttention

VOCAB = 320  # model/tests/fixtures/vocab.json (our tokenizer)


@pytest.fixture()
def debug_cfg() -> ModelConfig:
    mx.random.seed(0)
    return preset("debug", vocab_size=VOCAB)


def _ids(batch: int, length: int, seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    return mx.random.randint(0, VOCAB, (batch, length))


def test_forward_shapes(debug_cfg: ModelConfig):
    model = GPT(debug_cfg)
    for batch, length in [(1, 1), (2, 8), (2, 128), (4, 32)]:
        logits = model(_ids(batch, length))
        mx.eval(logits)
        assert logits.shape == (batch, length, VOCAB), logits.shape


def test_causality(debug_cfg: ModelConfig):
    """Changing token t must not move logits at positions < t."""
    model = GPT(debug_cfg)
    base = _ids(2, 16)
    altered = mx.array(base.tolist())
    altered = mx.where(mx.arange(16)[None, :] >= 10, (altered + 7) % VOCAB, altered)
    l0, l1 = model(base), model(altered)
    mx.eval(l0, l1)
    assert mx.allclose(l0[:, :10, :], l1[:, :10, :], atol=1e-5)
    assert not mx.allclose(l0[:, 10:, :], l1[:, 10:, :], atol=1e-5)


def test_param_count_matches_formula(debug_cfg: ModelConfig):
    model = GPT(debug_cfg)
    total, no_emb = model.count_params()
    assert total == debug_cfg.param_count(True), (total, debug_cfg.param_count(True))
    assert no_emb == debug_cfg.param_count(False)
    print(f"\ndebug preset: {total:,} total, {no_emb:,} without embeddings")


def test_param_count_untied():
    cfg = preset("debug", vocab_size=VOCAB, tie_weights=False)
    model = GPT(cfg)
    total, _ = model.count_params()
    assert total == cfg.param_count(True)
    assert total == preset("debug", vocab_size=VOCAB).param_count(True) + VOCAB * 64


def test_grads_finite_and_nonzero(debug_cfg: ModelConfig):
    model = GPT(debug_cfg)
    x, y = _ids(2, 16, seed=1), _ids(2, 16, seed=2)

    def loss_fn(m: GPT) -> mx.array:
        logits = m(x).reshape(-1, VOCAB)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.mean(logp[mx.arange(y.size), y.reshape(-1)])

    _, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(grads)
    flat: list[tuple[str, mx.array]] = []

    def walk(tree, prefix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(tree, list):
            for i, v in enumerate(tree):
                walk(v, f"{prefix}{i}.")
        else:
            flat.append((prefix, tree))

    walk(grads)
    assert flat, "no parameters found"
    for name, g in flat:
        assert mx.all(mx.isfinite(g)), f"non-finite grad: {name}"
        assert float(mx.max(mx.abs(g))) > 0, f"dead parameter (zero grad): {name}"


def test_overfit_one_batch(debug_cfg: ModelConfig):
    """Tiny config must drive a single fixed batch below 0.05 (M4 gate).
    If this fails the model/optimizer is wrong - do not proceed to M5."""
    mx.random.seed(7)
    model = GPT(debug_cfg)
    opt = optim.AdamW(learning_rate=3e-3, betas=(0.9, 0.95), eps=1e-8)
    x = _ids(8, 32, seed=11)
    y = _ids(8, 32, seed=12)

    def loss_fn(m: GPT) -> mx.array:
        logits = m(x).reshape(-1, VOCAB)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.mean(logp[mx.arange(y.size), y.reshape(-1)])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    final = float("inf")
    for step in range(500):
        loss, grads = loss_and_grad(model)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        final = float(loss)
        if final < 0.05:
            break
    print(f"\noverfit: loss={final:.4f} at step={step}")
    assert final < 0.05, f"failed to overfit single batch (loss={final:.4f})"


def test_gqa_runs_and_matches_tiled_reference():
    """GQA (kv < heads) must equal full-head attention with tiled KV weights."""
    mx.random.seed(3)
    cfg = ModelConfig(vocab_size=VOCAB, n_layers=1, d_model=32, n_heads=4, n_kv_heads=2, context=16)
    gqa = CausalSelfAttention(cfg)
    ref_cfg = ModelConfig(
        vocab_size=VOCAB, n_layers=1, d_model=32, n_heads=4, n_kv_heads=4, context=16
    )
    ref = CausalSelfAttention(ref_cfg)
    # Tile GQA's k/v/o weights into the full-head reference. Linear weights
    # are (out, in): reshape to (kv_heads, head_dim, in) so the repeat
    # duplicates whole heads ([kv0,kv0,kv1,kv1]), not individual rows.
    ref.q_proj.weight = gqa.q_proj.weight
    kv, hd, d = 2, cfg.head_dim, cfg.d_model
    ref.k_proj.weight = mx.repeat(gqa.k_proj.weight.reshape(kv, hd, d), 2, axis=0).reshape(
        4 * hd, d
    )
    ref.v_proj.weight = mx.repeat(gqa.v_proj.weight.reshape(kv, hd, d), 2, axis=0).reshape(
        4 * hd, d
    )
    ref.o_proj.weight = gqa.o_proj.weight
    x = mx.random.normal((2, 12, 32))
    a, b = gqa(x), ref(x)
    mx.eval(a, b)
    assert a.shape == (2, 12, 32)
    assert mx.allclose(a, b, atol=1e-5), float(mx.max(mx.abs(a - b)))
