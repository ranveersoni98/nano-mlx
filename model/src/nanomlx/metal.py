"""Our own Metal kernels: fused RMSNorm for now.

One thread per row. We accumulate in float on purpose so fp32 and bf16
inputs match what the hand-written path does. The kernel compiles lazily
on first use (first call costs ~100ms of Metal compile, so warm it up
and throw that timing away in benchmarks).
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL = None

_SOURCE = """
    uint row = thread_position_in_grid.x;
    if (row >= N[0]) return;
    uint Dv = D[0];
    // Vectorized main loop (float4) + scalar tail.
    float4 acc4 = float4(0.0f);
    uint j = 0;
    for (; j + 4 <= Dv; j += 4) {
        float4 v = *((device float4*)(x + row * Dv + j));
        acc4 += v * v;
    }
    float acc = acc4.x + acc4.y + acc4.z + acc4.w;
    for (; j < Dv; ++j) {
        float v = x[row * Dv + j];
        acc += v * v;
    }
    float inv = metal::rsqrt(acc / (float)Dv + eps[0]);
    for (j = 0; j + 4 <= Dv; j += 4) {
        float4 v = *((device float4*)(x + row * Dv + j));
        float4 s = *((device float4*)(w + j));
        *((device float4*)(out + row * Dv + j)) = v * inv * s;
    }
    for (; j < Dv; ++j) {
        out[row * Dv + j] = x[row * Dv + j] * inv * w[j];
    }
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="nanomlx_fused_rmsnorm",
            input_names=["x", "w", "D", "eps", "N"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _KERNEL


def fused_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMSNorm in one fused kernel. x is (..., D) (any batch shape, rows
    of D have to be contiguous). weight is (D,). Same shape/dtype out as in.

    The kernel only does fp32 (Metal won't silently store float as bfloat):
    anything else gets upcast, normalized, and cast back. Same numbers as
    the hand-written path, which also works in fp32.
    """
    if x.shape[-1] != weight.shape[0]:
        raise TypeError(f"rmsnorm dim mismatch: x {x.shape} vs weight {weight.shape}")
    dtype = x.dtype
    if dtype != mx.float32:
        x, weight = x.astype(mx.float32), weight.astype(mx.float32)
    xc = mx.contiguous(x.reshape(-1, x.shape[-1]))
    wc = mx.contiguous(weight)
    nrows = xc.shape[0]
    dim = mx.array([xc.shape[1]], dtype=mx.uint32)
    e = mx.array([eps], dtype=mx.float32)
    n = mx.array([nrows], dtype=mx.uint32)
    (out,) = _kernel()(
        inputs=[xc, wc, dim, e, n],
        grid=(nrows, 1, 1),
        threadgroup=(min(256, max(32, nrows)), 1, 1),
        output_shapes=[xc.shape],
        output_dtypes=[mx.float32],
    )
    out = out.reshape(x.shape)
    return out if dtype == mx.float32 else out.astype(dtype)
