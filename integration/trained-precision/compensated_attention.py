"""Development-only two-float attention using ordinary GPU framework operations.

This same-input diagnostic is not a replacement framework arm. There is no
custom operator, native extension, CPU tensor fallback or change to stored dtype.
Error decompositions assume rounding at each declared FP32 operation. The original
bounded same-input probe uses finite moderate operands without compilation.
Compiled full-model candidates have separate graph, numerical and replay checks;
an uncompiled pass does not qualify a fused graph.
"""

import math


def add(a, b):
    high = a[0] + b[0]
    virtual = high - a[0]
    error = (a[0] - (high - virtual)) + (b[0] - virtual)
    tail = (a[1] + b[1]) + error
    total = high + tail
    residual = total - high
    return total, (high - (total - residual)) + (tail - residual)


def product(a, b):
    # Dekker splitting, since ordinary tensor APIs expose no guaranteed FMA.
    split_a, split_b = 4097.0 * a, 4097.0 * b
    ah, bh = split_a - (split_a - a), split_b - (split_b - b)
    al, bl = a - ah, b - bh
    high = a * b
    low = ((ah * bh - high) + ah * bl + al * bh) + al * bl
    return high, low


def multiply(a, b):
    return add(add(product(a[0], b[0]), product(a[0], b[1])), product(a[1], b[0]))


def reduce_last(pair, ops):
    high, low = pair
    while high.shape[-1] > 1:
        if high.shape[-1] % 2:
            high = ops.concatenate((high, ops.zeros_like(high[..., :1])), axis=-1)
            low = ops.concatenate((low, ops.zeros_like(low[..., :1])), axis=-1)
        high, low = add((high[..., ::2], low[..., ::2]), (high[..., 1::2], low[..., 1::2]))
    return high[..., 0], low[..., 0]


def attention(q, k, v, ops):
    """[B,H,T,D], causal, no dropout. Return FP32 output and probabilities."""
    if q.shape != k.shape or q.shape != v.shape or len(q.shape) != 4:
        raise ValueError("This diagnostic requires matching self-attention operands")
    scale = 1 / math.sqrt(q.shape[-1])
    # Split the same scalar used by the FP64 mathematical equation.
    import struct

    hi = struct.unpack("f", struct.pack("f", scale))[0]
    lo = struct.unpack("f", struct.pack("f", scale - hi))[0]
    score = reduce_last(product(q[..., :, None, :], k[..., None, :, :]), ops)
    score = multiply(score, (ops.zeros_like(score[0]) + hi, ops.zeros_like(score[0]) + lo))
    length = q.shape[-2]
    causal = ops.arange(length)[:, None] >= ops.arange(length)[None, :]
    maximum = ops.max(ops.where(causal, score[0], -math.inf), axis=-1, keepdims=True)
    centered = add(score, (-maximum, 0.0))
    exponent = ops.exp(ops.where(causal, centered[0], -math.inf))
    exponent = multiply((exponent, 0.0), (1.0, ops.where(causal, centered[1], 0.0)))
    denominator = reduce_last(exponent, ops)
    dh, dl = (part[..., None] for part in denominator)
    approximate = exponent[0] / dh
    residual = add(exponent, tuple(-part for part in product(approximate, dh)))
    residual = add(residual, tuple(-part for part in product(approximate, dl)))
    probability = approximate + (residual[0] + residual[1]) / dh
    output = reduce_last(
        product(probability[..., :, None, :], ops.swapaxes(v, -1, -2)[..., None, :, :]), ops
    )
    return output[0] + output[1], probability
