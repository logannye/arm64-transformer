"""Two-float FP32 array arithmetic for explicit first-order precision APIs.

Finite moderate operands only. These generic equations contain no transformer,
checkpoint, vocabulary, width selection or trained-state special case.
"""

import math
import struct


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


def attention(q, k, v, ops, scale=None):
    """[B,H,T,D], causal, no dropout. Return FP32 output and probabilities."""
    if q.shape != k.shape or q.shape != v.shape or len(q.shape) != 4:
        raise ValueError("This diagnostic requires matching self-attention operands")
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
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


def split_constant(value, like, ops):
    high = struct.unpack("f", struct.pack("f", value))[0]
    low = struct.unpack("f", struct.pack("f", value - high))[0]
    return ops.zeros_like(like) + high, ops.zeros_like(like) + low


def normalization(x, weight, bias, ops, epsilon):
    zeros = ops.zeros_like(x)
    mean = reduce_last((x, zeros), ops)
    mean = tuple(value[..., None] / x.shape[-1] for value in mean)
    centered = add((x, zeros), tuple(-value for value in mean))
    variance = reduce_last(multiply(centered, centered), ops)
    variance = tuple(value[..., None] / x.shape[-1] for value in variance)
    variance = add(variance, split_constant(epsilon, variance[0], ops))
    approximate = ops.rsqrt(variance[0] + variance[1])
    variance_scale = multiply(variance, product(approximate, approximate))
    residual = add(
        (ops.zeros_like(approximate) + 1, ops.zeros_like(approximate)),
        (-variance_scale[0], -variance_scale[1]),
    )
    correction = (0.5 * approximate) * (residual[0] + residual[1])
    inverse = add(
        (approximate, ops.zeros_like(approximate)), (correction, ops.zeros_like(correction))
    )
    normal = multiply(centered, inverse)
    affine = add(multiply(normal, (weight, ops.zeros_like(weight))), (bias, ops.zeros_like(bias)))
    return affine[0] + affine[1]
