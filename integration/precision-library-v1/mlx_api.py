"""Experimental FP32 precision operations for a maintained MLX GPU package.

Intended for mlx.nn.precise_fp32. Ordinary core/fast operators are unchanged.
"""

import math
from types import SimpleNamespace

import mlx.core as mx

from . import _fp32_precise_dense as DENSE
from . import _fp32_precise_math as PAIR

IDENTITY = "fp32-saved-precision-api-v1"
OPS = SimpleNamespace(
    concatenate=mx.concatenate,
    zeros_like=mx.zeros_like,
    arange=mx.arange,
    max=mx.max,
    where=mx.where,
    exp=mx.exp,
    rsqrt=mx.rsqrt,
    swapaxes=mx.swapaxes,
)
_LINEAR = DENSE.mlx_function(mx, OPS)


def _require(*values):
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError("The MLX precision API requires an available default GPU")
    if any(not isinstance(value, mx.array) or value.dtype != mx.float32 for value in values):
        raise ValueError("Precision APIs require FP32 arrays")


def linear(x, weight, bias=None):
    """[..., input] by [output, input], using bounded two-float dense arithmetic."""
    _require(x, weight)
    if (
        x.ndim < 1
        or weight.ndim != 2
        or min(weight.shape) < 1
        or x.shape[-1] != weight.shape[-1]
        or x.size == 0
    ):
        raise ValueError("Expected nonempty compatible dense operands")
    if bias is None:
        bias = mx.zeros((weight.shape[0],), dtype=mx.float32)
    _require(x, weight, bias)
    if bias.shape != (weight.shape[0],):
        raise ValueError("Expected a bias vector matching output channels")
    return _LINEAR(x, weight, bias)


def layer_norm(x, weight=None, bias=None, epsilon=1e-5):
    """Last-axis population normalization with FP32 arrays throughout."""
    _require(x)
    if x.ndim < 1 or x.size == 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Expected nonempty input and finite positive epsilon")
    if weight is None:
        weight = mx.ones((x.shape[-1],), dtype=mx.float32)
    if bias is None:
        bias = mx.zeros((x.shape[-1],), dtype=mx.float32)
    _require(x, weight, bias)
    if weight.shape != (x.shape[-1],) or bias.shape != weight.shape:
        raise ValueError("LayerNorm affine vectors must match the last axis")
    return PAIR.normalization(x, weight, bias, OPS, float(epsilon))


def scaled_dot_product_attention(q, k, v, *, scale=None, dropout_p=0.0, is_causal=True):
    """Matching 4D causal self-attention, without dropout or CPU tensor fallback."""
    _require(q, k, v)
    if (
        q.ndim != 4
        or q.shape != k.shape
        or q.shape != v.shape
        or min(q.shape) < 1
        or dropout_p != 0
        or is_causal is not True
    ):
        raise ValueError("Expected nonempty matching 4D causal self-attention without dropout")
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("Attention scale must be finite")
    return PAIR.attention(q, k, v, OPS, scale=scale)[0]
