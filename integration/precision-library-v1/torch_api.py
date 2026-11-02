"""Experimental FP32-saved precision operations for CPU and MPS.

This source is intended for torch.nn.precise_fp32 in a separately authenticated
maintained package. It does not replace the ordinary functional operators.
"""

import math
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from . import _fp32_precise_dense as DENSE
from . import _fp32_precise_math as PAIR

IDENTITY = "fp32-saved-precision-api-v1"
_DENSE_OPS = SimpleNamespace(
    concatenate=lambda values, axis: torch.cat(values, dim=axis),
    zeros_like=torch.zeros_like,
    swapaxes=torch.swapaxes,
)
_MPS_LINEAR = DENSE.torch_function(torch, _DENSE_OPS)


def _require(*values):
    if not values or any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.device.type not in {"cpu", "mps"}
        or value.device != values[0].device
        for value in values
    ):
        raise ValueError("Precision APIs require FP32 tensors on one CPU or MPS device")


def _ops(anchor):
    device = anchor.device
    return SimpleNamespace(
        concatenate=lambda values, axis: torch.cat(values, dim=axis),
        zeros_like=torch.zeros_like,
        arange=lambda n: torch.arange(n, device=device),
        max=lambda x, axis, keepdims: torch.amax(x, dim=axis, keepdim=keepdims),
        where=torch.where,
        exp=torch.exp,
        rsqrt=torch.rsqrt,
        swapaxes=torch.swapaxes,
    )


class _CPULinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        return F.linear(x.double(), weight.double(), bias.double()).float()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        d = dy.double()
        dx = d @ weight.double()
        flat = d.reshape(-1, d.shape[-1])
        dw = flat.T @ x.double().reshape(-1, x.shape[-1])
        return dx.float(), dw.float(), flat.sum(0).float()


def linear(x, weight, bias=None):
    """[..., input] by [output, input]; only FP32 state crosses the operation."""
    _require(x, weight)
    if (
        x.ndim < 1
        or weight.ndim != 2
        or min(weight.shape) < 1
        or x.shape[-1] != weight.shape[-1]
        or x.numel() == 0
    ):
        raise ValueError("Expected nonempty compatible dense operands")
    if bias is None:
        bias = x.new_zeros(weight.shape[0])
    _require(x, weight, bias)
    if bias.shape != (weight.shape[0],):
        raise ValueError("Expected a bias vector matching output channels")
    if x.device.type == "cpu":
        return _CPULinear.apply(x, weight, bias)
    return _MPS_LINEAR(x, weight, bias)


class _CPULayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, epsilon):
        centered = x.double() - x.double().mean(-1, keepdim=True)
        inverse = torch.rsqrt(centered.square().mean(-1, keepdim=True) + epsilon)
        normalized = centered * inverse
        ctx.save_for_backward(normalized.float(), inverse.float(), weight)
        return (normalized * weight.double() + bias.double()).float()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        normalized, inverse, weight = (x.double() for x in ctx.saved_tensors)
        upstream = dy.double()
        dn = upstream * weight
        dx = inverse * (
            dn - dn.mean(-1, keepdim=True) - normalized * (dn * normalized).mean(-1, keepdim=True)
        )
        axes = tuple(range(upstream.ndim - 1))
        dw = (upstream * normalized).sum(axes) if axes else upstream * normalized
        db = upstream.sum(axes) if axes else upstream
        return dx.float(), dw.float(), db.float(), None


def layer_norm(x, weight=None, bias=None, epsilon=1e-5):
    """Normalize the last axis, with population variance and positive epsilon."""
    _require(x)
    if x.ndim < 1 or x.numel() == 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Expected nonempty input and finite positive epsilon")
    if weight is None:
        weight = x.new_ones(x.shape[-1])
    if bias is None:
        bias = x.new_zeros(x.shape[-1])
    _require(x, weight, bias)
    if weight.shape != (x.shape[-1],) or bias.shape != weight.shape:
        raise ValueError("LayerNorm affine vectors must match the last axis")
    if x.device.type == "cpu":
        return _CPULayerNorm.apply(x, weight, bias, float(epsilon))
    return PAIR.normalization(x, weight, bias, _ops(x), float(epsilon))


class _CPUAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        length = q.shape[-2]
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        scores = (q.double() @ k.double().transpose(-2, -1)) * scale
        probability = scores.masked_fill(mask, -torch.inf).softmax(-1)
        ctx.save_for_backward(q, k, v, probability.float())
        ctx.scale = scale
        return (probability @ v.double()).float()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        q, k, v, probability = ctx.saved_tensors
        p, d = probability.double(), dy.double()
        dp = d @ v.double().transpose(-2, -1)
        ds = p * (dp - (dp * p).sum(-1, keepdim=True)) * ctx.scale
        return (
            (ds @ k.double()).float(),
            (ds.transpose(-2, -1) @ q.double()).float(),
            (p.transpose(-2, -1) @ d).float(),
            None,
        )


def scaled_dot_product_attention(q, k, v, *, scale=None, dropout_p=0.0, is_causal=True):
    """Matching [batch, heads, time, width] causal self-attention, zero dropout."""
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
    if q.device.type == "cpu":
        return _CPUAttention.apply(q, k, v, scale)
    return PAIR.attention(q, k, v, _ops(q), scale=scale)[0]
