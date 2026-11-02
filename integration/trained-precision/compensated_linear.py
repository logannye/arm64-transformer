"""Tiled two-float dense diagnostic with FP32 state and explicit first derivatives.

Uses ordinary tensor operations on the selected device. It is not a framework
operator replacement. Tiling bounds temporary products; custom derivatives avoid
saving every product/reduction intermediate for the backward pass.
"""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "linear_pair_math", Path(__file__).with_name("compensated_attention.py")
)
PAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIR)
ROW_TILE = 32
OUTPUT_TILE = 64


def dense(x, weight, bias, ops):
    """[..., input] times [output, input], accumulating bias before final rounding."""
    if len(weight.shape) != 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("Dense diagnostic dimensions differ")
    original = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    rows = []
    for start in range(0, flat.shape[0], ROW_TILE):
        columns = []
        value = flat[start : start + ROW_TILE]
        for column in range(0, weight.shape[0], OUTPUT_TILE):
            part = weight[column : column + OUTPUT_TILE]
            total = PAIR.reduce_last(PAIR.product(value[:, None, :], part[None, :, :]), ops)
            if bias is not None:
                total = PAIR.add(total, (bias[column : column + OUTPUT_TILE], 0.0))
            columns.append(total[0] + total[1])
        rows.append(ops.concatenate(columns, axis=-1))
    return ops.concatenate(rows, axis=0).reshape(*original, weight.shape[0])


def backward(x, weight, dy, ops):
    flat, upstream = x.reshape(-1, x.shape[-1]), dy.reshape(-1, dy.shape[-1])
    dx = dense(dy, ops.swapaxes(weight, -1, -2), None, ops)
    dw = dense(ops.swapaxes(upstream, -1, -2), ops.swapaxes(flat, -1, -2), None, ops)
    transposed = ops.swapaxes(upstream, -1, -2)
    bias = PAIR.reduce_last((transposed, ops.zeros_like(transposed)), ops)
    return dx, dw, bias[0] + bias[1]


def torch_function(torch, ops):
    class Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias):
            if any(value.dtype != torch.float32 for value in (x, weight, bias)):
                raise ValueError("Compensated dense diagnostic requires FP32")
            ctx.save_for_backward(x, weight)
            return dense(x, weight, bias, ops)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, dy):
            return backward(*ctx.saved_tensors, dy, ops)

    return Linear.apply


def mlx_function(mx, ops):
    @mx.custom_function
    def linear(x, weight, bias):
        if any(value.dtype != mx.float32 for value in (x, weight, bias)):
            raise ValueError("Compensated dense diagnostic requires FP32")
        return dense(x, weight, bias, ops)

    @linear.vjp
    def linear_vjp(primals, cotangent, output):
        x, weight, _ = primals
        return backward(x, weight, cotangent, ops)

    return linear
