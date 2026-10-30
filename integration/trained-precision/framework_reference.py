"""Separate same-model GPU diagnostics using ordinary framework operations.

Policy bits: 1 = compensated first normalization, 2 = compensated attention.
These are candidate model equations, not operator overrides or admitted arms.
They enable testing complete forward/backward propagation before source builds.
"""

import importlib.util
import math
import struct
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "gpu_compensated_math", Path(__file__).with_name("compensated_attention.py")
)
FF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FF)


def split_constant(value, like, ops):
    high = struct.unpack("f", struct.pack("f", value))[0]
    low = struct.unpack("f", struct.pack("f", value - high))[0]
    return ops.zeros_like(like) + high, ops.zeros_like(like) + low


def normalization(x, weight, bias, ops):
    zeros = ops.zeros_like(x)
    mean = FF.reduce_last((x, zeros), ops)
    mean = tuple(value[..., None] / x.shape[-1] for value in mean)
    centered = FF.add((x, zeros), tuple(-value for value in mean))
    variance = FF.reduce_last(FF.multiply(centered, centered), ops)
    variance = tuple(value[..., None] / x.shape[-1] for value in variance)
    variance = FF.add(variance, split_constant(1e-5, variance[0], ops))
    approximate = ops.rsqrt(variance[0] + variance[1])
    product = FF.multiply(variance, FF.product(approximate, approximate))
    residual = FF.add(
        (ops.zeros_like(approximate) + 1, ops.zeros_like(approximate)), (-product[0], -product[1])
    )
    correction = (0.5 * approximate) * (residual[0] + residual[1])
    inverse = FF.add(
        (approximate, ops.zeros_like(approximate)), (correction, ops.zeros_like(correction))
    )
    normal = FF.multiply(centered, inverse)
    affine = FF.add(
        FF.multiply(normal, (weight, ops.zeros_like(weight))), (bias, ops.zeros_like(bias))
    )
    return affine[0] + affine[1]


def forward(parameters, tokens, ops, policy):
    if policy not in range(4):
        raise ValueError("Unexpected framework diagnostic policy")
    p = parameters
    batch, length = tokens.shape
    width = p["token"].shape[-1]
    x = ops.embedding(tokens, p["token"]) + p["position"][:length]
    n1 = (
        normalization(x, p["ln1_weight"], p["ln1_bias"], ops)
        if policy & 1
        else ops.norm(x, p["ln1_weight"], p["ln1_bias"])
    )
    qkv = ops.linear(n1, p["qkv_weight"], p["qkv_bias"])
    q, k, v = [
        ops.swapaxes(part.reshape(batch, length, 4, width // 4), 1, 2)
        for part in ops.split_qkv(qkv)
    ]
    attention = FF.attention(q, k, v, ops)[0] if policy & 2 else ops.attention(q, k, v)
    attention = ops.swapaxes(attention, 1, 2).reshape(batch, length, width)
    r1 = x + ops.linear(attention, p["attention_weight"], p["attention_bias"])
    n2 = ops.norm(r1, p["ln2_weight"], p["ln2_bias"])
    hidden = ops.relu(ops.linear(n2, p["ff1_weight"], p["ff1_bias"]))
    r2 = r1 + ops.linear(hidden, p["ff2_weight"], p["ff2_bias"])
    n3 = ops.norm(r2, p["ln3_weight"], p["ln3_bias"])
    return ops.linear(n3, p["head_weight"], p["head_bias"])


def torch_ops(torch, device):
    import types

    f = torch.nn.functional
    return types.SimpleNamespace(
        concatenate=lambda values, axis: torch.cat(values, dim=axis),
        zeros_like=torch.zeros_like,
        arange=lambda n: torch.arange(n, device=device),
        max=lambda value, axis, keepdims: torch.amax(value, dim=axis, keepdim=keepdims),
        where=torch.where,
        exp=torch.exp,
        swapaxes=torch.swapaxes,
        rsqrt=torch.rsqrt,
        embedding=f.embedding,
        norm=lambda x, weight, bias: f.layer_norm(x, (x.shape[-1],), weight, bias, 1e-5),
        linear=f.linear,
        split_qkv=lambda x: x.chunk(3, dim=-1),
        attention=lambda q, k, v: f.scaled_dot_product_attention(q, k, v, is_causal=True),
        relu=f.relu,
    )


def mlx_ops(mx):
    import types

    return types.SimpleNamespace(
        concatenate=mx.concatenate,
        zeros_like=mx.zeros_like,
        arange=mx.arange,
        max=mx.max,
        where=mx.where,
        exp=mx.exp,
        swapaxes=mx.swapaxes,
        rsqrt=mx.rsqrt,
        embedding=lambda tokens, weight: weight[tokens],
        norm=lambda x, weight, bias: mx.fast.layer_norm(x, weight, bias, 1e-5),
        linear=lambda x, weight, bias: mx.addmm(bias, x, weight.T),
        split_qkv=lambda x: mx.split(x, 3, axis=-1),
        attention=lambda q, k, v: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(q.shape[-1]), mask="causal"
        ),
        relu=lambda x: mx.maximum(x, 0),
    )
