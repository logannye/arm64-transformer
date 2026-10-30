"""Prospective first-LayerNorm refinement of the FP32-saved Python candidate.

The separately retained v1 candidate has a position-gradient regression. This
version adds FP64 LayerNorm arithmetic with FP32 saved normalized values and
inverse standard deviations, matching the native candidate's storage contract.
"""

import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F

from transformer.spec import EPS, HEADS, WIDTH

SPEC = importlib.util.spec_from_file_location(
    "fp32_saved_v1", Path(__file__).with_name("precision_reference.py")
)
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)
IDENTITY = "python-fp32-saved-ln1-qkv-attention-v2"


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        V1.require_cpu_float(x, weight, bias)
        centered = x.double() - x.double().mean(-1, keepdim=True)
        inverse = torch.rsqrt(centered.square().mean(-1, keepdim=True) + EPS)
        normalized = centered * inverse
        ctx.save_for_backward(normalized.float(), inverse.float(), weight)
        return (normalized * weight.double() + bias.double()).float()

    @staticmethod
    def backward(ctx, dy):
        V1.require_cpu_float(dy)
        normalized, inverse, weight = (value.double() for value in ctx.saved_tensors)
        upstream = dy.double()
        dn = upstream * weight
        dx = inverse * (
            dn - dn.mean(-1, keepdim=True) - normalized * (dn * normalized).mean(-1, keepdim=True)
        )
        axes = tuple(range(upstream.ndim - 1))
        dw = (upstream * normalized).sum(axes)
        db = upstream.sum(axes)
        return dx.float(), dw.float(), db.float()


class PrecisionReference(V1.PrecisionReference):
    def forward(self, tokens):
        V1.require_cpu_float(self.token)
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= 32:
            raise ValueError("Expected [batch, length] tokens with length <= 32")
        batch, length = tokens.shape
        x = F.embedding(tokens, self.token) + self.position[:length]
        n1 = LayerNorm.apply(x, self.ln1_weight, self.ln1_bias)
        qkv = V1.Linear.apply(n1, self.qkv_weight, self.qkv_bias)
        q, k, v = [
            part.reshape(batch, length, HEADS, WIDTH // HEADS).transpose(1, 2)
            for part in qkv.chunk(3, dim=-1)
        ]
        attention = V1.Attention.apply(q, k, v).transpose(1, 2).reshape(batch, length, WIDTH)
        r1 = x + F.linear(attention, self.attention_weight, self.attention_bias)
        n2 = F.layer_norm(r1, (WIDTH,), self.ln2_weight, self.ln2_bias, EPS)
        hidden = F.relu(F.linear(n2, self.ff1_weight, self.ff1_bias))
        r2 = r1 + F.linear(hidden, self.ff2_weight, self.ff2_bias)
        n3 = F.layer_norm(r2, (WIDTH,), self.ln3_weight, self.ln3_bias, EPS)
        return F.linear(n3, self.head_weight, self.head_bias)
