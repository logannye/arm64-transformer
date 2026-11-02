"""Prospective Python CPU candidate: FP64 arithmetic, FP32 saved tensors.

This is a separately named model implementation, not the ordinary PyTorch arm.
It uses standard PyTorch tensor operations and explicit first-order derivatives
for QKV and causal attention. No operator registration, monkeypatch or native
extension is installed. Qualification and timing remain separate requirements.
"""

import math

import torch
from torch.nn import functional as F

from transformer.reference import Transformer
from transformer.spec import EPS, HEADS, WIDTH

IDENTITY = "python-fp32-saved-qkv-attention-v1"


def require_cpu_float(*values):
    if any(value.device.type != "cpu" or value.dtype != torch.float32 for value in values):
        raise ValueError("Selective precision requires CPU FP32 tensors")


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        require_cpu_float(x, weight, bias)
        ctx.save_for_backward(x, weight)
        return F.linear(x.double(), weight.double(), bias.double()).float()

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        require_cpu_float(dy)
        d = dy.double()
        dx = d @ weight.double()
        dw = d.reshape(-1, d.shape[-1]).T @ x.double().reshape(-1, x.shape[-1])
        db = d.reshape(-1, d.shape[-1]).sum(0)
        return dx.float(), dw.float(), db.float()


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        require_cpu_float(q, k, v)
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("Expected matching [batch, heads, length, width] tensors")
        scale = 1 / math.sqrt(q.shape[-1])
        length = q.shape[-2]
        causal = torch.ones(length, length, dtype=torch.bool).triu(1)
        scores = (q.double() @ k.double().transpose(-2, -1)) * scale
        probability = scores.masked_fill(causal, -torch.inf).softmax(-1)
        ctx.save_for_backward(q, k, v, probability.float())
        ctx.scale = scale
        return (probability @ v.double()).float()

    @staticmethod
    def backward(ctx, dout):
        q, k, v, probability = ctx.saved_tensors
        require_cpu_float(dout)
        p, d = probability.double(), dout.double()
        dp = d @ v.double().transpose(-2, -1)
        ds = p * (dp - (dp * p).sum(-1, keepdim=True)) * ctx.scale
        return (
            (ds @ k.double()).float(),
            (ds.transpose(-2, -1) @ q.double()).float(),
            (p.transpose(-2, -1) @ d).float(),
        )


class PrecisionReference(Transformer):
    """The fixed model, changing only QKV and attention arithmetic/saved state."""

    def forward(self, tokens):
        require_cpu_float(self.token)
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= 32:
            raise ValueError("Expected [batch, length] tokens with length <= 32")
        batch, length = tokens.shape
        x = F.embedding(tokens, self.token) + self.position[:length]
        n1 = F.layer_norm(x, (WIDTH,), self.ln1_weight, self.ln1_bias, EPS)
        qkv = Linear.apply(n1, self.qkv_weight, self.qkv_bias)
        q, k, v = [
            part.reshape(batch, length, HEADS, WIDTH // HEADS).transpose(1, 2)
            for part in qkv.chunk(3, dim=-1)
        ]
        attention = Attention.apply(q, k, v).transpose(1, 2).reshape(batch, length, WIDTH)
        r1 = x + F.linear(attention, self.attention_weight, self.attention_bias)
        n2 = F.layer_norm(r1, (WIDTH,), self.ln2_weight, self.ln2_bias, EPS)
        hidden = F.relu(F.linear(n2, self.ff1_weight, self.ff1_bias))
        r2 = r1 + F.linear(hidden, self.ff2_weight, self.ff2_bias)
        n3 = F.layer_norm(r2, (WIDTH,), self.ln3_weight, self.ln3_bias, EPS)
        return F.linear(n3, self.head_weight, self.head_bias)
