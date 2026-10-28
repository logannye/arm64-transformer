"""Ordinary batched PyTorch reference for the fixed transformer.

The native implementation shares the parameter layout, not these calculations.
Normal execution uses PyTorch's linear, layer_norm, scaled_dot_product_attention
and autograd kernels. Expanded attention probabilities exist only in audit traces.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from transformer.spec import CONTEXT, EPS, HEADS, PARAM_COUNT, PARAM_SHAPES, PARAMS, WIDTH


class Transformer(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for name, shape in PARAM_SHAPES.items():
            if name.endswith("bias"):
                values = torch.zeros(shape, dtype=torch.float32)
            elif name.startswith("ln"):
                values = torch.ones(shape, dtype=torch.float32)
            else:
                values = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02
            self.register_parameter(name, nn.Parameter(values))
        self.register_buffer("mask", torch.ones(CONTEXT, CONTEXT, dtype=torch.bool).triu(1))

    def load_flat(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (PARAM_COUNT,) or not np.isfinite(values).all():
            raise ValueError("Expected a finite flat checkpoint matching PARAM_COUNT")
        with torch.no_grad():
            for name, shape in PARAM_SHAPES.items():
                offset, size = PARAMS[name]
                getattr(self, name).copy_(
                    torch.from_numpy(values[offset : offset + size].reshape(shape))
                )
        return self

    def flat_parameters(self):
        return (
            torch.cat([getattr(self, name).detach().reshape(-1) for name in PARAM_SHAPES])
            .numpy()
            .copy()
        )

    def flat_gradients(self):
        parts = []
        for name in PARAM_SHAPES:
            parameter = getattr(self, name)
            if parameter.grad is None:
                raise ValueError(f"Parameter {name} has no gradient")
            parts.append(parameter.grad.detach().reshape(-1))
        return torch.cat(parts).numpy().copy()

    def forward(self, tokens, return_trace=False):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= CONTEXT:
            raise ValueError("tokens must have shape [batch, length], with 1 <= length <= 32")
        batch, length = tokens.shape
        x = F.embedding(tokens, self.token) + self.position[:length]
        n1 = F.layer_norm(x, (WIDTH,), self.ln1_weight, self.ln1_bias, EPS)
        qkv = F.linear(n1, self.qkv_weight, self.qkv_bias)
        q, k, v = [
            part.reshape(batch, length, HEADS, WIDTH // HEADS).transpose(1, 2)
            for part in qkv.chunk(3, dim=-1)
        ]
        attention = (
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch, length, WIDTH)
        )
        projected = F.linear(attention, self.attention_weight, self.attention_bias)
        r1 = x + projected
        n2 = F.layer_norm(r1, (WIDTH,), self.ln2_weight, self.ln2_bias, EPS)
        ff1 = F.linear(n2, self.ff1_weight, self.ff1_bias)
        relu = F.relu(ff1)
        ff2 = F.linear(relu, self.ff2_weight, self.ff2_bias)
        r2 = r1 + ff2
        n3 = F.layer_norm(r2, (WIDTH,), self.ln3_weight, self.ln3_bias, EPS)
        logits = F.linear(n3, self.head_weight, self.head_bias)
        if not return_trace:
            return logits
        # These probabilities are a mathematical audit reference, while actual
        # forward/backward values above use PyTorch's selected SDPA CPU kernel.
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(WIDTH // HEADS))
        scores = scores.masked_fill(self.mask[:length, :length], float("-inf"))
        probs = F.softmax(scores, dim=-1)
        trace = {
            "x": x,
            "n1": n1,
            "qkv": qkv,
            "attention": attention,
            "projected": projected,
            "r1": r1,
            "n2": n2,
            "ff1": ff1,
            "relu": relu,
            "ff2": ff2,
            "r2": r2,
            "n3": n3,
            "logits": logits,
            "probs": probs,
        }
        for index, value in enumerate((x, r1, r2), start=1):
            mean = value.mean(dim=-1, keepdim=True)
            variance = ((value - mean) ** 2).mean(dim=-1, keepdim=True)
            inv = torch.rsqrt(variance + EPS)
            trace[f"norm{index}"] = (value - mean) * inv
            trace[f"inv{index}"] = inv.squeeze(-1)
        for value in trace.values():
            if value.requires_grad:
                value.retain_grad()
        return logits, trace


TinyTransformer = Transformer
