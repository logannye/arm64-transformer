"""Explicit CPU diagnostic substitutions, never a standard-framework baseline.

Bits select first LayerNorm (1), QKV (2), and attention (4) FP64 intermediates.
Every operation returns FP32 at its module boundary. Autograd differentiates the
chosen operations; the FP64 attention candidate retains FP64 internal saved
state and is only a localization probe, not an admitted FP32-state study arm.
"""

import torch
from torch.nn import functional as F

from transformer.reference import Transformer
from transformer.spec import EPS, HEADS, WIDTH


class PrecisionProbe(Transformer):
    def __init__(self, policy):
        if type(policy) is not int or not 0 <= policy <= 7:
            raise ValueError("Unknown diagnostic precision policy")
        super().__init__()
        self.precision_policy = policy

    def forward(self, tokens, return_trace=False):
        if self.precision_policy == 0:
            return super().forward(tokens, return_trace=return_trace)
        if self.token.device.type != "cpu" or self.token.dtype != torch.float32:
            raise ValueError("Precision probes require FP32 CPU parameters")
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= 32:
            raise ValueError("Invalid diagnostic token shape")
        batch, length = tokens.shape
        x = F.embedding(tokens, self.token) + self.position[:length]
        if self.precision_policy & 1:
            n1 = F.layer_norm(
                x.double(), (WIDTH,), self.ln1_weight.double(), self.ln1_bias.double(), EPS
            ).float()
        else:
            n1 = F.layer_norm(x, (WIDTH,), self.ln1_weight, self.ln1_bias, EPS)
        if self.precision_policy & 2:
            qkv = F.linear(n1.double(), self.qkv_weight.double(), self.qkv_bias.double()).float()
        else:
            qkv = F.linear(n1, self.qkv_weight, self.qkv_bias)
        q, k, v = [
            part.reshape(batch, length, HEADS, WIDTH // HEADS).transpose(1, 2)
            for part in qkv.chunk(3, dim=-1)
        ]
        if self.precision_policy & 4:
            attention = F.scaled_dot_product_attention(
                q.double(), k.double(), v.double(), dropout_p=0.0, is_causal=True
            ).float()
        else:
            attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attention = attention.transpose(1, 2).contiguous().reshape(batch, length, WIDTH)
        projected = F.linear(attention, self.attention_weight, self.attention_bias)
        r1 = x + projected
        n2 = F.layer_norm(r1, (WIDTH,), self.ln2_weight, self.ln2_bias, EPS)
        ff1 = F.linear(n2, self.ff1_weight, self.ff1_bias)
        relu = F.relu(ff1)
        ff2 = F.linear(relu, self.ff2_weight, self.ff2_bias)
        r2 = r1 + ff2
        n3 = F.layer_norm(r2, (WIDTH,), self.ln3_weight, self.ln3_bias, EPS)
        logits = F.linear(n3, self.head_weight, self.head_bias)
        if return_trace:
            return logits, dict(
                x=x,
                n1=n1,
                qkv=qkv,
                attention=attention,
                projected=projected,
                r1=r1,
                n2=n2,
                ff1=ff1,
                relu=relu,
                ff2=ff2,
                r2=r2,
                n3=n3,
                logits=logits,
            )
        return logits
