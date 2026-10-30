"""CPU equation/gradient screening of the prospective GPU arithmetic diagnostic."""

import importlib.util
import types
from pathlib import Path

import pytest
import torch

PATH = (
    Path(__file__).resolve().parents[1] / "integration/trained-precision/compensated_attention.py"
)
SPEC = importlib.util.spec_from_file_location("compensated_attention", PATH)
CANDIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CANDIDATE)
OPS = types.SimpleNamespace(
    concatenate=lambda a, axis: torch.cat(a, dim=axis),
    zeros_like=torch.zeros_like,
    arange=torch.arange,
    max=lambda a, axis, keepdims: torch.amax(a, dim=axis, keepdim=keepdims),
    where=torch.where,
    exp=torch.exp,
    swapaxes=torch.swapaxes,
)


@pytest.mark.parametrize("head_width", [128, 256])
def test_compensated_attention_and_autograd_against_fp64(head_width):
    rng = torch.Generator().manual_seed(20261103)
    inputs = [torch.randn((1, 4, 17, head_width), generator=rng).requires_grad_() for _ in range(3)]
    doubles = [a.detach().double().requires_grad_() for a in inputs]
    upstream = torch.randn(inputs[0].shape, generator=rng) * 0.01
    saved = []

    def retain(value):
        saved.append(value.dtype)
        return value

    with torch.autograd.graph.saved_tensors_hooks(retain, lambda value: value):
        actual, probability = CANDIDATE.attention(*inputs, OPS)
        (actual * upstream).sum().backward()
    expected = torch.nn.functional.scaled_dot_product_attention(*doubles, is_causal=True)
    (expected * upstream.double()).sum().backward()
    assert actual.dtype == probability.dtype == torch.float32
    assert torch.float64 not in saved
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-4)
    for value, double in zip(inputs, doubles, strict=True):
        torch.testing.assert_close(value.grad.double(), double.grad, atol=3e-5, rtol=3e-3)
    assert torch.equal(probability.triu(1), torch.zeros_like(probability))
    torch.testing.assert_close(probability.sum(-1), torch.ones_like(probability[..., 0]))


def test_compensated_attention_is_causal_for_repeated_values():
    q = torch.ones((1, 4, 9, 128))
    k, v = q.clone(), q.clone()
    before, _ = CANDIDATE.attention(q, k, v, OPS)
    k[..., 4:, :] *= 3
    v[..., 4:, :] *= -5
    after, _ = CANDIDATE.attention(q, k, v, OPS)
    assert torch.equal(before[..., :4, :], after[..., :4, :])
    torch.testing.assert_close(before, q)
