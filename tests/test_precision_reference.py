"""Prospective CPU arithmetic and saved-state checks; no admission or timing."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

PATH = Path(__file__).resolve().parents[1] / "integration/trained-precision/precision_reference.py"
SPEC = importlib.util.spec_from_file_location("precision_candidate", PATH)
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


def check_operation(operation, reference, shapes):
    generator = torch.Generator().manual_seed(20261103)
    inputs = [torch.randn(shape, generator=generator, requires_grad=True) for shape in shapes]
    doubles = [value.detach().double().requires_grad_() for value in inputs]
    saved = []

    def retain(tensor):
        saved.append(tensor.dtype)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(retain, lambda tensor: tensor):
        actual = operation(*inputs)
        upstream = torch.randn(actual.shape, generator=generator)
        actual.backward(upstream)
    expected = reference(*doubles)
    expected.backward(upstream.double())
    assert saved and set(saved) == {torch.float32}
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-4)
    for value, double in zip(inputs, doubles, strict=True):
        assert value.grad.dtype == torch.float32
        torch.testing.assert_close(value.grad.double(), double.grad, atol=3e-5, rtol=3e-3)


def test_qkv_forward_backward_has_fp32_saved_state():
    check_operation(candidate.Linear.apply, F.linear, [(2, 3, 7), (11, 7), (11,)])


def test_attention_forward_backward_has_fp32_saved_state():
    check_operation(
        candidate.Attention.apply,
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        [(2, 4, 5, 8)] * 3,
    )


def test_attention_repeated_tokens_and_causal_prefix():
    generator = torch.Generator().manual_seed(20261103)
    q, k, v = [torch.randn(1, 4, 8, 16, generator=generator) for _ in range(3)]
    full = candidate.Attention.apply(q, k, v)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, :, 4:] *= 17
    changed_v[:, :, 4:] *= -13
    changed = candidate.Attention.apply(q, changed_k, changed_v)
    assert torch.equal(full[:, :, :4], changed[:, :, :4])
    equal = torch.ones_like(q)
    assert torch.equal(candidate.Attention.apply(equal, equal, equal), equal)


def test_complete_candidate_saves_no_fp64_tensors():
    model = candidate.PrecisionReference()
    saved = []

    def retain(value):
        saved.append(value.dtype)
        return value

    with torch.autograd.graph.saved_tensors_hooks(retain, lambda value: value):
        logits = model(torch.zeros((1, 3), dtype=torch.int64))
        logits.square().mean().backward()
    assert torch.float64 not in saved
    assert all(
        parameter.dtype == parameter.grad.dtype == torch.float32 for parameter in model.parameters()
    )


def test_candidate_rejects_undeclared_precision():
    with pytest.raises(ValueError, match="CPU FP32"):
        candidate.Linear.apply(torch.zeros(2, 3).double(), torch.zeros(4, 3), torch.zeros(4))
