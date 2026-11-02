"""Screen tiled dense arithmetic and its explicit saved-state derivative."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "integration/trained-precision"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LINEAR = load("compensated_linear")
MODEL = load("framework_reference")


@pytest.mark.parametrize("shape", [(2, 3, 513, 67), (1, 33, 1024, 7)])
def test_tiling_forward_and_all_derivatives_against_fp64(shape):
    b, t, i, o = shape
    rng = torch.Generator().manual_seed(20261103)
    inputs = [torch.randn(s, generator=rng).requires_grad_() for s in ((b, t, i), (o, i), (o,))]
    doubles = [v.detach().double().requires_grad_() for v in inputs]
    dy = torch.randn((b, t, o), generator=rng) * 0.01
    saved = []

    def retain(value):
        saved.append((value.shape, value.dtype))
        return value

    operation = LINEAR.torch_function(torch, MODEL.torch_ops(torch, "cpu"))
    with torch.autograd.graph.saved_tensors_hooks(retain, lambda v: v):
        actual = operation(*inputs)
        actual.backward(dy)
    expected = torch.nn.functional.linear(*doubles)
    expected.backward(dy.double())
    assert saved == [(inputs[0].shape, torch.float32), (inputs[1].shape, torch.float32)]
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-4)
    # Float32 rounding is the expected error for these bounded operands.
    torch.testing.assert_close(actual, expected.float(), atol=1e-6, rtol=1e-7)
    for actual_input, reference in zip(inputs, doubles, strict=True):
        torch.testing.assert_close(actual_input.grad.double(), reference.grad, atol=3e-5, rtol=3e-3)


def test_cancellation_and_bias_are_accumulated_before_rounding():
    ops = MODEL.torch_ops(torch, "cpu")
    x = torch.tensor([[1e4, 1.0, -1e4, 1e-4]])
    weight = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    bias = torch.tensor([-1.0])
    actual = LINEAR.dense(x, weight, bias, ops)
    expected = torch.nn.functional.linear(x.double(), weight.double(), bias.double()).float()
    assert torch.equal(actual, expected)
