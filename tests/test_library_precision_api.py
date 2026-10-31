"""Prospective CPU operator checks; device/package qualification is separate."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

DIRECTORY = Path(__file__).resolve().parents[1] / "integration/precision-library-v1"
SPEC = importlib.util.spec_from_file_location(
    "prospective_precision_api",
    DIRECTORY / "torch_api.py",
    submodule_search_locations=[str(DIRECTORY)],
)
API = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = API
SPEC.loader.exec_module(API)


def random(shape, generator):
    return torch.randn(shape, generator=generator, dtype=torch.float32).requires_grad_()


def bytes_of(value):
    return value.detach().contiguous().view(torch.uint8)


def check_operation(function, reference, inputs):
    before = [bytes_of(value).clone() for value in inputs]
    actual = function(*inputs)
    doubles = [value.detach().double().requires_grad_() for value in inputs]
    expected = reference(*doubles)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=3e-6)
    generator = torch.Generator().manual_seed(20261103)
    upstream = torch.randn(actual.shape, generator=generator, dtype=torch.float32)
    gradients = torch.autograd.grad(actual, inputs, upstream)
    wanted = torch.autograd.grad(expected, doubles, upstream.double())
    for observed, exact in zip(gradients, wanted, strict=True):
        assert observed.dtype == torch.float32
        torch.testing.assert_close(observed.double(), exact, atol=3e-6, rtol=3e-5)
    repeated = function(*inputs)
    again = torch.autograd.grad(repeated, inputs, upstream)
    assert torch.equal(bytes_of(actual), bytes_of(repeated))
    assert all(torch.equal(bytes_of(x), bytes_of(y)) for x, y in zip(gradients, again, strict=True))
    assert all(torch.equal(bytes_of(x), y) for x, y in zip(inputs, before, strict=True))


@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 7)])
def test_cpu_dense_forward_derivatives_and_replay(shape):
    g = torch.Generator().manual_seed(20261103)
    x, weight, bias = random(shape, g), random((5, 7), g), random((5,), g)
    check_operation(API.linear, F.linear, (x, weight, bias))


def test_cpu_dense_accepts_noncontiguous_input_and_optional_bias():
    g = torch.Generator().manual_seed(20261103)
    x = random((3, 14), g)[:, ::2]
    weight = random((5, 7), g)
    assert not x.is_contiguous()
    check_operation(lambda a, b: API.linear(a, b), lambda a, b: F.linear(a, b), (x, weight))


@pytest.mark.parametrize("shape", [(17,), (3, 17), (2, 3, 32)])
def test_cpu_normalization_forward_derivatives_and_replay(shape):
    g = torch.Generator().manual_seed(20261103)
    x, weight, bias = random(shape, g), random((shape[-1],), g), random((shape[-1],), g)
    check_operation(
        API.layer_norm,
        lambda a, w, b: F.layer_norm(a, (a.shape[-1],), w, b, 1e-5),
        (x, weight, bias),
    )


def test_cpu_normalization_accepts_noncontiguous_input_without_affine_vectors():
    g = torch.Generator().manual_seed(20261103)
    x = random((3, 14), g)[:, ::2]
    check_operation(API.layer_norm, lambda a: F.layer_norm(a, (7,), eps=1e-5), (x,))


def test_cpu_attention_forward_derivatives_and_replay():
    g = torch.Generator().manual_seed(20261103)
    values = tuple(random((2, 3, 5, 7), g) for _ in range(3))

    def reference(q, k, v):
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
        return scores.masked_fill(mask, -torch.inf).softmax(-1) @ v

    check_operation(API.scaled_dot_product_attention, reference, values)


def test_cpu_attention_preserves_causal_prefix_exactly():
    g = torch.Generator().manual_seed(20261103)
    values = tuple(random((2, 3, 5, 7), g) for _ in range(3))
    output = API.scaled_dot_product_attention(*values)
    changed = tuple(value.detach().clone() for value in values)
    for value in changed:
        value[:, :, 3:, :] += 200
    repeated = API.scaled_dot_product_attention(*changed)
    assert torch.equal(bytes_of(output[:, :, :3]), bytes_of(repeated[:, :, :3]))


@pytest.mark.parametrize("operation", ["linear", "layer_norm", "attention"])
def test_cpu_saved_tensors_remain_fp32(operation):
    g = torch.Generator().manual_seed(20261103)
    saved = []

    def pack(value):
        saved.append((value.dtype, value.device.type))
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        if operation == "linear":
            result = API.linear(random((3, 7), g), random((5, 7), g), random((5,), g))
        elif operation == "layer_norm":
            result = API.layer_norm(random((3, 7), g), random((7,), g), random((7,), g))
        else:
            result = API.scaled_dot_product_attention(*(random((2, 3, 5, 7), g) for _ in range(3)))
        result.sum().backward()
    assert saved and all(dtype == torch.float32 and device == "cpu" for dtype, device in saved)


def test_precision_api_rejects_wrong_dtype_shapes_and_unsupported_attention():
    x, weight = torch.ones(3, 7), torch.ones(5, 7)
    for args in ((x.double(), weight), (x, weight.double()), (x, weight[:, :4]), (x[:0], weight)):
        with pytest.raises(ValueError):
            API.linear(*args)
    with pytest.raises(ValueError):
        API.layer_norm(x, epsilon=-1)
    with pytest.raises(ValueError):
        API.layer_norm(x, torch.ones(8))
    q = torch.ones(2, 3, 5, 7)
    for kwargs in ({"dropout_p": 0.1}, {"is_causal": False}, {"scale": float("nan")}):
        with pytest.raises(ValueError):
            API.scaled_dot_product_attention(q, q, q, **kwargs)
