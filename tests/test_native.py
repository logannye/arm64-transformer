"""Native tests use an explicitly built library; they never invoke a compiler."""

import platform

import numpy as np
import pytest
import torch

from transformer import spec
from transformer.audit import prediction_metrics
from transformer.reference import Transformer
from transformer.validate import (
    DEFAULT_LIBRARY,
    Checks,
    Native,
    numerical_checks,
    reference_forward,
    tensor,
)

NATIVE_AVAILABLE = (
    platform.system() == "Darwin" and platform.machine() == "arm64" and DEFAULT_LIBRARY.is_file()
)


def test_checks_reject_nan_and_wrong_shape():
    checks = Checks()
    checks.close("nan", [np.nan], [np.nan])
    checks.close("shape", [1, 2], [[1, 2]])
    assert not checks.passed
    assert all(not item["passed"] for item in checks.records)


def test_independent_functional_reference_matches_production_model():
    torch.set_num_threads(1)
    model = Transformer(1729)
    tokens = torch.arange(spec.CONTEXT).reshape(1, -1)
    actual = model(tokens)
    flat = tensor(model.flat_parameters(), grad=True)
    expected = reference_forward(flat, tokens[0])
    torch.testing.assert_close(actual[0], expected, atol=2e-6, rtol=2e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    np.testing.assert_allclose(model.flat_gradients(), flat.grad.numpy(), atol=2e-6, rtol=2e-4)


def test_independent_uniform_metrics_and_shape_validation():
    logits = np.zeros((2, 3, spec.VOCAB), dtype=np.float32)
    labels = np.zeros((2, 3), dtype=np.int32)
    metrics = prediction_metrics(logits, labels)
    assert metrics["tokens"] == 6
    assert metrics["accuracy"] == 1
    assert metrics["bits_per_character"] == pytest.approx(7)
    assert metrics["perplexity"] == pytest.approx(128)
    with pytest.raises(ValueError, match="matching shapes"):
        prediction_metrics(logits, labels[:, :2])


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Requires an explicitly built macOS ARM64 library")
def test_native_numerical_contract():
    checks = Checks()
    numerical_checks(Native(), checks)
    failures = [item for item in checks.records if not item["passed"]]
    assert not failures, failures
