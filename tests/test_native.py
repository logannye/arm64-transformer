"""Native tests use an explicitly built library; they never invoke a compiler."""

import platform
from types import SimpleNamespace

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
    guard_intact,
    guarded_array,
    numerical_checks,
    reference_forward,
    tensor,
    verify_library_model,
)

NATIVE_AVAILABLE = (
    platform.system() == "Darwin" and platform.machine() == "arm64" and DEFAULT_LIBRARY.is_file()
)


class FakeSymbol:
    def __init__(self, result):
        self.result = result

    def __call__(self):
        return self.result


def model_library(width=spec.WIDTH, parameter_count=spec.PARAM_COUNT):
    return SimpleNamespace(
        tf_model_width=FakeSymbol(width), tf_parameter_count=FakeSymbol(parameter_count)
    )


def test_native_identity_requires_both_matching_model_dimensions():
    library = model_library()
    assert verify_library_model(library) == {
        "width": spec.WIDTH,
        "parameter_count": spec.PARAM_COUNT,
        "verified": True,
    }
    assert library.tf_model_width.argtypes == []
    assert library.tf_parameter_count.argtypes == []
    for wrong in (model_library(width=spec.WIDTH + 1), model_library(parameter_count=1)):
        with pytest.raises(RuntimeError, match="Native model mismatch"):
            verify_library_model(wrong)
        with pytest.raises(RuntimeError, match="Native model mismatch"):
            verify_library_model(wrong, allow_legacy=True)


def test_native_identity_legacy_mode_is_explicit_and_width_32_only(monkeypatch):
    missing = SimpleNamespace()
    with pytest.raises(RuntimeError, match="identity symbols"):
        verify_library_model(missing)
    monkeypatch.setattr(spec, "WIDTH", 32)
    monkeypatch.setattr(spec, "PARAM_COUNT", 17952)
    assert verify_library_model(missing, allow_legacy=True) == {
        "width": 32,
        "parameter_count": 17952,
        "verified": False,
    }
    partial = SimpleNamespace(tf_model_width=FakeSymbol(32))
    with pytest.raises(RuntimeError, match="identity symbols"):
        verify_library_model(partial, allow_legacy=True)
    monkeypatch.setattr(spec, "WIDTH", 64)
    with pytest.raises(RuntimeError, match="identity symbols"):
        verify_library_model(missing, allow_legacy=True)


def test_native_default_paths_are_width_specific():
    assert DEFAULT_LIBRARY.parent.name == f"width-{spec.WIDTH}"


def test_checks_reject_nan_and_wrong_shape():
    checks = Checks()
    checks.close("nan", [np.nan], [np.nan])
    checks.close("shape", [1, 2], [[1, 2]])
    assert not checks.passed
    assert all(not item["passed"] for item in checks.records)


def test_guarded_views_detect_vector_tail_overwrites():
    values, storage = guarded_array((3, 5))
    assert values.flags.c_contiguous
    values.fill(0)
    assert guard_intact(storage)
    storage[-8] = 0
    assert not guard_intact(storage)


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
