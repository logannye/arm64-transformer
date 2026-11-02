"""Check replay inputs stay dynamic when one diagnostic evaluator is reused."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from transformer.reference import Transformer
from transformer.trained_precision import torch_evaluate

PATH = Path(__file__).resolve().parents[1] / "integration/trained-precision/framework_candidate.py"
SPEC = importlib.util.spec_from_file_location("framework_candidate_test", PATH)
CANDIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CANDIDATE)


@pytest.mark.parametrize("gradients", [False, True])
def test_reused_evaluator_uses_each_checkpoint_and_batch(gradients):
    evaluator = CANDIDATE.Evaluator("cpu_torch", 0, gradients=gradients)
    tokens = np.arange(6, dtype="i4").reshape(2, 3)
    previous = None
    for index in range(3):
        flat = Transformer(seed=42 + index).flat_parameters()
        batch = tokens + index
        target = batch + 1
        actual = evaluator.evaluate(flat, batch, target)
        expected = torch_evaluate(flat, batch, target, dtype=torch.float32, gradients=gradients)
        assert actual.keys() == expected.keys()
        for name in actual:
            assert actual[name].dtype == expected[name].dtype
            assert actual[name].tobytes() == expected[name].tobytes()
        assert previous is None or not np.array_equal(previous, actual["logits"])
        previous = actual["logits"]
    assert evaluator.executions == 3


def test_evaluator_rejects_wrong_precision_or_compilation_target():
    with pytest.raises(ValueError, match="Only MLX"):
        CANDIDATE.Evaluator("cpu_torch", 2, gradients=False, compiled=True)
    evaluator = CANDIDATE.Evaluator("cpu_torch", 2, gradients=False)
    with pytest.raises(ValueError, match="FP32"):
        evaluator.evaluate(Transformer().flat_parameters().astype("f8"), None, None)


def test_feed_forward_refinement_has_a_separate_identity():
    assert CANDIDATE.identity(7) == "compensated-framework-model-diagnostic-v2"
    assert CANDIDATE.identity(15) == "compensated-framework-model-diagnostic-v3"
    for value in (-1, 16, True):
        with pytest.raises(ValueError, match="policy"):
            CANDIDATE.identity(value)


def test_feed_forward_refinement_preserves_full_model_values_and_gradients():
    flat = Transformer(seed=42).flat_parameters()
    tokens = np.array([[0, 127], [127, 0]], "i4")
    targets = np.array([[127, 0], [0, 127]], "i4")
    evaluator = CANDIDATE.Evaluator("cpu_torch", 15, gradients=True)
    actual = evaluator.evaluate(flat, tokens, targets)
    expected = torch_evaluate(flat, tokens, targets, dtype=torch.float64, gradients=True)
    for name, value in actual.items():
        tolerance = (3e-5, 3e-3) if name == "gradient" else (2e-5, 2e-4)
        assert value.dtype == np.float32 and value.shape == expected[name].shape
        assert np.allclose(value, expected[name], atol=tolerance[0], rtol=tolerance[1])
