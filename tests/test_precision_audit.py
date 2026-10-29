"""Independent-oracle and qualification regressions, runnable without assembly."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from transformer import precision_audit as audit
from transformer import spec
from transformer.reference import Transformer
from transformer.validate import Checks, reference_forward, workspace_tensor


@pytest.fixture
def checkpoint():
    torch.set_num_threads(1)
    return Transformer(seed=87241).flat_parameters()


def test_numpy_fp64_model_matches_independent_torch_equations(checkpoint):
    rng = np.random.default_rng(20261106)
    tokens = rng.integers(0, spec.VOCAB, (2, spec.CONTEXT), dtype=np.int64)
    numpy = audit.FP64Transformer(checkpoint)(tokens)
    flat = torch.tensor(checkpoint, dtype=torch.float64)
    with torch.no_grad():
        torch_values = np.stack(
            [reference_forward(flat, torch.tensor(row)).numpy() for row in tokens]
        )
    assert numpy.dtype == np.float64
    assert np.allclose(numpy, torch_values, **audit.ORACLE_TOLERANCES)


def test_fp64_oracle_preserves_causal_prefix(checkpoint):
    tokens = np.arange(spec.CONTEXT, dtype=np.int64)
    altered = tokens.copy()
    altered[16:] = (altered[16:] + 19) % spec.VOCAB
    model = audit.FP64Transformer(checkpoint)
    first, second = model(tokens), model(altered)
    assert np.array_equal(first[:16], second[:16])
    assert not np.array_equal(first[16:], second[16:])


def test_comparison_retains_failure_coordinates_and_fixed_limit():
    result = audit.comparison("case", np.array([0.00003, 1.0]), np.array([0.0, 1.0]))
    assert result["passed"] is False
    assert result["failed_coordinates"] == 1
    assert result["failure_examples"][0]["index"] == [0]
    assert result["failure_examples"][0]["allowed"] == 2e-5
    assert audit.comparison("shape", [1, 2], [1])["passed"] is False
    assert audit.comparison("finite", [np.nan], [1])["passed"] is False


def test_predictive_differences_respect_shift_invariance_and_detect_changed_winner():
    logits = np.zeros((2, spec.VOCAB), np.float32)
    targets = np.array([0, 1], np.int64)
    shifted = audit.prediction_differences(logits + 20, logits, targets)
    assert shifted["argmax_changes"] == 0
    assert shifted["maximum_probability_absolute_difference"] == 0
    assert shifted["probability_rmse"] == 0
    assert shifted["nll_delta"] == pytest.approx(0, abs=1e-14)
    changed = logits.copy()
    changed[0, 3] = 1
    result = audit.prediction_differences(changed, logits, targets)
    assert result["token_positions"] == 2
    assert result["argmax_changes"] == 1
    assert result["maximum_probability_absolute_difference"] > 0
    assert result["probability_rmse"] > 0
    assert result["nll_delta"] > 0


@pytest.mark.parametrize(
    "integrity,native,python,compatibility,status,qualified",
    [
        (True, True, True, True, "completed", True),
        (True, True, True, False, "completed", True),
        (True, True, False, False, "baseline_precision_limit", False),
        (True, False, True, False, "native_precision_limit", False),
        (True, False, False, False, "native_precision_limit", False),
        (False, True, True, True, "audit_failed", False),
    ],
)
def test_no_numerical_failure_disappears_from_qualification(
    integrity, native, python, compatibility, status, qualified
):
    categories = {
        "native_fp64": {"passed": native},
        "python_fp64": {"passed": python},
        "fp32_compatibility": {"passed": compatibility},
    }
    result = audit.numerical_status(integrity, categories)
    assert result["status"] == status
    assert result["passed"] is qualified
    assert result["numerically_qualified"] is qualified
    assert result["cross_implementation_fp32_compatibility_passed"] is compatibility


def test_checkpoint_replay_uses_original_partial_batches(checkpoint, monkeypatch):
    """Batch-dependent rounding must not become a false saved-replay failure."""
    original_model = Transformer

    class BatchSensitiveModel(Transformer):
        def forward(self, tokens, **kwargs):
            # Deliberately make the grouping visible, independently of a BLAS
            # release's particular rounding behavior.
            return super().forward(tokens, **kwargs) + np.float32(len(tokens) * 0.01)

    class NativeReplay:
        def __init__(self):
            self.model = original_model().load_flat(checkpoint)

        def forward(self, params, tokens):
            workspace = np.zeros(spec.WORK_COUNT, np.float32)
            with torch.no_grad():
                workspace_tensor(workspace, "logits")[:] = self.model(
                    torch.tensor(tokens, dtype=torch.long)[None]
                )[0].numpy()
            return workspace

        def call(self, name, logits, targets, dlogits, context, vocab):
            assert name == "cross_entropy"
            return F.cross_entropy(torch.tensor(logits), torch.tensor(targets).long()).item()

    monkeypatch.setattr(audit, "Transformer", BatchSensitiveModel)
    rng = np.random.default_rng(613)
    rows = rng.integers(0, spec.VOCAB, (3, spec.CONTEXT + 1), dtype=np.int32)
    model = BatchSensitiveModel().load_flat(checkpoint)
    saved = []
    total = 0.0
    with torch.no_grad():
        for start in (0, 2):
            selected = rows[start : start + 2]
            logits = model(torch.tensor(selected[:, :-1]).long())
            saved.extend(logits.numpy())
            total += F.cross_entropy(
                logits.reshape(-1, spec.VOCAB), torch.tensor(selected[:, 1:]).long().reshape(-1)
            ).item() * len(selected)
    history = np.array([[1, total / len(rows), 1]], np.float64)
    checks = Checks()
    result = audit.checkpoint_audit(
        checkpoint,
        rows,
        rows,
        np.asarray(saved),
        history,
        2,
        "python",
        "fixture",
        checks,
        native=NativeReplay(),
    )
    assert checks.passed
    assert result["python_replay_batch_size"] == 2
    assert result["categories"]["python_fp64"]["passed"] is False
    for split in ("test", "validation"):
        assert set(result["prediction_differences"][split]) == {
            "native_vs_fp64",
            "python_vs_fp64",
            "native_vs_python",
        }
        assert (
            result["prediction_differences"][split]["native_vs_python"]["token_positions"]
            == 3 * spec.CONTEXT
        )
    altered = np.asarray(saved).copy()
    altered[1, 0, 0] += np.float32(0.25)
    checks = Checks()
    audit.checkpoint_audit(
        checkpoint,
        rows,
        rows,
        altered,
        history,
        2,
        "python",
        "fixture",
        checks,
        native=NativeReplay(),
    )
    assert [r["name"] for r in checks.records if not r["passed"]] == [
        "fixture/v6/test/sequence_1/originating_replay_exact"
    ]
