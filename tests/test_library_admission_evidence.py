"""Lossless raw pairs must survive independent reading and reject altered decisions."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_admission_evidence", ROOT / "integration/library-study-v1/evidence.py"
)
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


@pytest.mark.parametrize(
    "value", [np.array(7, np.int64), np.arange(12, dtype=np.float32).reshape(3, 4).T]
)
def test_raw_storage_preserves_scalar_shape_dtype_and_noncontiguous_values(tmp_path, value):
    row = E.save_pair(value, value, tmp_path, ROOT)
    actual, expected = E.load_pair(row, tmp_path, ROOT)
    assert actual.shape == expected.shape == value.shape
    assert actual.dtype == expected.dtype == value.dtype
    assert actual.tobytes() == expected.tobytes() == value.tobytes()
    objects_before = sorted((tmp_path / "objects").iterdir())
    assert E.save_pair(value, value, tmp_path, ROOT) == row
    assert sorted((tmp_path / "objects").iterdir()) == objects_before
    objects_before[0].write_bytes(b"changed raw evidence")
    with pytest.raises(ValueError, match="size|bytes"):
        E.load_pair(row, tmp_path, ROOT)


def test_reported_pass_cannot_replace_raw_accuracy_or_exact_identity(tmp_path):
    actual, expected = np.array([1], np.float32), np.array([2], np.float64)
    spec = dict(id="mps/32/full/B1T32", members={"logits": "forward"}, blocking=True)
    row = dict(
        id=spec["id"],
        members={"logits": E.save_pair(actual, expected, tmp_path, ROOT)},
        decisions={"logits": E.decision(actual, expected, "forward")},
        passed=False,
    )
    assert not E.audit_gate(spec, row, tmp_path, ROOT)
    broken = copy.deepcopy(row)
    broken["passed"] = True
    with pytest.raises(ValueError, match="differs"):
        E.audit_gate(spec, broken, tmp_path, ROOT)
    assert not E.decision(actual, actual.astype(np.float64), "exact")["passed"]
    assert not E.decision(np.array([np.nan]), np.array([np.nan]), "exact")["passed"]
    with pytest.raises(ValueError, match="empty"):
        E.decision(np.array([]), np.array([]), "exact")
