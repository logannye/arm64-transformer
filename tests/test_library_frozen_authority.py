"""A prospective or mutated plan cannot authorize a reserved stream."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_frozen_authority", ROOT / "integration/library-study-v1/frozen.py"
)
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def test_frozen_shape_preserves_exact_comparison_and_no_permission_when_closed(tmp_path):
    value = F.C.plan()
    value.update(
        phase="frozen", bindings={k: None for k in F.KEYS}, frozen_utc="2026-10-31T00:00:00Z"
    )
    assert F.validate_shape(value) is value
    broken = copy.deepcopy(value)
    broken["comparison"]["cells"].pop()
    with pytest.raises(ValueError, match="comparison changed"):
        F.validate_shape(broken)
    broken = copy.deepcopy(value)
    broken["admission_seed"] = 20261108
    with pytest.raises(ValueError, match="comparison changed"):
        F.validate_shape(broken)
    with pytest.raises(ValueError, match="not implemented"):
        F.verify(value)
    with pytest.raises(ValueError, match="not implemented"):
        F.AdmissionPermission(tmp_path / "unqualified.json", "0" * 64)
    assert not (tmp_path / "unqualified.json").exists()
