"""Original published inputs cannot differ between arms or drop a seed/repeat."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_original_fixtures", ROOT / "integration/library-study-v1/fixtures.py"
)
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def test_fixture_selection_rejects_changed_repeat_or_missing_original_arm():
    hashes = {name: "a" * 64 for name in F.FILES}
    summary = dict(
        runs=[dict(seed=42, fixture="seed-42", fixture_sha256=dict(hashes)) for _ in range(4)]
    )
    assert F.published_inputs(summary, 42) == hashes
    broken = copy.deepcopy(summary)
    broken["runs"][3]["fixture_sha256"]["initial.f32"] = "b" * 64
    with pytest.raises(ValueError, match="differ"):
        F.published_inputs(broken, 42)
    with pytest.raises(ValueError, match="denominator"):
        F.published_inputs(dict(runs=summary["runs"][:3]), 42)
    with pytest.raises(ValueError, match="Unplanned"):
        F.published_inputs(summary, 20261112)
    with pytest.raises(ValueError, match="Unplanned"):
        F.published_inputs(summary, True)
