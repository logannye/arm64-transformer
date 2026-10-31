"""Every exact qualification cell is required before scheduling raw re-audits."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_qualification_plan", ROOT / "integration/library-study-v1/qualification_plan.py"
)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def test_cell_schedule_rejects_missing_duplicate_or_reordered_arms():
    rows = [dict(engine=e, width=w) for e in Q.FRAMEWORKS for w in Q.WIDTHS]
    assert Q.cell_inventory(rows, Q.FRAMEWORKS) is rows
    for broken in (
        rows[:-1],
        rows + rows[:1],
        list(reversed(rows)),
        [dict(rows[0], width=True), *rows[1:]],
    ):
        with pytest.raises(ValueError, match="qualification cell"):
            Q.cell_inventory(broken, Q.FRAMEWORKS)
    with pytest.raises(ValueError, match="collector set"):
        Q.build({}, {}, {}, "/raw")
