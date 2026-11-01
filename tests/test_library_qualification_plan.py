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


@pytest.mark.parametrize("kind", ["optimizer", "common_state", "collection", "provenance"])
def test_pre_ff1_cpu_model_results_cannot_qualify_revised_cpu(kind):
    old, new = {"path": "/old"}, {"path": "/new"}
    reuse = {"capture": old}
    with pytest.raises(ValueError, match="Pre-FF1 CPU model"):
        Q.previous_scope(old, new, reuse, kind, "cpu_torch", library_api=True)
    assert not Q.previous_scope(new, new, reuse, kind, "cpu_torch", library_api=True)
    assert Q.previous_scope(old, new, reuse, kind, "mps", library_api=True)
    assert Q.previous_scope(old, new, reuse, kind, "mlx", library_api=True)
    with pytest.raises(ValueError, match="Foreign audit"):
        Q.previous_scope({"path": "/foreign"}, new, reuse, kind, "mps")


def test_pre_ff1_ordinary_cpu_and_primitives_remain_separate_from_cpu_model_gates():
    old, new = {"path": "/old"}, {"path": "/new"}
    reuse = {"capture": old}
    for kind in ("primitive", "collection", "provenance"):
        assert Q.previous_scope(old, new, reuse, kind, "cpu_torch")
    for kind in ("optimizer", "common_state"):
        with pytest.raises(ValueError, match="Pre-FF1 CPU model"):
            Q.previous_scope(old, new, reuse, kind, "cpu_torch")
