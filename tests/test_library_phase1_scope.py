"""The approved scope amendment cannot change any retained cell or gate."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_phase1_contract", ROOT / "integration/library-study-v1/contract.py"
)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def test_phase1_is_exact_projection_and_preserves_the_original_six_width_contract():
    old, core = C.original_comparison(), C.comparison()
    assert old["planned_fits"] == len(old["cells"]) == 180
    assert core["cells"] == [c for c in old["cells"] if c["width"] <= 256]
    assert core["gate_inventory"] == [
        g for g in old["gate_inventory"] if int(g["id"].split("/")[1]) <= 256
    ]
    assert len(core["cells"]) == core["planned_fits"] == 120
    assert core["planned_test_token_predictions"] == 983040
    assert len(core["gate_inventory"]) == 1392
    assert sum(g["blocking"] for g in core["gate_inventory"]) == 1032
    assert sum(len(g["members"]) for g in core["gate_inventory"]) == 6572
    assert core["scope_amendment"]["deferred_fits"] == 60
    assert core["scope_amendment"]["deferred_widths"] == [512, 1024]
    assert core["scope_amendment"]["original_study_complete"] is False
    changed = {
        "cells",
        "gate_inventory",
        "gate_inventory_sha256",
        "planned_fits",
        "planned_test_token_predictions",
        "required_gates",
        "workload",
    }
    assert all(old[k] == core[k] for k in old if k not in changed)
    expected_workload = dict(old["workload"], widths=[32, 64, 128, 256])
    assert core["workload"] == expected_workload
    assert C.original_comparison() == old


@pytest.mark.parametrize("change", ["cell", "gate", "deferred", "schedule", "precision"])
def test_amended_plan_rejects_omissions_or_numerical_changes(change):
    value = copy.deepcopy(C.plan())
    comparison = value["comparison"]
    if change == "cell":
        comparison["cells"].pop()
    elif change == "gate":
        comparison["gate_inventory"].pop()
    elif change == "deferred":
        comparison["scope_amendment"]["original_study_complete"] = True
    elif change == "schedule":
        comparison["workload"]["steps"] -= 1
    else:
        comparison["model_selection"]["cpu_torch"]["stages"].remove("ff1")
    with pytest.raises(ValueError, match="contract changed"):
        C.validate_plan(value)
