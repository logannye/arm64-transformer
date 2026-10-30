"""A raw auditor must reject forged success, input changes and missing coverage."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "integration/trained-precision/audit_replay.py"
SPEC = importlib.util.spec_from_file_location("raw_auditor", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def replay(tmp_path):
    root, output = tmp_path / "inputs", tmp_path / "worker"
    root.mkdir()
    output.mkdir()
    checkpoint = root / "checkpoint.f32"
    with checkpoint.open("wb") as stream:
        stream.truncate(sum(audit.parameter_sizes(512).values()) * 4)
    rows = np.zeros((2, 33), "i4")
    rows.tofile(root / "rows.i32")
    case = dict(
        id="first",
        width=512,
        sequence=0,
        checkpoint="checkpoint.f32",
        checkpoint_sha256=audit.sha256(checkpoint),
        rows="rows.i32",
        rows_sha256=audit.sha256(root / "rows.i32"),
    )
    plan = dict(schema="trained-precision-development-v1", cases=[case])
    plan_path = tmp_path / "inventory.json"
    plan_path.write_text(json.dumps(plan))
    arrays = dict(tokens=rows[:1, :-1], targets=rows[:1, 1:])
    checks = {}
    for name, shape in (("logits", (1, 32, 128)), ("loss", ())):
        for label in ("actual", "repeat", "numpy64", "torch64"):
            arrays[label + "_" + name] = np.zeros(shape, "f8" if label.endswith("64") else "f4")
        for suffix in ("", "_oracle", "_repeat"):
            checks[name + suffix] = {"passed": True}
    raw = output / "first.npz"
    np.savez_compressed(raw, **arrays)
    receipt = dict(
        schema=plan["schema"],
        inventory_sha256=audit.sha256(plan_path),
        formal_eligible=False,
        admission_stream_opened=False,
        complete=True,
        passed=True,
        width=512,
        batch_size=1,
        expected_cases=["first"],
        full_width_inventory=True,
        gradients_checked=False,
        engine="native",
        torch_diagnostic_policy=0,
        cases=[
            dict(
                case=case,
                first_sequence=0,
                raw=raw.name,
                raw_sha256=audit.sha256(raw),
                passed=True,
                checks=checks,
            )
        ],
    )
    return root, output, plan_path, receipt, arrays


def run(replay):
    root, output, plan, receipt, _ = replay
    target = output / "result.json"
    target.write_text(json.dumps(receipt))
    return audit.audit_worker(output, audit.sha256(target), plan, audit.sha256(plan), root)


def test_valid_raw_evidence_is_separate_from_formal_qualification(replay):
    result = run(replay)
    assert result["evidence_valid"] and result["numerics_passed"]
    assert result["formal_eligible"] is False


@pytest.mark.parametrize("change", ["decision", "oracle", "repeat", "input", "dtype", "shape"])
def test_rehashed_raw_mutations_cannot_preserve_a_success_claim(replay, change):
    _, output, _, receipt, arrays = replay
    key = {"oracle": "torch64_logits", "repeat": "repeat_logits", "input": "tokens"}.get(
        change, "actual_logits"
    )
    if change == "dtype":
        arrays[key] = arrays[key].astype("f8")
    elif change == "shape":
        arrays[key] = arrays[key].reshape(-1)
    else:
        arrays[key] = arrays[key].copy()
        arrays[key].flat[0] = 1
    raw = output / "first.npz"
    np.savez_compressed(raw, **arrays)
    receipt["cases"][0]["raw_sha256"] = audit.sha256(raw)
    with pytest.raises(ValueError):
        run(replay)


@pytest.mark.parametrize("change", ["coverage", "extra_check", "formal", "incomplete"])
def test_receipt_mutations_are_rejected(replay, change):
    receipt = replay[3]
    if change == "coverage":
        receipt["cases"] = []
    elif change == "extra_check":
        receipt["cases"][0]["checks"]["invented"] = {"passed": True}
    elif change == "formal":
        receipt["formal_eligible"] = True
    else:
        receipt["complete"] = False
    with pytest.raises(ValueError):
        run(replay)


def test_tolerance_boundary_is_inclusive_and_does_not_expand():
    expected = np.zeros(1, "f8")
    assert audit.compare(np.array([2e-5]), expected, 2e-5, 2e-4)["passed"]
    assert not audit.compare(np.array([np.nextafter(2e-5, 1)]), expected, 2e-5, 2e-4)["passed"]
