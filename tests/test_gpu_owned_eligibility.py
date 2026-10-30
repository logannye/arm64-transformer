"""Full-suite completeness and stale-authority controls; no framework execution."""

import copy
import json
import xml.etree.ElementTree as ET

import pytest

from transformer import gpu_owned_eligibility as gate
from transformer import gpu_study as study


def test_original_inventory_and_failure_eligibility_are_frozen():
    contract = gate.suite_contract()
    mps, mlx = contract["pytorch"], contract["mlx"]
    assert len(mps["test_ids"]) == len(set(mps["test_ids"])) == 10353
    assert len(mps["skip_ids"]) == 744
    assert len(mps["expected_failure_ids"]) == 382
    assert len(contract["prospective_unselected_pytorch_additions"]) == 4
    assert len(mlx["python_cases"]) == 1053
    assert sum(row["status"] == "skipped" for row in mlx["python_cases"]) == 7
    assert len(mlx["test_source_sha256"]) == 73


@pytest.fixture
def mps():
    contract = gate.suite_contract()
    expected = contract["pytorch"]
    tests = {
        "run": 10353,
        "executed_ids": copy.deepcopy(expected["test_ids"]),
        "skipped": [
            {"id": name, "reason": "retained upstream skip"} for name in expected["skip_ids"]
        ],
        "expected_failures": [
            {"id": name, "traceback": "retained upstream expected failure"}
            for name in expected["expected_failure_ids"]
        ],
        "errors": [],
        "failures": [],
        "unexpected_successes": [],
        "successful": True,
    }
    return contract, tests


def test_full_mps_raw_inventory_passes_only_complete_original_selection(mps):
    contract, tests = mps
    assert gate.validate_mps(tests, contract)["cases"] == 10353


@pytest.mark.parametrize(
    "mutation",
    [
        "drop",
        "duplicate",
        "reverse",
        "skip",
        "xfail",
        "failure",
        "error",
        "xpass",
        "short",
        "false",
    ],
)
def test_full_mps_cannot_remove_failed_or_ineligible_methods(mps, mutation):
    contract, tests = mps
    if mutation == "drop":
        tests["executed_ids"].pop()
    elif mutation == "duplicate":
        tests["executed_ids"][-1] = tests["executed_ids"][0]
    elif mutation == "reverse":
        tests["executed_ids"].reverse()
    elif mutation == "skip":
        tests["skipped"].pop()
    elif mutation == "xfail":
        tests["expected_failures"].pop()
    elif mutation in ("failure", "error"):
        tests[mutation + "s"].append(
            {"id": tests["executed_ids"][0], "traceback": "real failure retained"}
        )
    elif mutation == "xpass":
        tests["unexpected_successes"] = [tests["executed_ids"][0]]
    elif mutation == "short":
        tests["run"] = 34
    else:
        tests["successful"] = False
    with pytest.raises(ValueError):
        gate.validate_mps(tests, contract)


@pytest.fixture
def mlx_files(tmp_path):
    contract = gate.suite_contract()
    xml = tmp_path / "python.xml"
    root = ET.Element("testsuite")
    for row in contract["mlx"]["python_cases"]:
        case = ET.SubElement(root, "testcase", classname=row["class"], name=row["name"])
        if row["status"] == "skipped":
            ET.SubElement(case, "skipped", message=row["reason"])
    ET.ElementTree(root).write(xml)
    log = tmp_path / "python.log"
    log.write_text("1046 passed, 7 skipped, 17701 subtests passed in 2.0s\n")
    cpp = tmp_path / "cpp.log"
    cpp.write_text(
        "[doctest] test cases:305|305 passed|0 failed|0 skipped\n"
        "[doctest] assertions:4792|4792 passed|0 failed\n"
    )
    return contract, xml, log, cpp


def test_mlx_raw_full_suites_revalidate_without_importing_framework(mlx_files):
    contract, xml, log, cpp = mlx_files
    assert gate.validate_mlx_python(xml, log, contract)["cases"] == 1053
    assert gate.validate_mlx_cpp(cpp, contract) == {"cases": 305, "assertions": 4792}


@pytest.mark.parametrize(
    "mutation",
    ["xml_drop", "xml_error", "xml_skip", "subtest", "cpp_case", "cpp_assert", "duplicate_log"],
)
def test_mlx_full_inventory_eligibility_and_raw_totals_are_not_pass_flags(mlx_files, mutation):
    contract, xml, log, cpp = mlx_files
    if mutation.startswith("xml"):
        tree = ET.parse(xml)
        case = tree.getroot()[0]
        if mutation == "xml_drop":
            tree.getroot().remove(case)
        elif mutation == "xml_error":
            ET.SubElement(case, "error", message="actual error")
        else:
            ET.SubElement(case, "skipped", message="new waiver")
        tree.write(xml)
    elif mutation == "subtest":
        log.write_text(log.read_text().replace("17701", "17700"))
    elif mutation == "cpp_case":
        cpp.write_text(cpp.read_text().replace("305", "304"))
    elif mutation == "cpp_assert":
        cpp.write_text(cpp.read_text().replace("4792", "4791"))
    else:
        log.write_text(log.read_text() * 2)
    with pytest.raises(ValueError):
        if mutation.startswith("cpp"):
            gate.validate_mlx_cpp(cpp, contract)
        else:
            gate.validate_mlx_python(xml, log, contract)


def test_suite_stale_runtime_rejected_before_any_raw_outcome_is_read(tmp_path):
    manifest = study.planned_manifest("owned")
    manifest["bindings"] = {
        "runtime": {"native": "0" * 64},
        "baseline": {"owned-runtime.json": "1" * 64},
    }
    record = {
        "schema": "gpu-owned-suite-execution-v1",
        "protocol": gate.OWNED.protocol,
        "name": "mlx_python",
        "complete": True,
        "integrity_after": True,
        "formal_admission_opened": False,
        "formal_fits": 0,
        "runtime_identity_sha256_before": "old-runtime",
        "runtime_identity_sha256_after": "old-runtime",
    }
    with pytest.raises(ValueError, match="stale"):
        gate.validate_suite(record, tmp_path, manifest, gate.suite_contract(), "mlx_python")
    record["integrity_after"] = False
    with pytest.raises(ValueError, match="finish"):
        gate.validate_suite(record, tmp_path, manifest, gate.suite_contract(), "mlx_python")


def test_eligibility_never_accepts_smoke_or_boolean_as_full_suites(tmp_path):
    manifest = study.planned_manifest("owned")
    path = tmp_path / "owned-eligibility.json"
    path.write_text(json.dumps({"passed": True, "formal_eligible": True}))
    manifest["bindings"] = {"baseline": {path.name: study.digest(path)}}
    with pytest.raises(ValueError, match="missing"):
        gate.verify_eligibility(manifest, {"baseline": tmp_path})


def test_generic_passing_mps_suite_cannot_bypass_unfinished_archive_audit(tmp_path):
    record = {
        "schema": "gpu-owned-suite-execution-v1",
        "protocol": gate.OWNED.protocol,
        "name": "pytorch_mps",
        "complete": True,
        "integrity_after": True,
        "passed": True,
    }
    with pytest.raises(ValueError, match="MPS archive qualification remains blocked"):
        gate.validate_suite(record, tmp_path, {}, gate.suite_contract(), "pytorch_mps")
