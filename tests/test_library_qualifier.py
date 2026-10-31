"""Raw acceptance distinguishes accuracy, repeatability and negative controls."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_qualifier", ROOT / "integration/library-study-v1/qualify.py"
)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def wrapped(kind, result):
    return dict(schema="frozen-library-raw-audit-dispatch-v1", kind=kind, width=512, result=result)


def test_ordinary_failure_is_retained_but_library_accuracy_must_pass():
    job = dict(
        kind="collection",
        width=512,
        name="historical-cpu_torch-w512-ordinary-forward-raw",
        expected_numerics=None,
    )
    value = wrapped(
        "collection",
        dict(
            groups=[
                dict(
                    name=job["name"].removesuffix("-raw"),
                    evidence_valid=True,
                    numerics_passed=False,
                )
            ]
        ),
    )
    assert Q.require_result(job, value) == dict(evidence_valid=True)
    with pytest.raises(ValueError, match="Required raw"):
        Q.require_result(dict(job, expected_numerics=True), value)
    value["result"]["groups"][0]["evidence_valid"] = False
    with pytest.raises(ValueError, match="Invalid raw"):
        Q.require_result(job, value)


def test_exact_optimizer_repetition_does_not_become_an_accuracy_claim():
    job = dict(kind="optimizer", width=512, expected_numerics=True)
    value = wrapped(
        "optimizer", dict(evidence_valid=True, exact_passed=True, accuracy_gate_included=False)
    )
    assert Q.require_result(job, value) == dict(evidence_valid=True, exact_passed=True)
    value["result"]["exact_passed"] = False
    with pytest.raises(ValueError, match="Required raw"):
        Q.require_result(job, value)


def test_full_repeatability_requires_both_exactness_and_accuracy():
    job = dict(kind="repeatability", width=1024, expected_numerics=True)
    for exact, accurate in ((False, True), (True, False)):
        with pytest.raises(ValueError, match="Required raw"):
            Q.require_result(
                job, dict(evidence_valid=True, exact_passed=exact, numerics_passed=accurate)
            )
