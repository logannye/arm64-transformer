"""Small saved-evidence fixtures for prospective admission-contract tests."""

from transformer.backend import expected_legacy_names, required_contract_records


def attach_contract(gate, width):
    wide = width >= 512
    gate["validation_contract"] = "v5-common-state-replay-v1" if wide else "strict-free-running-v1"
    gate["checks"] = [{"name": "synthetic", "passed": True}]
    if wide:
        for name, tolerance in required_contract_records(
            gate.get("runtime_included") is True
        ).items():
            record = {"name": name, "passed": True}
            if tolerance is not None:
                record["tolerance"] = tolerance
            gate["checks"].append(record)
    gate["check_count"] = len(gate["checks"])
    diagnostics = []
    if wide:
        for name in expected_legacy_names(gate.get("runtime_included") is True):
            record = {
                "name": name,
                "passed": name != "trajectory/1/parameters",
                "admission_role": "diagnostic",
            }
            if name != "runtime/python/best_epoch":
                record.update(tolerance="trajectory", max_absolute_error=0.0004)
            diagnostics.append(record)
    gate["legacy_diagnostics"] = diagnostics
    gate["legacy_diagnostic_count"] = len(diagnostics)
    gate["legacy_failed_checks"] = [
        record["name"] for record in diagnostics if not record["passed"]
    ]
    gate["legacy_contract_passed"] = not gate["legacy_failed_checks"]
    return gate


def attach_precision(gate, width):
    from transformer.precision import (
        ORACLE_TOLERANCES,
        PRECISION_CONTRACT,
        PRECISION_SEEDS,
        expected_precision_check_names,
        expected_precision_diagnostic_names,
        precision_case_specs,
    )
    from transformer.validate import TOLERANCES

    gate["tolerances"] = TOLERANCES.copy()
    tolerances = {
        "output_fp64": "forward",
        "probabilities_fp64": "math",
        "dqkv_fp64": "gradient",
        "dqkv_same_probabilities_fp64": "gradient",
        "probability_mass": "math",
        "prefix_dqkv_fp64": "gradient",
    }
    for name in expected_precision_check_names(width):
        record = {"name": name, "passed": True}
        tolerance = tolerances.get(name.rsplit("/", 1)[-1])
        if tolerance is not None:
            record["tolerance"] = tolerance
        gate["checks"].append(record)
    for name in ("audit/precision_checks_complete", "audit/precision_diagnostics_complete"):
        gate["checks"].append({"name": name, "passed": True})
    diagnostics = [
        {
            "name": name,
            "passed": False,
            "admission_role": "diagnostic",
            "tolerance": {"output": "forward", "probabilities": "math", "dqkv": "gradient"}[
                name.rsplit("/", 1)[-1]
            ],
        }
        for name in expected_precision_diagnostic_names(width)
    ]
    gate["precision"] = {
        "contract": PRECISION_CONTRACT,
        "fixture_stream": "admission",
        "fixture_seed": PRECISION_SEEDS["admission"],
        "case_count": len(precision_case_specs(width)),
        "oracle_tolerances": ORACLE_TOLERANCES,
        "diagnostics": diagnostics,
        "diagnostic_count": len(diagnostics),
        "failed_diagnostics": [record["name"] for record in diagnostics],
    }
    gate["check_count"] = len(gate["checks"])
    return gate
