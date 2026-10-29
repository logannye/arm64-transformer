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
