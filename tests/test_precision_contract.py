"""Saved gate fault injection; no training or native timing."""

import copy

import pytest
from contract_fixtures import attach_contract, attach_precision

from transformer.backend import validate_precision_preflight


@pytest.mark.parametrize("width", (32, 512, 1024))
def test_precision_contract_retains_failed_diagnostics(width):
    gate = attach_precision(attach_contract({"runtime_included": True}, width), width)
    result = validate_precision_preflight(gate, width)
    assert result["failed_diagnostics"] == result["diagnostics"] > 0


@pytest.mark.parametrize(
    "mutation", ("missing", "duplicate", "false", "tolerance", "diagnostic", "seed")
)
def test_precision_contract_rejects_incomplete_or_altered_evidence(mutation):
    gate = attach_precision(attach_contract({"runtime_included": True}, 512), 512)
    index = next(
        i for i, record in enumerate(gate["checks"]) if record["name"].endswith("/output_fp64")
    )
    if mutation == "missing":
        gate["checks"].pop(index)
    elif mutation == "duplicate":
        gate["checks"].insert(index, copy.deepcopy(gate["checks"][index]))
    elif mutation == "false":
        gate["checks"][index]["passed"] = False
    elif mutation == "tolerance":
        gate["checks"][index]["tolerance"] = "trajectory"
    elif mutation == "diagnostic":
        gate["precision"]["diagnostics"].pop()
    else:
        gate["precision"]["fixture_seed"] += 1
    with pytest.raises(ValueError, match="precision"):
        validate_precision_preflight(gate, 512)
