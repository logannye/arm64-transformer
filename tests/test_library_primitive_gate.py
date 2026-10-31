"""Prospective primitive gate and dual-oracle checks, hosted at all six widths.

Prototype source here is distinct from real captured-package/device execution.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from transformer import spec

DIRECTORY = Path(__file__).resolve().parents[1] / "integration/precision-library-v1"


def load(name, filename, package=False):
    module_spec = importlib.util.spec_from_file_location(
        name, DIRECTORY / filename, submodule_search_locations=[str(DIRECTORY)] if package else None
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


GATE = load("tested_library_primitive_gate", "validate_runtime.py")
AUDIT = load("tested_independent_library_primitive_gate", "audit_runtime.py")
API = load("primitive_gate_prototype_api", "torch_api.py", True)


@pytest.mark.parametrize("operation", GATE.OPERATIONS)
@pytest.mark.parametrize("index", (0, 1))
def test_cpu_prototype_wide_primitive_and_every_derivative_matches_independent_oracles(
    operation, index
):
    values, upstream = GATE.operands(np, spec.WIDTH, operation, index)
    numpy64 = GATE.numpy_oracle(np, operation, values, upstream)
    torch64 = GATE.torch_oracle(torch, operation, values, upstream)
    arrays = {}
    for label, observed in (
        ("numpy64", (numpy64[0], *numpy64[1])),
        ("torch64", (torch64[0], *torch64[1])),
    ):
        for name, value in zip(
            ("output", "gradient0", "gradient1", "gradient2"), observed, strict=True
        ):
            arrays[label + "_" + name] = value
    for label in ("actual", "repeat"):
        observed, proof = GATE.torch_execution(
            torch, np, API, operation, values, upstream, "cpu", bool(index)
        )
        assert proof["noncontiguous_operands"] is bool(index)
        for name, value in zip(
            ("output", "gradient0", "gradient1", "gradient2"), observed, strict=True
        ):
            arrays[label + "_" + name] = value
    arrays.update({"input" + str(i): value for i, value in enumerate(values)})
    arrays["upstream"] = upstream
    probability_basis = operation == "scaled_dot_product_attention" and index == 1
    result = GATE.checks(np, arrays, probability_basis=probability_basis)
    assert all(
        row["agreement"] and row["dual_oracle"] and row["exact_replay"] for row in result.values()
    )
    AUDIT.validate_shapes(arrays, spec.WIDTH, operation)
    _, independently_checked = AUDIT.arrays_decision(arrays, probability_basis=probability_basis)
    assert independently_checked == result
    if probability_basis:
        assert result["probabilities"]["atol"] == 2e-7
        assert result["probabilities"]["rtol"] == 2e-6
        arrays["input2"][0, 0, 0, 0] = 0.5
        with pytest.raises(ValueError, match="probability basis"):
            AUDIT.arrays_decision(arrays, probability_basis=True)
    arrays["input0"] = arrays["input0"][..., :-1]
    with pytest.raises(ValueError):
        AUDIT.validate_shapes(arrays, spec.WIDTH, operation)


def test_primitive_checker_rejects_wrong_dtype_oracle_and_signed_zero_replay_changes():
    arrays = {}
    for name in ("output", "gradient0", "gradient1", "gradient2"):
        for label in ("actual", "repeat", "numpy64", "torch64"):
            arrays[label + "_" + name] = np.zeros((2,), "f8" if label.endswith("64") else "f4")
    baseline = GATE.checks(np, arrays)
    assert baseline["output"]["atol"] == 2e-5 and baseline["output"]["rtol"] == 2e-4
    assert baseline["gradient0"]["atol"] == 3e-5 and baseline["gradient0"]["rtol"] == 3e-3
    arrays["repeat_output"][0] = -0.0
    assert not GATE.checks(np, arrays)["output"]["exact_replay"]
    arrays["numpy64_gradient0"][0] = 1.0
    changed = GATE.checks(np, arrays)["gradient0"]
    assert not changed["agreement"] and not changed["dual_oracle"]
    arrays["actual_output"] = arrays["actual_output"].astype("f8")
    with pytest.raises(ValueError):
        GATE.checks(np, arrays)


def test_cpu_prototype_malformed_input_gate_is_not_silently_skipped():
    cases = GATE.malformed_inputs(API, "cpu_torch", torch)
    assert len(cases) == 6 and all(row["rejected"] and row["message"] for row in cases)
