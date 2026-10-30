"""The branch counterfactual must preserve forward values and expose its altered derivative."""

import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "integration/trained-precision/relu_localize.py"
SPEC = importlib.util.spec_from_file_location("relu_localize", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_gate_changes_only_the_explicit_backward_branch():
    value = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float64, requires_grad=True)
    gate = torch.tensor([True, False, False])
    result = MODULE.frozen_gate_relu(torch, gate)(value)
    assert torch.equal(result, value.relu())
    (result * torch.tensor([2.0, 3.0, 4.0])).sum().backward()
    assert torch.equal(value.grad, torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))


@pytest.mark.parametrize("gate", [torch.ones(2, dtype=torch.bool), torch.ones(3)])
def test_gate_must_have_exact_shape_and_boolean_storage(gate):
    with pytest.raises(ValueError, match="Gate"):
        MODULE.frozen_gate_relu(torch, gate)(torch.ones(3))
