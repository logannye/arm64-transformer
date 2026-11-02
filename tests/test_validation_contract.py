"""Fault injection for the revised gate; no native execution or performance work."""

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from contract_fixtures import attach_contract

from transformer import validate
from transformer.backend import validate_validation_contract
from transformer.validate import Checks, check_common_state


def test_diagnostic_magnitude_failure_stays_false_but_does_not_count_as_passing_gate():
    checks = Checks()
    checks.close("legacy", [0.0], [1.0], "trajectory", diagnostic=True)
    assert checks.passed
    assert checks.diagnostics[0]["passed"] is False
    assert checks.diagnostics[0]["tolerance"] == "trajectory"
    assert checks.diagnostics[0]["max_absolute_error"] == 1
    assert [record["name"] for record in checks.records] == ["legacy/finite_shape"]


@pytest.mark.parametrize(
    "actual,expected", [([np.nan], [np.nan]), ([np.inf], [np.inf]), ([1, 2], [1]), ([], [1])]
)
def test_diagnostic_domain_failures_always_block(actual, expected):
    checks = Checks()
    checks.close("legacy", actual, expected, "trajectory", diagnostic=True)
    assert not checks.passed
    assert checks.diagnostics[0]["passed"] is False


def test_contract_rejects_legacy_flags_missing_replays_or_erased_failures():
    gate = attach_contract(
        {"checks": [{"name": "synthetic", "passed": True}], "runtime_included": True}, 512
    )
    assert validate_validation_contract(gate, 512) == (73, 1)
    for field, value in (
        ("validation_contract", "strict-free-running-v1"),
        ("validation_contract", "trained-precision-development-v1"),
        ("legacy_failed_checks", []),
        ("legacy_contract_passed", True),
        ("legacy_diagnostics", []),
    ):
        with pytest.raises(ValueError):
            validate_validation_contract(gate | {field: value}, 512)
    missing = copy.deepcopy(gate)
    missing["checks"] = [
        record
        for record in missing["checks"]
        if record["name"] != "common_state/2/python/adam/native/step"
    ]
    with pytest.raises(ValueError, match="Missing|missing"):
        validate_validation_contract(missing, 512)
    with pytest.raises(ValueError):
        validate_validation_contract(gate, 32)


@pytest.mark.parametrize("fault", [None, "gradient", "adam", "nonfinite"])
def test_common_state_gate_detects_gradient_and_optimizer_faults(monkeypatch, fault):
    """A tiny analytic model isolates verifier routing from transformer implementation."""
    monkeypatch.setattr(validate.spec, "PARAM_COUNT", 6)
    monkeypatch.setattr(validate.spec, "PARAMS", {"weight": (0, 4), "bias": (4, 2)})

    def forward(flat, tokens, *, trace=False):
        logits = F.one_hot(tokens, 2).float() @ flat[:4].reshape(2, 2).T + flat[4:]
        return (logits, {"logits": logits}) if trace else logits

    monkeypatch.setattr(validate, "reference_forward", forward)
    monkeypatch.setattr(validate, "workspace_tensor", lambda workspace, name: workspace)
    calls = []

    class ToyEngine:
        def forward(self, params, tokens):
            return forward(torch.tensor(params), torch.tensor(tokens).long()).numpy()

        def backward(self, params, tokens, targets, workspace):
            flat = torch.tensor(params, requires_grad=True)
            loss = F.cross_entropy(
                forward(flat, torch.tensor(tokens).long()), torch.tensor(targets).long()
            )
            loss.backward()
            gradient = flat.grad.numpy().copy()
            if fault == "gradient":
                gradient[0] += 0.01
            if fault == "nonfinite":
                gradient[0] = np.nan
            return float(loss.detach()), gradient

        def call(self, name, params, gradient, m, v, step, count):
            assert name == "adam" and count == 6
            calls.append((step, gradient.copy(), params.copy(), m.copy(), v.copy()))
            flat = torch.tensor(params, requires_grad=True)
            optimizer = torch.optim.Adam([flat], lr=validate.spec.LR, foreach=False)
            optimizer.state[flat] = {
                "step": torch.tensor(float(step - 1)),
                "exp_avg": torch.tensor(m),
                "exp_avg_sq": torch.tensor(v),
            }
            flat.grad = torch.tensor(gradient)
            optimizer.step()
            params[:] = flat.detach().numpy()
            m[:] = optimizer.state[flat]["exp_avg"].numpy()
            v[:] = optimizer.state[flat]["exp_avg_sq"].numpy()
            if fault == "adam":
                params[0] += 0.01

    params = np.array([0.2, -0.3, 0.1, 0.4, 0.05, -0.02], np.float32)
    m, v = np.full(6, 0.02, np.float32), np.full(6, 0.03, np.float32)
    checks = Checks()
    check_common_state(
        ToyEngine(),
        checks,
        params,
        m,
        v,
        np.array([0, 1], np.int32),
        np.array([1, 0], np.int32),
        2,
        "python",
    )
    assert len(calls) == 2
    assert all(
        step == 2
        and np.array_equal(before, params)
        and np.array_equal(mm, m)
        and np.array_equal(vv, v)
        for step, _, before, mm, vv in calls
    )
    assert checks.passed is (fault is None)
    if fault == "gradient":
        assert any(not item["passed"] and "/gradient/" in item["name"] for item in checks.records)
    if fault == "adam":
        assert any(not item["passed"] and "/adam/" in item["name"] for item in checks.records)


@pytest.mark.parametrize(
    "name",
    [
        "common_state/1/native/trace/norm2",
        "common_state/2/python/gradient/head_weight",
        "common_state/3/native/adam/python/parameters",
        "common_state/2/python/adam/native/m",
        "common_state/1/native/adam/native/v",
        "trajectory/2/gradient/ff1_weight/finite_shape",
        "runtime/shared_checkpoint/native/best/test/own_engine_saved_logits_exact",
        "runtime/replay/native/last.f32",
        "runtime/replay/python/step-4/gradient_finite",
    ],
)
def test_contract_rejects_dropped_substantive_check_even_with_recounted_passing_records(name):
    gate = attach_contract(
        {"checks": [{"name": "synthetic", "passed": True}], "runtime_included": True}, 1024
    )
    gate["checks"] = [record for record in gate["checks"] if record["name"] != name]
    gate["check_count"] = len(gate["checks"])
    with pytest.raises(ValueError, match="contract check"):
        validate_validation_contract(gate, 1024)


def test_contract_rejects_erased_legacy_record_even_after_recounting_failure_fields():
    gate = attach_contract(
        {"checks": [{"name": "synthetic", "passed": True}], "runtime_included": True}, 512
    )
    gate["legacy_diagnostics"] = [
        record
        for record in gate["legacy_diagnostics"]
        if record["name"] != "trajectory/1/parameters"
    ]
    gate["legacy_diagnostic_count"] = len(gate["legacy_diagnostics"])
    gate["legacy_failed_checks"] = []
    gate["legacy_contract_passed"] = True
    with pytest.raises(ValueError, match="sequence"):
        validate_validation_contract(gate, 512)


def test_contract_parameter_enumeration_matches_model_family():
    from transformer.backend import PARAMETER_NAMES

    assert PARAMETER_NAMES == tuple(validate.spec.PARAMS)
