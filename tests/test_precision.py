"""Independent-oracle and fault-injection tests; no native compilation."""

import numpy as np
import pytest
import torch

from transformer.precision import (
    ORACLE_TOLERANCES,
    PRECISION_SEEDS,
    attention_reference,
    attention_stress_cases,
    expected_precision_check_names,
    expected_precision_diagnostic_names,
    precision_case_specs,
)
from transformer.validate import Checks, check_attention_precision


def torch_attention(qkv, heads, dy):
    values = torch.tensor(qkv, dtype=torch.float64, requires_grad=True)
    tokens, width = values.shape[0], values.shape[1] // 3
    q, k, v = [
        part.reshape(tokens, heads, width // heads).transpose(0, 1)
        for part in values.chunk(3, dim=-1)
    ]
    scores = q @ k.transpose(-1, -2) / np.sqrt(width // heads)
    probs = scores.masked_fill(
        torch.ones(tokens, tokens, dtype=torch.bool).triu(1), -torch.inf
    ).softmax(-1)
    output = (probs @ v).transpose(0, 1).reshape(tokens, width)
    (output * torch.tensor(dy, dtype=torch.float64)).sum().backward()
    return {
        "output": output.detach().numpy(),
        "probabilities": probs.detach().numpy(),
        "dqkv": values.grad.numpy(),
    }


@pytest.mark.parametrize("width", [512, 1024])
def test_independent_analytic_attention_matches_fp64_autograd(width):
    torch.set_num_threads(1)
    for case in attention_stress_cases(width, split="development"):
        expected = torch_attention(case["qkv"], case["heads"], case["dy"])
        actual = attention_reference(case["qkv"], case["heads"], case["dy"])
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], **ORACLE_TOLERANCES)


def test_analytic_attention_derivative_has_independent_finite_difference_check():
    rng = np.random.default_rng(2041)
    qkv = rng.normal(0, 0.3, (3, 24))
    dy = rng.normal(0, 0.2, (3, 8))
    expected = attention_reference(qkv, 2, dy)["dqkv"]
    for index in ((0, 0), (1, 2), (0, 10), (2, 15), (1, 16), (2, 23)):
        positive, negative = qkv.copy(), qkv.copy()
        positive[index] += 1e-5
        negative[index] -= 1e-5
        derivative = (
            np.sum(attention_reference(positive, 2)["output"] * dy)
            - np.sum(attention_reference(negative, 2)["output"] * dy)
        ) / 2e-5
        assert derivative == pytest.approx(expected[index], abs=1e-10, rel=1e-7)


def test_precision_fixtures_are_reproducible_disjoint_and_have_intended_score_geometry():
    assert PRECISION_SEEDS["development"] != PRECISION_SEEDS["admission"]
    for width in (512, 1024):
        first = list(attention_stress_cases(width, split="development"))
        repeat = list(attention_stress_cases(width, split="development"))
        admission = list(attention_stress_cases(width, split="admission"))
        assert len(first) == len(precision_case_specs(width)) == 23
        assert len(expected_precision_check_names(width)) == 437
        for a, b, c in zip(first, repeat, admission, strict=True):
            assert a["name"] == b["name"] == c["name"]
            assert np.array_equal(a["qkv"], b["qkv"])
            assert np.array_equal(a["dy"], b["dy"])
            assert not np.array_equal(a["qkv"], c["qkv"])
            assert a["qkv"].dtype == np.float32
            if a["kind"] == "controlled":
                scores = attention_reference(a["qkv"], 4)["scores"][:, -1]
                np.testing.assert_allclose(
                    scores[:, 0], a["target_common_score"], atol=1e-3, rtol=0
                )
                np.testing.assert_allclose(
                    scores[:, 0] - scores[:, 1], a["target_score_gap"], atol=1e-3, rtol=0
                )
                # Both competitors have meaningful probability: these are not
                # the old widely separated, saturated-attention examples.
                probs = attention_reference(a["qkv"], 4)["probabilities"][:, -1]
                assert np.all(probs[:, :2] > 0.2)


class OracleEngine:
    """Idealized correct engine or one explicit algorithmic fault."""

    def __init__(self, fault=None):
        self.fault = fault

    def call(self, name, *args):
        if name == "attention_forward":
            qkv, output, probabilities, shape = args
            heads = int(shape[2])
            result = attention_reference(qkv, heads)
            if self.fault == "early_score_rounding":
                # A plausible incorrect precision design: form accurate dots,
                # round the uncentered scores, then subtract their maximum.
                scores = result["scores"].astype(np.float32).astype(np.float64)
                p = np.exp(scores - scores.max(axis=-1, keepdims=True))
                p /= p.sum(axis=-1, keepdims=True)
                result = attention_reference(qkv, heads, probabilities=p)
            output[:] = result["output"]
            probabilities[:] = result["probabilities"]
            if self.fault == "future_leak" and len(qkv) > 1:
                output[0, 0] += qkv[-1, 0]
        elif name == "attention_backward":
            qkv, probabilities, dy, derivative, shape = args
            derivative[:] = attention_reference(
                qkv, int(shape[2]), dy, probabilities=probabilities
            )["dqkv"]
            if self.fault == "backward":
                derivative[0, 0] += 1.0
        else:
            raise AssertionError(name)


@pytest.mark.parametrize("fault", [None, "early_score_rounding", "future_leak", "backward"])
def test_precision_gate_detects_score_storage_causality_and_derivative_faults(fault):
    torch.set_num_threads(1)
    checks = Checks()
    check_attention_precision(OracleEngine(fault), checks, width=512, split="development")
    assert [record["name"] for record in checks.records] == expected_precision_check_names(512)
    assert [
        record["name"] for record in checks.precision_diagnostics
    ] == expected_precision_diagnostic_names(512)
    assert checks.passed is (fault is None)
    failed = [record["name"] for record in checks.records if not record["passed"]]
    if fault == "early_score_rounding":
        assert any(name.endswith("/probabilities_fp64") for name in failed)
    elif fault == "future_leak":
        assert any(name.endswith("/future_intervention_exact") for name in failed)
    elif fault == "backward":
        assert any(name.endswith("/dqkv_fp64") for name in failed)
    # The new compatibility diagnostics never alter the original legacy schema.
    assert checks.diagnostics == []


@pytest.mark.parametrize(
    "qkv,heads,dy,probabilities",
    [
        (np.zeros((0, 24)), 2, None, None),
        (np.zeros((3, 23)), 2, None, None),
        (np.zeros((3, 24)), 3, None, None),
        (np.full((3, 24), np.nan), 2, None, None),
        (np.zeros((3, 24)), 2, np.zeros((3, 7)), None),
        (np.zeros((3, 24)), 2, None, np.ones((2, 3, 3))),
    ],
)
def test_fp64_oracle_rejects_malformed_or_noncausal_operands(qkv, heads, dy, probabilities):
    with pytest.raises(ValueError):
        attention_reference(qkv, heads, dy, probabilities=probabilities)
