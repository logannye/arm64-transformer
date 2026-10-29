"""CPU-only independent GPU oracle/fixture tests; never inspect admission data."""

import numpy as np
import pytest
import torch

from transformer.gpu_precision import (
    GPU_ADMISSION_FROZEN,
    GPU_PRECISION_SEEDS,
    GPU_PRECISION_TOLERANCES,
    GPU_WIDTHS,
    attention_case_specs,
    attention_reference,
    attention_stress_cases,
    layernorm_qkv_case_specs,
    layernorm_qkv_reference,
    layernorm_qkv_stress_cases,
    precision_inventory,
)
from transformer.precision import PRECISION_SEEDS
from transformer.validate import TOLERANCES


def numpy_reference(case):
    return layernorm_qkv_reference(
        *(case[key] for key in ("x", "gamma", "beta", "weight", "bias")),
        case["dqkv"],
        epsilon=case["epsilon"],
    )


def torch_reference(case):
    inputs = {
        key: torch.tensor(case[key], dtype=torch.float64, requires_grad=True)
        for key in ("x", "gamma", "beta", "weight", "bias")
    }
    x = inputs["x"]
    centered = x - x.mean(-1, keepdim=True)
    normalized = centered / torch.sqrt(
        (centered * centered).mean(-1, keepdim=True) + case["epsilon"]
    )
    ln = normalized * inputs["gamma"] + inputs["beta"]
    qkv = torch.nn.functional.linear(ln, inputs["weight"], inputs["bias"])
    (qkv * torch.tensor(case["dqkv"], dtype=torch.float64)).sum().backward()
    return {
        "normalized": normalized.detach().numpy(),
        "ln": ln.detach().numpy(),
        "qkv": qkv.detach().numpy(),
        **{"d" + key: value.grad.numpy() for key, value in inputs.items()},
    }


def test_stream_and_tolerance_contract_does_not_reuse_cpu_admission():
    assert GPU_PRECISION_SEEDS == {"development": 20261103, "admission": 20261104}
    assert set(GPU_PRECISION_SEEDS.values()).isdisjoint(PRECISION_SEEDS.values())
    assert not GPU_ADMISSION_FROZEN
    for key in ("forward", "gradient", "math"):
        assert GPU_PRECISION_TOLERANCES[key] == TOLERANCES[key]
    assert GPU_PRECISION_TOLERANCES["oracle"] == {"atol": 1e-10, "rtol": 1e-10}
    for generator in (attention_stress_cases, layernorm_qkv_stress_cases):
        with pytest.raises(ValueError, match="admission is closed"):
            next(generator(32, split="admission"))


@pytest.mark.parametrize("width", GPU_WIDTHS)
def test_named_development_inventory_and_shapes(width):
    inventory = precision_inventory(width)
    assert inventory["admission_frozen"] is False
    assert len(inventory["attention"]) == len(attention_case_specs(width)) == 23
    assert len(inventory["layernorm_qkv"]) == len(layernorm_qkv_case_specs(width)) == 22
    assert len({case["name"] for case in inventory["attention"]}) == 23
    assert len({case["name"] for case in inventory["layernorm_qkv"]}) == 22
    assert {case["rows"] for case in inventory["layernorm_qkv"]} == {1, 3, 32, 33, 511, 512}
    assert all(case["weight_shape"] == [3 * width, width] for case in inventory["layernorm_qkv"])


def test_exact_fp32_development_fixtures_are_reproducible():
    for generator, fields in (
        (attention_stress_cases, ("qkv", "dy")),
        (layernorm_qkv_stress_cases, ("x", "gamma", "beta", "weight", "bias", "dqkv")),
    ):
        for a, b in zip(generator(32), generator(32), strict=True):
            assert a["name"] == b["name"]
            for field in fields:
                assert a[field].dtype == np.float32 and a[field].flags.c_contiguous
                assert np.isfinite(a[field]).all()
                np.testing.assert_array_equal(a[field], b[field])
            assert np.any(a[fields[-1]] != 0)


@pytest.mark.parametrize("width", [32, 512, 1024])
def test_independent_ln_qkv_fp64_forward_and_all_analytic_gradients(width):
    torch.set_num_threads(1)
    selected = {
        "random_rows3",
        "offset_4096_spread_0.000244141",
        "offset_64_spread_0",
        "affine_signed_scale",
        "qkv_common_bias",
        "qkv_reduction_cancellation",
    }
    for case in layernorm_qkv_stress_cases(width):
        if width != 32 and case["name"] not in selected:
            continue
        actual, expected = numpy_reference(case), torch_reference(case)
        for key, value in expected.items():
            np.testing.assert_allclose(
                actual[key],
                value,
                **GPU_PRECISION_TOLERANCES["oracle"],
                err_msg=f"{case['name']}/{key}",
            )


def test_epsilon_prevents_constant_input_singularity_without_erasing_dx():
    case = next(
        case for case in layernorm_qkv_stress_cases(32) if case["name"] == "offset_64_spread_0"
    )
    result = numpy_reference(case)
    assert np.array_equal(result["normalized"], np.zeros_like(result["normalized"]))
    assert np.isfinite(result["dx"]).all() and np.any(result["dx"] != 0)
    np.testing.assert_allclose(result["dx"].sum(-1), 0, atol=1e-10, rtol=0)


def test_ln_qkv_gradients_have_independent_finite_difference_check():
    case = next(layernorm_qkv_stress_cases(32))
    operands = {
        key: case[key].astype(np.float64) for key in ("x", "gamma", "beta", "weight", "bias")
    }
    reference = layernorm_qkv_reference(**operands, dqkv=case["dqkv"])
    for name, coordinate in (
        ("x", (0, 2)),
        ("gamma", (3,)),
        ("beta", (7,)),
        ("weight", (8, 5)),
        ("bias", (4,)),
    ):
        positive = {key: value.copy() for key, value in operands.items()}
        negative = {key: value.copy() for key, value in operands.items()}
        positive[name][coordinate] += 1e-5
        negative[name][coordinate] -= 1e-5
        derivative = (
            np.sum(layernorm_qkv_reference(**positive)["qkv"] * case["dqkv"])
            - np.sum(layernorm_qkv_reference(**negative)["qkv"] * case["dqkv"])
        ) / 2e-5
        assert derivative == pytest.approx(reference["d" + name][coordinate], abs=2e-7, rel=2e-5)


def test_attention_separates_exact_math_and_saved_probability_derivatives():
    case = next(
        case for case in attention_stress_cases(32) if case["name"] == "offset_4096_gap_0.125"
    )
    mathematical = attention_reference(case["qkv"], 4, case["dy"])
    rounded = mathematical["probabilities"].astype(np.float32)
    saved = attention_reference(case["qkv"], 4, case["dy"], probabilities=rounded)
    np.testing.assert_array_equal(saved["probabilities"], rounded.astype(np.float64))
    assert not np.array_equal(saved["dqkv"], mathematical["dqkv"])
    assert np.isfinite(saved["dqkv"]).all()
    changed = case["qkv"].copy()
    changed[-1] *= np.float32(-3)
    future = attention_reference(changed, 4, case["dy"])
    np.testing.assert_array_equal(future["output"][:-1], mathematical["output"][:-1])


@pytest.mark.parametrize("width", [32, 512, 1024])
def test_new_attention_development_stream_matches_independent_fp64_autograd(width):
    torch.set_num_threads(1)
    for case in attention_stress_cases(width):
        values = torch.tensor(case["qkv"], dtype=torch.float64, requires_grad=True)
        tokens = case["tokens"]
        q, k, v = [
            part.reshape(tokens, 4, width // 4).transpose(0, 1) for part in values.chunk(3, dim=-1)
        ]
        scores = q @ k.transpose(-1, -2) / np.sqrt(width // 4)
        mask = torch.ones(tokens, tokens, dtype=torch.bool).triu(1)
        probs = scores.masked_fill(mask, -torch.inf).softmax(-1)
        output = (probs @ v).transpose(0, 1).reshape(tokens, width)
        (output * torch.tensor(case["dy"], dtype=torch.float64)).sum().backward()
        actual = attention_reference(case["qkv"], 4, case["dy"])
        expected = {
            "output": output.detach().numpy(),
            "probabilities": probs.detach().numpy(),
            "dqkv": values.grad.numpy(),
        }
        for key, value in expected.items():
            np.testing.assert_allclose(
                actual[key],
                value,
                **GPU_PRECISION_TOLERANCES["oracle"],
                err_msg=f"{case['name']}/{key}",
            )


@pytest.mark.parametrize("fault", ["shape", "nan", "epsilon", "cotangent"])
def test_ln_qkv_oracle_rejects_invalid_operands(fault):
    case = next(layernorm_qkv_stress_cases(32))
    operands = {key: case[key] for key in ("x", "gamma", "beta", "weight", "bias")}
    if fault == "shape":
        operands["weight"] = operands["weight"].T
    elif fault == "nan":
        operands["x"] = operands["x"].copy()
        operands["x"][0, 0] = np.nan
    elif fault == "epsilon":
        operands["epsilon"] = 0.0
    else:
        operands["dqkv"] = case["dqkv"][:, :-1]
    with pytest.raises(ValueError):
        layernorm_qkv_reference(**operands)
