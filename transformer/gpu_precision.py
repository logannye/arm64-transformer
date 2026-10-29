"""CPU FP64 oracles and prospective GPU development stress operands.

This module imports no GPU runtime. Fixtures contain exact stored FP32 operands;
the oracle evaluates those values in FP64, not idealized construction targets.
Admission generation stays closed until the complete GPU contract is frozen.
"""

from __future__ import annotations

import math

import numpy as np

from transformer.precision import attention_reference as attention_reference
from transformer.precision import precision_case_specs

GPU_PRECISION_CONTRACT = "gpu-v1-precision-development-v1"
GPU_PRECISION_SEEDS = {"development": 20261103, "admission": 20261104}
GPU_ADMISSION_FROZEN = False
GPU_WIDTHS = (32, 64, 128, 256, 512, 1024)
GPU_PRECISION_TOLERANCES = {
    "forward": {"atol": 2e-5, "rtol": 2e-4},
    "gradient": {"atol": 3e-5, "rtol": 3e-3},
    "math": {"atol": 2e-7, "rtol": 2e-6},
    "oracle": {"atol": 1e-10, "rtol": 1e-10},
}
LAYERNORM_EPSILON = 1e-5


def _validate_stream(width, split):
    if width not in GPU_WIDTHS or split not in GPU_PRECISION_SEEDS:
        raise ValueError("Unknown GPU width or precision stream")
    if split == "admission" and not GPU_ADMISSION_FROZEN:
        raise ValueError("GPU admission is closed until its complete contract is committed")


def _rng(width, split, family, case):
    _validate_stream(width, split)
    return np.random.default_rng(
        np.random.SeedSequence([GPU_PRECISION_SEEDS[split], width, family, case])
    )


def attention_case_specs(width):
    """Exercise all 23 CPU attention patterns at every supported GPU width."""
    if width not in GPU_WIDTHS:
        raise ValueError("Unknown GPU width")
    # This is a new stream and broader inventory, not reuse of CPU admission data.
    return list(precision_case_specs(512))


def attention_stress_cases(width, *, split="development"):
    """Yield unbatched [T,3W] operands; callers may add an explicit B=1 axis."""
    _validate_stream(width, split)
    heads, head_width = 4, width // 4
    for index, specification in enumerate(attention_case_specs(width)):
        name, tokens, kind, magnitude, gap, cancellation = specification
        rng = _rng(width, split, 1, index)
        qkv = rng.normal(0, 0.3, (tokens, 3 * width)).astype(np.float32)
        if kind == "controlled":
            q, k, v = [part.reshape(tokens, heads, head_width) for part in np.split(qkv, 3, axis=1)]
            for head in range(heads):
                direction = rng.choice([-1.0, 1.0], head_width) * rng.uniform(
                    0.75, 1.25, head_width
                )
                norm = np.dot(direction, direction)
                offsets = -np.linspace(3.0, 9.0, tokens)
                offsets[0] = 0.0
                if tokens > 1:
                    offsets[1] = -gap
                noise = rng.normal(0, max(cancellation, 0.25), (tokens, head_width))
                noise -= (noise @ direction)[:, None] * direction[None] / norm
                k[:, head] = ((magnitude + offsets) * math.sqrt(head_width) / norm)[
                    :, None
                ] * direction + noise
                q[:, head] = direction
                contrast = rng.uniform(1.0, 3.0, head_width)
                v[0, head] = contrast
                if tokens > 1:
                    v[1, head] = -contrast
        yield {
            "name": name,
            "tokens": tokens,
            "width": width,
            "heads": heads,
            "kind": kind,
            "target_common_score": magnitude,
            "target_score_gap": gap,
            "cancellation_scale": cancellation,
            "qkv": np.ascontiguousarray(qkv),
            "dy": rng.normal(0, 0.2, (tokens, width)).astype(np.float32),
        }


def layernorm_qkv_case_specs(width):
    """Development inventory; no accuracy outcome selects or removes a case."""
    if width not in GPU_WIDTHS:
        raise ValueError("Unknown GPU width")
    cases = [(f"random_rows{rows}", rows, "random", 0.0, 1.0) for rows in (1, 3, 32, 33, 511, 512)]
    cases.extend(
        (f"offset_{offset:g}_spread_{spread:g}", 3, "offset", offset, spread)
        for offset in (0.0, 64.0, 4096.0)
        for spread in (0.0, 2.0**-12, 0.01, 1.0)
    )
    cases.extend(
        [
            ("affine_signed_scale", 3, "affine", 0.0, 1.0),
            ("affine_zero_channels", 3, "zero_channels", 0.0, 1.0),
            ("qkv_common_bias", 32, "qkv_common_bias", 64.0, 0.01),
            ("qkv_reduction_cancellation", 3, "qkv_cancellation", 64.0, 0.01),
        ]
    )
    return cases


def layernorm_qkv_stress_cases(width, *, split="development"):
    """Yield exact FP32 LN/QKV inputs, parameters and nonzero cotangents.

    Weight orientation is [3W,W], matching the persisted CPU flat layout.
    Offset/spread refer to pre-rounding construction only; zero-variance cases
    intentionally exercise epsilon and may have exact zero normalized outputs.
    """
    _validate_stream(width, split)
    for index, specification in enumerate(layernorm_qkv_case_specs(width)):
        name, rows, kind, offset, spread = specification
        rng = _rng(width, split, 2, index)
        x = (offset + rng.normal(0, spread, (rows, width))).astype(np.float32)
        gamma = rng.uniform(0.5, 1.5, width).astype(np.float32)
        beta = rng.normal(0, 0.1, width).astype(np.float32)
        weight = rng.normal(0, 1 / math.sqrt(width), (3 * width, width)).astype(np.float32)
        bias = rng.normal(0, 0.1, 3 * width).astype(np.float32)
        if kind == "affine":
            gamma *= np.resize(np.array([-2.0, 0.125, 4.0, -0.5], np.float32), width)
        elif kind == "zero_channels":
            gamma[::3] = 0.0
        elif kind == "qkv_common_bias":
            weight *= np.float32(0.02)
            bias[: 2 * width] = 32.0
        elif kind == "qkv_cancellation":
            paired = offset + rng.normal(0, spread, (rows, width // 2))
            x[:, 0::2] = paired
            x[:, 1::2] = paired + rng.normal(0, 1e-4, paired.shape)
            gamma[1::2], beta[1::2] = gamma[0::2], beta[0::2]
            positive = rng.normal(0, 64 / math.sqrt(width), (3 * width, width // 2))
            weight[:, 0::2] = positive
            weight[:, 1::2] = -positive + rng.normal(0, 1e-5, positive.shape)
        yield {
            "name": name,
            "rows": rows,
            "width": width,
            "kind": kind,
            "target_input_offset": offset,
            "target_input_spread": spread,
            "epsilon": LAYERNORM_EPSILON,
            "x": np.ascontiguousarray(x),
            "gamma": gamma,
            "beta": beta,
            "weight": weight,
            "bias": bias,
            "dqkv": rng.normal(0, 0.2, (rows, 3 * width)).astype(np.float32),
        }


def layernorm_qkv_reference(x, gamma, beta, weight, bias, dqkv=None, *, epsilon=LAYERNORM_EPSILON):
    """Independent FP64 equations and analytic gradients for LN followed by QKV.

    Float64 operands are accepted for independent finite differences. GPU
    accuracy checks must pass the exact original FP32 fixture operands.
    """
    arrays = [np.asarray(value) for value in (x, gamma, beta, weight, bias)]
    if any(
        not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all()
        for value in arrays
    ):
        raise ValueError("LayerNorm/QKV requires finite floating-point operands")
    x, gamma, beta, weight, bias = [value.astype(np.float64) for value in arrays]
    if (
        x.ndim != 2
        or min(x.shape) < 1
        or gamma.shape != x.shape[1:]
        or beta.shape != gamma.shape
        or weight.shape != (3 * x.shape[1], x.shape[1])
        or bias.shape != (3 * x.shape[1],)
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("Invalid LayerNorm/QKV shape or epsilon")
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    inverse_std = 1 / np.sqrt(variance + epsilon)
    normalized = centered * inverse_std
    ln = normalized * gamma + beta
    qkv = ln @ weight.T + bias
    result = {
        "mean": mean,
        "variance": variance,
        "inverse_std": inverse_std,
        "normalized": normalized,
        "ln": ln,
        "qkv": qkv,
    }
    if dqkv is not None:
        dqkv = np.asarray(dqkv, dtype=np.float64)
        if dqkv.shape != qkv.shape or not np.isfinite(dqkv).all():
            raise ValueError("QKV cotangent must match the finite output")
        dln = dqkv @ weight
        dnormalized = dln * gamma
        dcentered = inverse_std * (
            dnormalized - normalized * np.mean(dnormalized * normalized, axis=-1, keepdims=True)
        )
        # Keep the centering derivative explicit. This avoids assuming that a
        # finite-precision normalized row has an exactly zero arithmetic mean.
        dx = dcentered - dcentered.mean(axis=-1, keepdims=True)
        result.update(
            dx=dx,
            dgamma=np.sum(dln * normalized, axis=0),
            dbeta=dln.sum(axis=0),
            dweight=dqkv.T @ ln,
            dbias=dqkv.sum(axis=0),
            dln=dln,
        )
    return result


def precision_inventory(width):
    """Names and shapes only; does not expose the reserved admission operands."""
    return {
        "contract": GPU_PRECISION_CONTRACT,
        "admission_frozen": GPU_ADMISSION_FROZEN,
        "width": width,
        "heads": 4,
        "attention": [
            {"name": name, "tokens": tokens, "qkv_shape": [tokens, 3 * width]}
            for name, tokens, *_ in attention_case_specs(width)
        ],
        "layernorm_qkv": [
            {
                "name": name,
                "rows": rows,
                "x_shape": [rows, width],
                "weight_shape": [3 * width, width],
            }
            for name, rows, *_ in layernorm_qkv_case_specs(width)
        ],
        "required_attention_checks": [
            "output_fp64",
            "probabilities_fp64",
            "mathematical_dqkv_fp64",
            "same_saved_probabilities_dqkv_fp64",
            "probability_mass",
            "causal_mask_exact_zero",
            "future_intervention_exact",
            "prefix_gradients",
            "future_gradient_exact_zero",
            "repeat_exact",
            "input_immutability",
            "finite_shape",
            "guards",
        ],
        "required_layernorm_qkv_checks": [
            "normalized_fp64",
            "ln_fp64",
            "qkv_fp64",
            "dx_fp64",
            "dgamma_fp64",
            "dbeta_fp64",
            "dweight_fp64",
            "dbias_fp64",
            "epsilon_nonzero_derivative",
            "repeat_exact",
            "input_immutability",
            "finite_shape",
            "guards",
        ],
        "scope": (
            "Development inventory only; full GPU gate names, batched interventions, "
            "optimizer checks and exact admission counts must be committed before admission opens."
        ),
    }
