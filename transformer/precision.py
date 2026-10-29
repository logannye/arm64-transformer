"""Independent FP64 attention equations and prospective v6 stress fixtures.

All arithmetic in the oracle starts from the exact supplied FP32 operands cast
to FP64. It does not copy the native blocking/reduction order. The two fixture
streams are separate: development may guide implementation, admission is frozen
before the measured campaign. Neither stream contains held-out corpus examples.
"""

from __future__ import annotations

import math

import numpy as np

PRECISION_CONTRACT = "v6-fp64-centered-attention-v1"
PRECISION_SEEDS = {"development": 20261101, "admission": 20261102}
ORACLE_TOLERANCES = {"atol": 1e-10, "rtol": 1e-10}


def attention_reference(qkv, heads, dy=None, *, probabilities=None):
    """Return FP64 causal attention and its analytic reverse-mode derivative.

    Passing probabilities is a diagnostic operation: it isolates the backward
    arithmetic with exactly the same saved probabilities as the native kernel.
    The ordinary oracle always computes its own FP64 probabilities.
    """
    qkv = np.asarray(qkv, dtype=np.float64)
    if qkv.ndim != 2 or qkv.shape[1] % 3 or heads <= 0:
        raise ValueError("qkv must be a two-dimensional array of three equal projections")
    tokens, columns = qkv.shape
    width = columns // 3
    if tokens < 1 or width % heads or not np.isfinite(qkv).all():
        raise ValueError("attention requires finite operands and a positive integral head width")
    head_width = width // heads
    q, k, v = [
        part.reshape(tokens, heads, head_width).transpose(1, 0, 2)
        for part in np.split(qkv, 3, axis=1)
    ]
    scale = 1.0 / math.sqrt(head_width)
    scores = (q @ k.transpose(0, 2, 1)) * scale
    mask = np.tri(tokens, dtype=bool)
    scores = np.where(mask[None], scores, -np.inf)
    centered = scores - scores.max(axis=-1, keepdims=True)
    exponentials = np.exp(centered)
    probs = exponentials / exponentials.sum(axis=-1, keepdims=True)
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (heads, tokens, tokens) or not np.isfinite(probs).all():
            raise ValueError("saved probabilities have an invalid shape or value")
        if np.any(probs[:, ~mask] != 0):
            raise ValueError("saved probabilities violate the causal mask")
    output = (probs @ v).transpose(1, 0, 2).reshape(tokens, width)
    result = {"scores": scores, "probabilities": probs, "output": output}
    if dy is not None:
        dy = np.asarray(dy, dtype=np.float64)
        if dy.shape != output.shape or not np.isfinite(dy).all():
            raise ValueError("upstream derivative must match the finite attention output")
        dh = dy.reshape(tokens, heads, head_width).transpose(1, 0, 2)
        dp = dh @ v.transpose(0, 2, 1)
        ds = probs * (dp - (dp * probs).sum(axis=-1, keepdims=True))
        ds = np.where(mask[None], ds, 0.0) * scale
        derivatives = (ds @ k, ds.transpose(0, 2, 1) @ q, probs.transpose(0, 2, 1) @ dh)
        result["dqkv"] = np.concatenate(
            [part.transpose(1, 0, 2).reshape(tokens, width) for part in derivatives], axis=1
        )
    return result


def precision_case_specs(width):
    """A fixed case inventory, independent of observed implementation errors."""
    cases = [(f"random_t{tokens}", tokens, "random", 0.0, 0.0, 0.0) for tokens in (1, 3, 32)]
    if width < 512:
        return cases
    for magnitude in (0.0, 64.0, 512.0, 4096.0):
        for gap in (0.0, 0.125, 0.75):
            name = f"offset_{magnitude:g}_gap_{gap:g}"
            cases.append((name, 32, "controlled", magnitude, gap, 0.0))
    cases.extend(
        (f"prefix_t{tokens}", tokens, "controlled", 512.0, 0.125, 0.0) for tokens in (2, 3, 5, 31)
    )
    cases.append(("negative_offset", 32, "controlled", -4096.0, 0.125, 0.0))
    cases.extend(
        (f"cancellation_{scale:g}", 32, "controlled", 64.0, 0.125, scale)
        for scale in (16.0, 64.0, 256.0)
    )
    return cases


def attention_stress_cases(width, *, split="admission"):
    """Yield deterministic exact-FP32 Q/K/V inputs and upstream gradients.

    Controlled cases independently vary the common score scale and the gap
    between the first two keys. All later keys have lower scores. Per-key
    orthogonal components and mixed-sign Q values exercise actual dot-product
    rounding; FP64 evaluates the final stored values, not construction targets.
    """
    if width not in (32, 64, 128, 256, 512, 1024) or split not in PRECISION_SEEDS:
        raise ValueError("unknown width or precision fixture stream")
    heads, head_width = 4, width // 4
    rng = np.random.default_rng(np.random.SeedSequence([PRECISION_SEEDS[split], width]))
    for name, tokens, kind, magnitude, gap, cancellation in precision_case_specs(width):
        qkv = rng.normal(0, 0.3, (tokens, 3 * width)).astype(np.float32)
        if kind == "controlled":
            q, k, v = [part.reshape(tokens, heads, head_width) for part in np.split(qkv, 3, axis=1)]
            for head in range(heads):
                direction = rng.choice([-1.0, 1.0], head_width) * rng.uniform(
                    0.75, 1.25, head_width
                )
                norm = np.dot(direction, direction)
                q[:, head] = direction
                offsets = -np.linspace(3.0, 9.0, tokens)
                offsets[0] = 0.0
                if tokens > 1:
                    offsets[1] = -gap
                noise = rng.normal(0, max(cancellation, 0.25), (tokens, head_width))
                noise -= (noise @ direction)[:, None] * direction[None] / norm
                k[:, head] = ((magnitude + offsets) * math.sqrt(head_width) / norm)[
                    :, None
                ] * direction[None] + noise
                # Competing values point in opposite directions so a score-gap
                # error propagates into a visible output discrepancy.
                contrast = rng.uniform(1.0, 3.0, head_width)
                v[0, head] = contrast
                if tokens > 1:
                    v[1, head] = -contrast
        dy = rng.normal(0, 0.2, (tokens, width)).astype(np.float32)
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
            "dy": dy,
        }


PRECISION_CHECK_SUFFIXES = (
    "fp64_oracles_agree",
    "output_fp64",
    "probabilities_fp64",
    "dqkv_fp64",
    "dqkv_same_probabilities_fp64",
    "probability_mass",
    "masked_exact_zero",
    "repeat_forward_exact",
    "future_intervention_exact",
    "prefix_dqkv_fp64",
    "future_gradient_exact_zero",
    "guards",
    "inputs_unchanged",
)
PRECISION_DIAGNOSTIC_SUFFIXES = (
    "native_vs_torch32/output",
    "native_vs_torch32/probabilities",
    "native_vs_torch32/dqkv",
    "torch32_vs_fp64/output",
    "torch32_vs_fp64/probabilities",
    "torch32_vs_fp64/dqkv",
)


def expected_precision_check_names(width):
    names = []
    for name, *_ in precision_case_specs(width):
        prefix = f"v6/precision/{name}/"
        names.extend(prefix + suffix for suffix in PRECISION_CHECK_SUFFIXES)
        names.extend(prefix + suffix + "/finite_shape" for suffix in PRECISION_DIAGNOSTIC_SUFFIXES)
    return names


def expected_precision_diagnostic_names(width):
    return [
        f"v6/precision/{name}/{suffix}"
        for name, *_ in precision_case_specs(width)
        for suffix in PRECISION_DIAGNOSTIC_SUFFIXES
    ]
