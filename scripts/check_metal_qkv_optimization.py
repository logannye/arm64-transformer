"""Development-only qualification of ordered, protected QKV contractions.

This module imports no GPU runtime. Both policies must retain the same dense
weight-gradient and bias-reduction configuration. The new kernels must match
the retained protected path byte for byte, in addition to the existing FP64
accuracy gates. Admission stream 20261104 is never generated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from pathlib import Path

import numpy as np

from transformer.gpu_precision import GPU_PRECISION_SEEDS, GPU_WIDTHS
from transformer.precision_audit import ORACLE_TOLERANCES, comparison

CONTRACT = "gpu-v3-qkv-ordered-development-v1"
SEED = GPU_PRECISION_SEEDS["development"]
FAMILY = 53
ROWS = (1, 3, 9, 16, 17, 31, 32, 33, 48, 96, 511, 512)
PATTERNS = ("forward_cancellation", "dx_cancellation", "common_bias", "mixed_exponents")
INPUTS = ("x", "weight", "bias", "dy")
OUTPUTS = ("output", "dx", "dweight", "dbias")


def array_identity(value):
    value = np.asarray(value)
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "bytes": value.nbytes,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def qkv_case_specs(width):
    if width not in GPU_WIDTHS:
        raise ValueError("Unsupported QKV width")
    cases = [(f"random_rows{rows}", rows, "random") for rows in ROWS]
    cases.extend((name, 3, name) for name in PATTERNS)
    return [
        {
            "name": name,
            "width": width,
            "rows": rows,
            "kind": kind,
            "x_shape": [rows, width],
            "weight_shape": [3 * width, width],
            "bias_shape": [3 * width],
            "dy_shape": [rows, 3 * width],
        }
        for name, rows, kind in cases
    ]


def qkv_stress_cases(width):
    for index, specification in enumerate(qkv_case_specs(width)):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, width, FAMILY, index]))
        rows, kind = specification["rows"], specification["kind"]
        x = rng.normal(0, 0.25, (rows, width)).astype("f4")
        weight = rng.normal(0, 1 / math.sqrt(width), (3 * width, width)).astype("f4")
        bias = rng.normal(0, 0.125, 3 * width).astype("f4")
        dy = rng.normal(0, 0.25, (rows, 3 * width)).astype("f4")
        if kind == "forward_cancellation":
            x[:, 0::2] = rng.normal(64, 0.125, (rows, width // 2)).astype("f4")
            x[:, 1::2] = x[:, 0::2]
            positive = rng.normal(0, 8 / math.sqrt(width), (3 * width, width // 2))
            weight[:, 0::2] = positive.astype("f4")
            weight[:, 1::2] = -weight[:, 0::2] + np.float32(2**-18)
        elif kind == "dx_cancellation":
            dy[:, 0::2] = rng.normal(64, 0.125, (rows, 3 * width // 2)).astype("f4")
            dy[:, 1::2] = dy[:, 0::2]
            positive = rng.normal(0, 8 / math.sqrt(width), (3 * width // 2, width))
            weight[0::2] = positive.astype("f4")
            weight[1::2] = -weight[0::2] + np.float32(2**-18)
        elif kind == "common_bias":
            bias[: 2 * width] = np.float32(4096)
            weight *= np.float32(2**-8)
        elif kind == "mixed_exponents":
            exponents = np.resize(np.array([-10, -5, 0, 5, 10]), width)
            x = (rng.uniform(0.75, 1.25, x.shape) * np.exp2(exponents)).astype("f4")
            weight = (
                rng.uniform(0.75, 1.25, weight.shape)
                * np.exp2(-exponents)
                * np.resize(np.array([-1.0, 1.0]), width)
            ).astype("f4")
        yield dict(specification, x=x, weight=weight, bias=bias, dy=dy)


def qkv_reference(case):
    """Independent FP64 equations with prespecified scalar fsum crosschecks."""
    x, weight, bias, dy = (np.asarray(case[name], dtype=np.float64) for name in INPUTS)
    values = {
        "output": x @ weight.T + bias,
        "dx": dy @ weight,
        "dweight": dy.T @ x,
        "dbias": dy.sum(axis=0),
    }
    m, k = x.shape
    n = weight.shape[0]
    positions = {
        "output": [(0, 0), (m // 2, n // 2), (m - 1, n - 1)],
        "dx": [(0, 0), (m // 2, k // 2), (m - 1, k - 1)],
        "dweight": [(0, 0), (n // 2, k // 2), (n - 1, k - 1)],
        "dbias": [(0,), (n // 2,), (n - 1,)],
    }
    checks = []
    for name, indices in positions.items():
        for coordinate in sorted(set(indices)):
            if name == "output":
                row, channel = coordinate
                expected = math.fsum(
                    float(a) * float(b) for a, b in zip(x[row], weight[channel], strict=True)
                ) + float(bias[channel])
            elif name == "dx":
                row, channel = coordinate
                expected = math.fsum(
                    float(a) * float(b) for a, b in zip(dy[row], weight[:, channel], strict=True)
                )
            elif name == "dweight":
                output, channel = coordinate
                expected = math.fsum(
                    float(a) * float(b) for a, b in zip(dy[:, output], x[:, channel], strict=True)
                )
            else:
                expected = math.fsum(float(value) for value in dy[:, coordinate[0]])
            observed = float(values[name][coordinate])
            error = abs(observed - expected)
            limit = ORACLE_TOLERANCES["atol"] + ORACLE_TOLERANCES["rtol"] * abs(expected)
            checks.append(
                {
                    "name": name,
                    "coordinate": list(coordinate),
                    "actual": observed,
                    "expected": expected,
                    "absolute_error": error,
                    "allowed": limit,
                    "passed": math.isfinite(observed) and error <= limit,
                }
            )
    return {
        "values": values,
        "crosschecks": checks,
        "oracle_tolerances": dict(ORACLE_TOLERANCES),
        "passed": all(check["passed"] for check in checks),
    }


def _capture(backend, case):
    views = backend.linear(*(case[name] for name in INPUTS))
    backend.synchronize()
    return {name: np.array(backend.to_numpy(view), copy=True) for name, view in views.items()}


def _exact(left, right):
    return left.keys() == right.keys() and all(
        left[name].dtype == right[name].dtype
        and left[name].shape == right[name].shape
        and left[name].tobytes(order="C") == right[name].tobytes(order="C")
        for name in left
    )


def check_qkv_case(backend, reference_backend, case, *, oracle=None):
    before = {name: array_identity(case[name]) for name in INPUTS}
    oracle = qkv_reference(case) if oracle is None else oracle
    reference = _capture(reference_backend, case)
    reference_repeat = _capture(reference_backend, case)
    reference_unchanged = before == {name: array_identity(case[name]) for name in INPUTS}
    actual = _capture(backend, case)
    repeat = _capture(backend, case)
    after = {name: array_identity(case[name]) for name in INPUTS}
    checks = []
    controls = {
        "oracle_keys": set(oracle["values"]) == set(OUTPUTS),
        "oracle_crosscheck": (
            oracle["passed"]
            and len(oracle["crosschecks"]) == 12
            and all(check["passed"] for check in oracle["crosschecks"])
        ),
        "reference_input_immutability": reference_unchanged,
        "candidate_input_immutability": before == after,
        "reference_keys": set(reference) == set(OUTPUTS),
        "candidate_keys": set(actual) == set(OUTPUTS),
        "reference_repeat_exact": _exact(reference, reference_repeat),
        "candidate_repeat_exact": _exact(actual, repeat),
        "reference_guards": bool(reference_backend.check_guards()),
        "candidate_guards": bool(backend.check_guards()),
    }
    for name, expected in oracle["values"].items():
        for label, values in (("reference", reference), ("candidate", actual)):
            value = values.get(name, np.empty(0, np.float32))
            controls[f"{label}/{name}/dtype"] = value.dtype == np.dtype("float32")
            checks.append(
                comparison(
                    f"{case['name']}/{label}/{name}/fp64",
                    value,
                    expected,
                    tolerance="forward" if name == "output" else "gradient",
                )
            )
        controls[f"{name}/reference_exact"] = (
            name in actual
            and name in reference
            and _exact({name: actual[name]}, {name: reference[name]})
        )
    return {
        "name": case["name"],
        "rows": case["rows"],
        "width": case["width"],
        "kind": case["kind"],
        "operands": before,
        "oracle_crosschecks": oracle["crosschecks"],
        "checks": checks,
        "controls": controls,
        "output_identities": {
            label: {name: array_identity(value) for name, value in values.items()}
            for label, values in (("reference", reference), ("candidate", actual))
        },
        "passed": all(check["passed"] for check in checks) and all(controls.values()),
    }


def qualify_qkv(backend, reference_backend, width, *, cached_cases=None):
    cases = (
        ((case, None) for case in qkv_stress_cases(width)) if cached_cases is None else cached_cases
    )
    results = [
        check_qkv_case(backend, reference_backend, case, oracle=oracle) for case, oracle in cases
    ]
    expected = [case["name"] for case in qkv_case_specs(width)]
    complete = [record["name"] for record in results] == expected
    return {
        "contract": CONTRACT,
        "scope": "development_only_no_admission_operands_or_formal_measurement",
        "seed": SEED,
        "family": FAMILY,
        "width": width,
        "inventory": qkv_case_specs(width),
        "cases": results,
        "inventory_complete": complete,
        "passed": complete and all(record["passed"] for record in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, required=True, choices=GPU_WIDTHS)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--reference-policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(qkv_case_specs(args.width), indent=2))
        return 0
    if args.policy is None or args.reference_policy is None or args.output is None:
        parser.error("--policy, --reference-policy and --output are required for qualification")
    from transformer import spec
    from transformer.gpu_metal import Backend
    from transformer.gpu_optimize import source_identity
    from transformer.gpu_policy import bundle

    if spec.WIDTH != args.width:
        raise ValueError("Set ARM64_TRANSFORMER_WIDTH before starting this process")
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists() or args.output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x") as handle:
        handle.write("{}\n")
    record = {
        "contract": CONTRACT,
        "seed": SEED,
        "family": FAMILY,
        "width": args.width,
        "inventory": qkv_case_specs(args.width),
        "cases": [],
        "passed": False,
        "source_sha256_before": source_identity(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    candidate = reference_backend = None
    try:
        policy = bundle(json.loads(args.policy.read_text()), args.width)
        baseline = bundle(json.loads(args.reference_policy.read_text()), args.width)
        initial = np.zeros(spec.PARAM_COUNT, np.float32)
        candidate = Backend(initial, kernel_policy=policy)
        reference_backend = Backend(initial, kernel_policy=baseline)
        record["policies"] = {"candidate": policy, "reference": baseline}
        record["capabilities"] = {
            "candidate": candidate.capabilities(),
            "reference": reference_backend.capabilities(),
        }
        for case in qkv_stress_cases(args.width):
            item = check_qkv_case(candidate, reference_backend, case)
            record["cases"].append(item)
            partial.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"name": item["name"], "passed": item["passed"]}), flush=True)
        record["source_sha256_after"] = source_identity()
        record["source_unchanged"] = record["source_sha256_before"] == record["source_sha256_after"]
        record["passed"] = record["source_unchanged"] and all(
            item["passed"] for item in record["cases"]
        )
    except Exception as error:
        record.update(error=repr(error), traceback=traceback.format_exc(), passed=False)
    finally:
        for backend in (candidate, reference_backend):
            if backend is not None:
                try:
                    backend.close()
                except Exception as error:
                    record.setdefault("cleanup_errors", []).append(repr(error))
                    record["passed"] = False
        partial.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.link(partial, output)
        partial.unlink()
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
