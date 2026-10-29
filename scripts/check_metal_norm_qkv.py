"""Development-only Metal LN/QKV probes; never run concurrently with GPU timing.

Example from repository root, after acquiring the shared GPU execution slot:
ARM64_TRANSFORMER_WIDTH=32 PYTHONPATH=. .venv/bin/python \
  work/gpu-v1/precision/metal_ln_qkv_development.py --width 32 --output FRESH.json

Host readbacks bridge the actual LN/linear primitives for diagnosis only. This is
not an end-to-end production timing or a formal admission record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import numpy as np

from transformer import spec
from transformer.gpu_precision import (
    GPU_PRECISION_CONTRACT,
    GPU_PRECISION_SEEDS,
    layernorm_qkv_case_specs,
    layernorm_qkv_reference,
    layernorm_qkv_stress_cases,
)
from transformer.precision_audit import comparison

FIELDS = ("x", "gamma", "beta", "weight", "bias", "dqkv")
SOURCE_FILES = (
    "transformer/gpu_precision.py",
    "transformer/precision.py",
    "transformer/precision_audit.py",
    "transformer/gpu_metal.py",
    "transformer/gpu_contract.py",
    "transformer/spec.py",
    "transformer/validate.py",
    "native/metal/backend.mm",
    "native/metal/kernels.metal",
    "native/metal/Makefile",
    "native/metal/build/libtransformer_metal.dylib",
)


def array_identity(array):
    array = np.asarray(array)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": array.nbytes,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def source_identity(root):
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in SOURCE_FILES
    }


def capture(backend, result):
    # All borrowed scratch views are copied before any next primitive call.
    backend.synchronize()
    return {key: backend.to_numpy(value) for key, value in result.items()}


def evaluate(backend, case):
    norm = capture(backend, backend.layernorm(case["x"], case["gamma"], case["beta"]))
    linear = capture(
        backend, backend.linear(norm["output"], case["weight"], case["bias"], case["dqkv"])
    )
    backward = capture(
        backend, backend.layernorm(case["x"], case["gamma"], case["beta"], linear["dx"])
    )
    return {"norm": norm, "linear": linear, "backward": backward}


def check_case(backend, case):
    before = {key: array_identity(case[key]) for key in FIELDS}
    oracle = layernorm_qkv_reference(*(case[key] for key in FIELDS), epsilon=case["epsilon"])
    actual = evaluate(backend, case)
    repeat = evaluate(backend, case)
    after = {key: array_identity(case[key]) for key in FIELDS}
    prefix = case["name"] + "/"
    records = []

    def compare(name, value, expected, tolerance="forward"):
        records.append(comparison(prefix + name, value, expected, tolerance=tolerance))

    compare("joint/normalized_fp64", actual["norm"]["norm"], oracle["normalized"])
    compare(
        "joint/inverse_std_fp64", actual["norm"]["inv"], oracle["inverse_std"].reshape(-1), "math"
    )
    compare("joint/ln_fp64", actual["norm"]["output"], oracle["ln"])
    compare("joint/qkv_fp64", actual["linear"]["output"], oracle["qkv"])
    for key in ("dx", "dgamma", "dbeta"):
        compare("joint/" + key + "_fp64", actual["backward"][key], oracle[key], "gradient")
    for key in ("dweight", "dbias"):
        compare("joint/" + key + "_fp64", actual["linear"][key], oracle[key], "gradient")
    compare("joint/dln_fp64", actual["linear"]["dx"], oracle["dln"], "gradient")

    # Same-stored-operand stage checks localize propagation versus primitive
    # arithmetic. They do not replace the full mathematical joint checks above.
    rounded_ln = actual["norm"]["output"].astype(np.float64)
    weight, bias, dy = (case[key].astype(np.float64) for key in ("weight", "bias", "dqkv"))
    compare(
        "linear_same_operands/output_fp64", actual["linear"]["output"], rounded_ln @ weight.T + bias
    )
    compare("linear_same_operands/dln_fp64", actual["linear"]["dx"], dy @ weight, "gradient")
    compare(
        "linear_same_operands/dweight_fp64",
        actual["linear"]["dweight"],
        dy.T @ rounded_ln,
        "gradient",
    )
    compare(
        "linear_same_operands/dbias_fp64", actual["linear"]["dbias"], dy.sum(axis=0), "gradient"
    )
    incoming = actual["linear"]["dx"].astype(np.float64)
    dnormalized = incoming * case["gamma"].astype(np.float64)
    dcentered = oracle["inverse_std"] * (
        dnormalized
        - oracle["normalized"] * (dnormalized * oracle["normalized"]).mean(-1, keepdims=True)
    )
    compare(
        "norm_same_incoming/dx_fp64",
        actual["backward"]["dx"],
        dcentered - dcentered.mean(-1, keepdims=True),
        "gradient",
    )
    compare(
        "norm_same_incoming/dgamma_fp64",
        actual["backward"]["dgamma"],
        (incoming * oracle["normalized"]).sum(axis=0),
        "gradient",
    )
    compare(
        "norm_same_incoming/dbeta_fp64",
        actual["backward"]["dbeta"],
        incoming.sum(axis=0),
        "gradient",
    )

    exact = {
        f"{stage}/{key}": value.tobytes() == repeat[stage][key].tobytes()
        for stage, outputs in actual.items()
        for key, value in outputs.items()
    }
    guards = backend.check_guards()
    finite = all(np.isfinite(value).all() for stage in actual.values() for value in stage.values())
    return {
        "name": case["name"],
        "width": case["width"],
        "rows": case["rows"],
        "kind": case["kind"],
        "attempted": True,
        "fixture": before,
        "inputs_unchanged": before == after,
        "epsilon": case["epsilon"],
        "repeat_exact": all(exact.values()),
        "repeat_checks": exact,
        "guards_passed": bool(guards),
        "outputs_finite": bool(finite),
        "output_identity": {
            stage: {key: array_identity(value) for key, value in outputs.items()}
            for stage, outputs in actual.items()
        },
        "checks": records,
        "failed_checks": [record["name"] for record in records if not record["passed"]],
        "passed": (
            all(record["passed"] for record in records)
            and all(exact.values())
            and before == after
            and bool(guards)
            and bool(finite)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.width != spec.WIDTH:
        raise ValueError("Set ARM64_TRANSFORMER_WIDTH before starting this process")
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists() or args.output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a fresh path before initializing any GPU object.
    with partial.open("x") as handle:
        handle.write("{}\n")
    result = {
        "contract": GPU_PRECISION_CONTRACT,
        "scope": (
            "Development only; CPU readbacks bridge actual Metal primitives. "
            "Not a timed production path or admission."
        ),
        "width": spec.WIDTH,
        "stream": "development",
        "seed": GPU_PRECISION_SEEDS["development"],
        "planned_case_count": len(layernorm_qkv_case_specs(spec.WIDTH)),
        "cases": [],
        "numpy_version": np.__version__,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    backend = None
    stopped = False
    try:
        root = Path.cwd()
        result["source_sha256_before"] = source_identity(root)
        from transformer.gpu_metal import Backend

        backend = Backend(np.zeros(spec.PARAM_COUNT, dtype=np.float32))
        result["capabilities"] = backend.capabilities()
        for case in layernorm_qkv_stress_cases(spec.WIDTH, split="development"):
            if stopped:
                record = {
                    "name": case["name"],
                    "attempted": False,
                    "passed": False,
                    "reason": "Prior runtime exception; no unsafe continuation",
                }
            else:
                try:
                    record = check_case(backend, case)
                except Exception as error:
                    stopped = True
                    record = {
                        "name": case["name"],
                        "attempted": True,
                        "passed": False,
                        "exception": repr(error),
                        "traceback": traceback.format_exc(),
                        "fixture": {key: array_identity(case[key]) for key in FIELDS},
                    }
            result["cases"].append(record)
            partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "case": record["name"],
                        "passed": record["passed"],
                        "failed_checks": record.get("failed_checks", []),
                        "exception": record.get("exception"),
                    }
                ),
                flush=True,
            )
        result["source_sha256_after"] = source_identity(root)
        result["source_unchanged"] = result["source_sha256_before"] == result["source_sha256_after"]
    except Exception as error:
        result["fatal_exception"] = repr(error)
        result["fatal_traceback"] = traceback.format_exc()
        existing = {case["name"] for case in result["cases"]}
        result["cases"].extend(
            {
                "name": name,
                "attempted": False,
                "passed": False,
                "reason": "Initialization or driver exception",
            }
            for name, *_ in layernorm_qkv_case_specs(spec.WIDTH)
            if name not in existing
        )
    finally:
        if backend is not None:
            backend.close()
    result["case_count"] = len(result["cases"])
    result["passed"] = (
        result.get("source_unchanged") is True
        and result["case_count"] == result["planned_case_count"]
        and all(case["passed"] for case in result["cases"])
    )
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.link(partial, output)
    partial.unlink()
    print(
        json.dumps(
            {"output": str(output), "passed": result["passed"], "case_count": result["case_count"]}
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
