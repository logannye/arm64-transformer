"""Recompute development decisions from authenticated raw replay arrays.

No production comparison helper, framework import, admission stream or training.
The caller supplies receipt and inventory hashes from a separately frozen list.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(root, relative, expected):
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), "Evidence escaped its root")
    require(sha256(path) == expected, "Evidence bytes changed: " + relative)
    return path


def parameter_sizes(width):
    # Independent declaration of the fixed model layout, including hidden=2W.
    return dict(
        token=128 * width,
        position=32 * width,
        ln1_weight=width,
        ln1_bias=width,
        qkv_weight=3 * width * width,
        qkv_bias=3 * width,
        attention_weight=width * width,
        attention_bias=width,
        ln2_weight=width,
        ln2_bias=width,
        ff1_weight=2 * width * width,
        ff1_bias=2 * width,
        ff2_weight=2 * width * width,
        ff2_bias=width,
        ln3_weight=width,
        ln3_bias=width,
        head_weight=128 * width,
        head_bias=128,
    )


def compare(actual, expected, atol, rtol):
    require(actual.shape == expected.shape, "Raw array shape mismatch")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite raw array")
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    limit = atol + rtol * np.abs(expected.astype(np.float64))
    count = int(np.count_nonzero(error > limit))
    return dict(
        passed=count == 0,
        failed_coordinates=count,
        max_absolute_error=float(error.max()),
        max_tolerance_ratio=float((error / limit).max()),
    )


def audit_worker(output, receipt_sha, plan_path, plan_sha, evidence_root):
    output, evidence_root = Path(output), Path(evidence_root)
    result = json.loads(checked(output, "result.json", receipt_sha).read_text())
    require(sha256(plan_path) == plan_sha, "Inventory identity changed")
    plan = json.loads(Path(plan_path).read_text())
    require(
        result["schema"] == plan["schema"] == "trained-precision-development-v1"
        and result["inventory_sha256"] == plan_sha
        and result["formal_eligible"] is False
        and result["admission_stream_opened"] is False,
        "Wrong development identity or scope",
    )
    require(result["complete"] is True, "Worker did not complete")
    require(not result.get("error") and not result.get("postflight_error"), "Worker error")
    width, batch = result["width"], result["batch_size"]
    require(width in (512, 1024) and batch in (1, 16), "Invalid model dimensions")
    inventory = {case["id"]: case for case in plan["cases"] if case["width"] == width}
    expected = result["expected_cases"]
    require(bool(expected) and len(set(expected)) == len(expected), "Empty or duplicate cases")
    require([c["case"]["id"] for c in result["cases"]] == expected, "Case coverage changed")
    if result["full_width_inventory"]:
        require(expected == list(inventory), "Full inventory claim omits or reorders cases")
    checked_inputs, audited = set(), []
    sizes = parameter_sizes(width)
    for row in result["cases"]:
        case = row["case"]
        require(inventory.get(case["id"]) == case, "Case differs from inventory")
        for key in ("checkpoint", "rows"):
            identity = (case[key], case[key + "_sha256"])
            if identity not in checked_inputs:
                checked(evidence_root, *identity)
                checked_inputs.add(identity)
        checkpoint = evidence_root / case["checkpoint"]
        require(checkpoint.stat().st_size == sum(sizes.values()) * 4, "Checkpoint size differs")
        rows = np.fromfile(evidence_root / case["rows"], dtype="i4").reshape(-1, 33)
        first = case["sequence"] // batch * batch
        require(row["first_sequence"] == first, "Batch grouping changed")
        raw = checked(output, row["raw"], row["raw_sha256"])
        quantities = ["logits", "loss"] + (["gradient"] if result["gradients_checked"] else [])
        required = {"tokens", "targets"} | {
            f"{label}_{name}"
            for label in ("actual", "repeat", "numpy64", "torch64")
            for name in quantities
        }
        checks = {}
        with np.load(raw, allow_pickle=False) as arrays:
            require(set(arrays.files) == required, "Raw quantity inventory changed")
            for name, values in (
                ("tokens", rows[first : first + batch, :-1]),
                ("targets", rows[first : first + batch, 1:]),
            ):
                require(
                    arrays[name].dtype == np.dtype("i4") and np.array_equal(arrays[name], values),
                    "Raw inputs differ from original batch",
                )
            for name in quantities:
                actual, repeat, oracle, double = (
                    arrays[label + "_" + name]
                    for label in ("actual", "repeat", "numpy64", "torch64")
                )
                shape = (
                    (batch, 32, 128)
                    if name == "logits"
                    else ((sum(sizes.values()),) if name == "gradient" else ())
                )
                require(actual.shape == shape, "Wrong quantity dimensions")
                if name != "loss":
                    require(actual.dtype == np.dtype("f4"), "Expected FP32 stored result")
                require(oracle.dtype == double.dtype == np.dtype("f8"), "Expected FP64 oracle")
                tolerance = (3e-5, 3e-3) if name == "gradient" else (2e-5, 2e-4)
                checks[name] = compare(actual, oracle, *tolerance)
                checks[name + "_oracle"] = compare(double, oracle, 1e-10, 1e-10)
                checks[name + "_repeat"] = {
                    "passed": actual.dtype == repeat.dtype
                    and actual.shape == repeat.shape
                    and actual.tobytes() == repeat.tobytes()
                }
                if name == "gradient":
                    offset = 0
                    for parameter, size in sizes.items():
                        section = slice(offset, offset + size)
                        checks["gradient/" + parameter] = compare(
                            actual[section], oracle[section], *tolerance
                        )
                        offset += size
        require(set(row["checks"]) == set(checks), "Recorded check inventory changed")
        for name, check in checks.items():
            recorded = row["checks"][name]
            require(recorded["passed"] == check["passed"], "Recorded decision disagrees with raw")
            for metric in ("failed_coordinates", "max_absolute_error", "max_tolerance_ratio"):
                if metric in recorded:
                    require(recorded[metric] == check[metric], "Recorded numeric metric differs")
        passed = all(check["passed"] for check in checks.values())
        require(row["passed"] == passed, "Case decision disagrees with raw")
        audited.append({"id": case["id"], "passed": passed, "checks": checks})
    passed = all(case["passed"] for case in audited)
    require(result["passed"] == passed, "Worker decision disagrees with raw")
    return {
        "schema": "trained-precision-independent-raw-audit-v1",
        "receipt_sha256": receipt_sha,
        "inventory_sha256": plan_sha,
        "evidence_valid": True,
        "numerics_passed": passed,
        "formal_eligible": False,
        "engine": result["engine"],
        "width": width,
        "batch_size": batch,
        "gradients_checked": result["gradients_checked"],
        "full_width_inventory": result["full_width_inventory"],
        "native_policy": result.get("native", {}).get("precision_policy"),
        # Early retained workers predate the factorial field. Preserve missing
        # metadata explicitly; raw numeric evidence does not infer a policy.
        "torch_diagnostic_policy": result.get("torch_diagnostic_policy"),
        "policy_metadata_present": "torch_diagnostic_policy" in result,
        "python_candidate": result.get("python_candidate"),
        "metal_candidate_shader_sha256": result.get("metal_candidate_shader_sha256"),
        "cases": audited,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("worker", "receipt-sha256", "inventory", "inventory-sha256", "evidence-root"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_worker(
                args.worker,
                args.receipt_sha256,
                args.inventory,
                args.inventory_sha256,
                args.evidence_root,
            ),
            indent=2,
        )
    )
