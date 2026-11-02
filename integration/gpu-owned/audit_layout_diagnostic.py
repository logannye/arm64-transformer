"""Recompute closed layout diagnostics from saved arrays, without importing a GPU framework."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SUMMARY_SHA256 = "08f98e4ea6adf7b1257d6dcf598205fd0414333d5f286c3177f64b6f81ad6f48"
PLAN_SHA256 = "c515ea6bee9115b32ee65f3f78b9aec76e1251af547d0fc7380c74490ce003fc"
MODES = ("ordinary", "metal32_diagnostic")
DTYPES = ("float32", "float16", "bfloat16")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_file(path, sha256, size=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing regular evidence: {path}")
    if size is not None:
        require(path.stat().st_size == size, f"Changed evidence size: {path}")
    require(digest(path) == sha256, f"Changed evidence hash: {path}")
    return {"sha256": sha256, "bytes": path.stat().st_size}


def inventory():
    result = []
    for dtype in DTYPES:
        for operation in ("mm", "addmm", "bmm"):
            shapes = (
                ((2, 2, 2, 2), (4, 8, 64, 12), (3, 33, 17, 49))
                if operation == "bmm"
                else ((1, 64, 12), (12, 64, 1), (8, 64, 12), (33, 17, 49))
            )
            for shape in shapes:
                for layout in ("inner", "transposed", "outer"):
                    if operation != "bmm" and min(shape[0], shape[-1]) == 1:
                        if layout != "inner":
                            continue
                    result.append(
                        dict(dtype=dtype, operation=operation, shape=list(shape), layout=layout)
                    )
    result.append(dict(dtype="float32", operation="mm", shape=[1, 512, 64], layout="inner"))
    return result


def comparison(actual, expected, atol, rtol):
    require(actual.shape == expected.shape, "Comparison shape differs")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(e)
    with np.errstate(invalid="ignore", over="ignore"):
        error = np.abs(a - e)
        bad = ~finite | (error > atol + rtol * np.abs(e))
    return {
        "passed": not bool(bad.any()),
        "failed_coordinates": int(bad.sum()),
        "coordinates": int(a.size),
        "all_finite": bool(finite.all()),
        "max_absolute_error": float(error[finite].max()) if finite.any() else None,
        "atol": atol,
        "rtol": rtol,
    }


def cstrides(shape):
    stride, result = 1, []
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return list(reversed(result))


def layout(shape, strides, dtype):
    return {
        "shape": list(shape),
        "strides": strides,
        "storage_offset": 0,
        "dtype": f"torch.{dtype}",
    }


def matrix_layouts(case):
    prefix = case["shape"][:-3]
    m, k, n = case["shape"][-3:]
    shape = [*prefix, m, n]
    strides = cstrides(shape)
    if case["layout"] == "inner":
        strides = [value * 2 for value in strides]
    elif case["layout"] == "outer":
        strides[0] *= 2
    else:
        strides[-2:] = [1, m]
    return {
        "a": layout([*prefix, m, k], cstrides([*prefix, m, k]), case["dtype"]),
        "b": layout([*prefix, k, n], cstrides([*prefix, k, n]), case["dtype"]),
        "bias": layout(shape, cstrides(shape), case["dtype"]),
        "ordinary": layout(shape, strides, case["dtype"]),
        "control": layout(shape, strides, case["dtype"]),
    }


def array_identity(value):
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def validate_dtype(array, declared, fp64=False):
    expected = np.dtype("float64" if fp64 else "float32" if declared == "bfloat16" else declared)
    require(array.dtype == expected, "Saved dtype differs from declared storage conversion")
    if declared == "bfloat16" and not fp64:
        bits = np.ascontiguousarray(array).view(np.uint32)
        require(
            bool(((bits & 0xFFFF) == 0).all()), "FP32 values are not exact bfloat16 conversions"
        )


def load_arrays(folder, record, expected_keys, dtype, shapes, fp64=()):
    raw = record["raw"]
    require(Path(raw["path"]).name == raw["path"], "Unexpected raw-array path")
    path = folder / raw["path"]
    binding = frozen_file(path, raw["sha256"], raw["bytes"])
    with np.load(path, allow_pickle=False) as saved:
        require(
            len(saved.files) == len(expected_keys) and set(saved.files) == set(expected_keys),
            "Missing, duplicate or extra raw-array member",
        )
        arrays = {key: saved[key] for key in saved.files}
    for key, value in arrays.items():
        validate_dtype(value, dtype, key in fp64)
        require(list(value.shape) == list(shapes[key]), f"Wrong shape: {key}")
    return arrays, binding


def audit(root):
    root = Path(root)
    bindings = {
        "summary.json": frozen_file(root / "summary.json", SUMMARY_SHA256),
        "plan.json": frozen_file(root / "plan.json", PLAN_SHA256),
    }
    summary, plan = read(root / "summary.json"), read(root / "plan.json")
    require(summary["plan_sha256"] == PLAN_SHA256 and plan["modes"] == list(MODES), "Plan mismatch")
    require(
        plan["processes_per_mode"] == 2 and plan["development_seed"] == 20261103,
        "Plan budget changed",
    )
    expected_workers = [(mode, index) for mode in MODES for index in (0, 1)]
    require(
        [(x["mode"], x["process_index"]) for x in summary["workers"]] == expected_workers,
        "Missing, duplicate or reordered worker",
    )
    common, output_hashes, rows, worker_counts = {}, {}, [], []
    expected_cases = inventory()
    require(len(expected_cases) == 76, "Auditor inventory error")
    for row in summary["workers"]:
        name = f"{row['mode']}-p{row['process_index']}"
        folder = root / name
        bindings[f"{name}/result.json"] = frozen_file(folder / "result.json", row["sha256"])
        worker = read(folder / "result.json")
        require(
            row["returncode"] == 0 and worker["complete"] and worker["execution_integrity_passed"],
            "Worker execution incomplete",
        )
        require(
            worker["mode"] == row["mode"]
            and worker["sources"] == plan["sources"]
            and worker["plan_sha256"] == PLAN_SHA256,
            "Worker identity mismatch",
        )
        require(
            worker["metal_language_override"] == (None if row["mode"] == "ordinary" else "3.2"),
            "Wrong diagnostic execution mode",
        )
        require(
            worker["postflight"].get("passed") is True
            and worker["postflight"].get("all_declared_input_bytes_and_file_identities_unchanged")
            is True,
            "Worker postflight failed",
        )
        require(
            worker["formal_eligible"] is False
            and worker["formal_fits"] == 0
            and worker["admission_stream_opened"] is False,
            "Scope changed",
        )
        data = worker["diagnostic"]
        declared = read(folder / "cases/inventory.json")
        require(
            declared["matrix"] == expected_cases
            and declared["convolution_dtypes"] == list(DTYPES)
            and declared["convolution_bias"] == [False, True],
            "Case inventory changed",
        )
        require(
            [x["case"] for x in data["matrix_cases"]] == expected_cases,
            "Matrix cases missing/reordered",
        )
        require(
            [(x["dtype"], x["bias"]) for x in data["convolution_cases"]]
            == [(d, b) for d in DTYPES for b in (False, True)],
            "Convolution cases missing/reordered",
        )
        counts = dict(
            mode=row["mode"],
            process_index=row["process_index"],
            matrix_cases=76,
            ordinary_matrix_failures=0,
            control_matrix_failures=0,
            guard_failures=0,
            convolution_layout_failures=0,
            contiguous_convolution_fp64_failures=0,
            channels_last_convolution_fp64_failures=0,
        )
        for index, record in enumerate(data["matrix_cases"]):
            key, case = f"matrix-{index:03}", expected_cases[index]
            require(read(folder / f"cases/{key}.result.json") == record, "Matrix sidecar changed")
            require(
                record["complete"] and record["layouts"] == matrix_layouts(case),
                "Wrong matrix layout",
            )
            shapes = {k: v["shape"] for k, v in record["layouts"].items()}
            shape = shapes["ordinary"]
            shapes.update(reference=shape, ordinary_guard=shape, control_guard=shape)
            if case["layout"] == "transposed":
                shapes["ordinary_guard"] = shapes["control_guard"] = [0]
            arrays, binding = load_arrays(folder / "cases", record, shapes, case["dtype"], shapes)
            bindings[f"{name}/cases/{key}.npz"] = binding
            tolerance = 1e-4 if case["dtype"] == "float32" else 1e-2
            metrics = {
                "ordinary": comparison(
                    arrays["ordinary"], arrays["reference"], tolerance, tolerance
                ),
                "contiguous_then_copy": comparison(
                    arrays["control"], arrays["reference"], tolerance, tolerance
                ),
            }
            for metric, result in metrics.items():
                require(
                    result == record[metric], f"Recorded comparison differs: {name}/{key}/{metric}"
                )
            guards = {
                kind: bool((arrays[kind + "_guard"] == 0.125).all())
                for kind in ("ordinary", "control")
            }
            require(
                all(guards[k] == record[k + "_guard_intact"] for k in guards),
                "Guard summary differs",
            )
            counts["ordinary_matrix_failures"] += not metrics["ordinary"]["passed"]
            counts["control_matrix_failures"] += not metrics["contiguous_then_copy"]["passed"]
            counts["guard_failures"] += not all(guards.values())
            inputs = {k: array_identity(arrays[k]) for k in ("a", "b", "bias", "reference")}
            if key in common:
                require(inputs == common[key], f"Common matrix operands/reference differ: {key}")
            common[key] = inputs
            output_hashes.setdefault(key, {})[name] = {
                k: array_identity(arrays[k])
                for k in ("ordinary", "control", "ordinary_guard", "control_guard")
            }
            rows.append(
                dict(
                    worker=name,
                    family="matrix",
                    case=key,
                    dtype=case["dtype"],
                    operation=case["operation"],
                    layout=case["layout"],
                    ordinary_failed_coordinates=metrics["ordinary"]["failed_coordinates"],
                    control_failed_coordinates=metrics["contiguous_then_copy"][
                        "failed_coordinates"
                    ],
                    ordinary_guard_intact=guards["ordinary"],
                    control_guard_intact=guards["control"],
                    reference="same_dtype_pytorch_cpu",
                )
            )
        for record in data["convolution_cases"]:
            dtype, bias = record["dtype"], record["bias"]
            key = f"conv-{dtype}-{int(bias)}"
            require(
                read(folder / f"cases/{key}.result.json") == record, "Convolution sidecar changed"
            )
            expected_layouts = {
                "input": layout([2, 4, 8, 8, 8], [2048, 1, 256, 32, 4], dtype),
                "ordinary": layout([2, 8, 8, 8, 8], [4096, 1, 512, 64, 8], dtype),
                "control": layout([2, 8, 8, 8, 8], [4096, 512, 64, 8, 1], dtype),
            }
            require(
                record["complete"]
                and record["channels_last_output"]
                and record["layouts"] == expected_layouts,
                "Wrong convolution layouts",
            )
            shapes = dict(
                input=[2, 4, 8, 8, 8],
                weight=[8, 4, 3, 3, 3],
                ordinary=[2, 8, 8, 8, 8],
                control=[2, 8, 8, 8, 8],
                reference=[2, 8, 8, 8, 8],
            )
            if bias:
                shapes["bias"] = [8]
            arrays, binding = load_arrays(
                folder / "cases", record, shapes, dtype, shapes, ("reference",)
            )
            bindings[f"{name}/cases/{key}.npz"] = binding
            atol, rtol = {
                "float32": (1e-5, 1e-5),
                "float16": (1e-3, 1e-3),
                "bfloat16": (1e-3, 1.6e-2),
            }[dtype]
            metrics = {
                "channels_last_vs_contiguous": comparison(
                    arrays["ordinary"], arrays["control"], atol, rtol
                ),
                "ordinary_vs_fp64": comparison(arrays["ordinary"], arrays["reference"], atol, rtol),
                "control_vs_fp64": comparison(arrays["control"], arrays["reference"], atol, rtol),
            }
            for metric, result in metrics.items():
                require(
                    result == record[metric], f"Recorded comparison differs: {name}/{key}/{metric}"
                )
            counts["convolution_layout_failures"] += not metrics["channels_last_vs_contiguous"][
                "passed"
            ]
            counts["contiguous_convolution_fp64_failures"] += not metrics["control_vs_fp64"][
                "passed"
            ]
            counts["channels_last_convolution_fp64_failures"] += not metrics["ordinary_vs_fp64"][
                "passed"
            ]
            inputs = {
                k: array_identity(arrays[k])
                for k in ("input", "weight", "reference", *(["bias"] if bias else []))
            }
            if key in common:
                require(
                    inputs == common[key], f"Common convolution operands/reference differ: {key}"
                )
            common[key] = inputs
            output_hashes.setdefault(key, {})[name] = {
                k: array_identity(arrays[k]) for k in ("ordinary", "control")
            }
            rows.append(
                dict(
                    worker=name,
                    family="convolution",
                    case=key,
                    dtype=dtype,
                    operation="conv3d",
                    layout="channels_last_vs_contiguous",
                    ordinary_failed_coordinates=metrics["ordinary_vs_fp64"]["failed_coordinates"],
                    control_failed_coordinates=metrics["control_vs_fp64"]["failed_coordinates"],
                    ordinary_guard_intact="not_applicable",
                    control_guard_intact="not_applicable",
                    reference="saved_pytorch_cpu_fp64",
                )
            )
        for key, value in data["summary"].items():
            require(value == counts[key], f"Worker summary differs: {key}")
        require(data["summary"] == row["diagnostic_counts"], "Parent summary differs")
        worker_counts.append(counts)
    return {
        "schema": "gpu-owned-layout-independent-array-audit-v1",
        "complete": True,
        "passed": True,
        "scope": "Closed saved-array/hash/layout audit; "
        "no GPU execution, timing or new FP64 equations",
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "array_archives": 328,
        "common_operand_sets_equal_across_four_workers": len(common),
        "all_recorded_array_comparisons_recomputed": 680,
        "workers": worker_counts,
        "bindings": bindings,
        "common_inputs_and_references": common,
        "output_identities": output_hashes,
        "case_rows": rows,
        "limits": [
            "The matrix reference is same-dtype PyTorch CPU, not independent FP64.",
            "Convolution comparisons reuse saved PyTorch CPU FP64 results; "
            "no new convolution is executed.",
            "NPZ preserves logical values, not original GPU physical strides; "
            "recorded layouts are checked.",
            "Execution authentication/postflight statements are bound, "
            "not re-executed by this audit.",
            "Diagnostic mode and contiguous-copy controls "
            "do not qualify the candidate framework patch.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["case_rows"][0]))
        writer.writeheader()
        writer.writerows(result["case_rows"])
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "passed",
                    "array_archives",
                    "common_operand_sets_equal_across_four_workers",
                    "workers",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
