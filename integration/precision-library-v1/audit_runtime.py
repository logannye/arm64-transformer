"""Independently audit retained primitive arrays and captured package provenance."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROVENANCE = load(
    "primitive_independent_provenance", "integration/trained-precision/audit_captured_replay.py"
)
RAW = load("primitive_independent_raw", "integration/trained-precision/audit_replay.py")
SUITE = PROVENANCE.SUITE
require = SUITE.require


def arrays_decision(arrays):
    names = ("output", "gradient0", "gradient1", "gradient2")
    expected = {
        f"{label}_{name}" for label in ("actual", "repeat", "numpy64", "torch64") for name in names
    }
    expected |= {"input0", "input1", "input2", "upstream"}
    require(set(arrays) == expected, "Primitive raw array inventory differs")
    inputs = {}
    for name in ("input0", "input1", "input2", "upstream"):
        value = arrays[name]
        require(
            value.dtype == np.float32 and np.isfinite(value).all(),
            "Primitive input dtype/value differs",
        )
        inputs[name] = dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )
    checks = {}
    for name in names:
        a, r, n, t = [
            arrays[label + "_" + name] for label in ("actual", "repeat", "numpy64", "torch64")
        ]
        require(
            a.shape == r.shape == n.shape == t.shape
            and a.dtype == r.dtype == np.float32
            and n.dtype == t.dtype == np.float64,
            "Primitive result shape/dtype differs",
        )
        require(
            all(np.isfinite(value).all() for value in (a, r, n, t)), "Nonfinite primitive result"
        )
        atol, rtol = (2e-5, 2e-4) if name == "output" else (3e-5, 3e-3)
        checks[name] = dict(
            agreement=bool(np.all(np.abs(a - n) <= atol + rtol * np.abs(n))),
            dual_oracle=bool(np.all(np.abs(t - n) <= 1e-10 + 1e-10 * np.abs(n))),
            exact_replay=a.tobytes() == r.tobytes(),
            max_absolute_error=float(np.max(np.abs(a - n))),
            atol=atol,
            rtol=rtol,
        )
    return inputs, checks


def validate_shapes(arrays, width, operation):
    require(width in (32, 64, 128, 256, 512, 1024), "Unknown primitive model width")
    if operation == "linear":
        shapes = ((2, 3, width), (3 * width, width), (3 * width,))
        output = (2, 3, 3 * width)
    elif operation == "layer_norm":
        shapes, output = ((2, 3, width), (width,), (width,)), (2, 3, width)
    else:
        require(operation == "scaled_dot_product_attention", "Unknown primitive operation")
        shapes = ((2, 4, 5, width // 4),) * 3
        output = shapes[0]
    require(
        all(arrays["input" + str(i)].shape == shape for i, shape in enumerate(shapes))
        and arrays["upstream"].shape == output,
        "Primitive input width/layout differs",
    )
    for label in ("actual", "repeat", "numpy64", "torch64"):
        require(
            arrays[label + "_output"].shape == output
            and all(
                arrays[label + "_gradient" + str(i)].shape == shape
                for i, shape in enumerate(shapes)
            ),
            "Primitive derivative width/layout differs",
        )


def audit(receipt_path, receipt_sha, capture_path, capture_sha, raw_store):
    bound = SUITE.pinned(receipt_path, receipt_sha)
    capture_binding = SUITE.pinned(capture_path, capture_sha)
    result, capture = SUITE.read_json(receipt_path), SUITE.read_json(capture_path)
    require(
        result["schema"] == "maintained-library-primitive-execution-v1"
        and result["capture"] == capture_binding
        and result["engine"] in {"cpu_torch", "mps", "mlx"}
        and result["width"] in {32, 64, 128, 256, 512, 1024}
        and result["complete"] is True
        and result["integrity_after"] is True
        and result["formal_eligible"] is False
        and result["formal_fits"] == 0
        and result["admission_stream_opened"] is False
        and "error" not in result
        and "integrity_error" not in result,
        "Incomplete, foreign or improperly qualified primitive worker",
    )
    for key in ("runtime_inventory", "runtime_receipt", "framework_selection"):
        SUITE.binding(capture[key]["path"], capture[key])
    runtime = SUITE.read_json(capture["runtime_inventory"]["path"])
    selected = SUITE.read_json(capture["runtime_receipt"]["path"])
    producer = result["producer"]
    SUITE.binding(producer["path"], producer)
    require(
        producer["sha256"]
        == selected["project"]["files"]["integration/precision-library-v1/validate_runtime.py"],
        "Primitive producer is unbound",
    )
    framework = "mlx" if result["engine"] == "mlx" else "pytorch"
    choice = selected["frameworks"][framework]["selection"]["precision_api"]
    require(
        choice == result["selected_api"]
        and choice["module"]
        == ("mlx.nn.precise_fp32" if framework == "mlx" else "torch.nn.precise_fp32")
        and choice["stock_operators_unchanged"] is True
        and choice["formal_eligible"] is False
        and choice["derivation_receipt_sha256"]
        == capture["precision_api_derivations"][framework]["receipt"]["sha256"],
        "Primitive API selection differs",
    )
    for phase in ("before", "after"):
        PROVENANCE.audit_origins(
            result["origins_" + phase], result["engine"], capture, runtime, selected
        )
    loaded = result["origins_after"]["loaded_python_files"]
    module = choice["module"]
    require(
        loaded[module]["path"] == result["module_path"], "Actual primitive package module differs"
    )
    prefix = module.rsplit(".", 1)[0]
    for name in (module, prefix + "._fp32_precise_math", prefix + "._fp32_precise_dense"):
        PROVENANCE.known_file(
            loaded[name]["path"], runtime, capture["runtime_root"], selected["project"]
        )
    require(
        [row["id"] for row in result["malformed_inputs"]]
        == [
            "linear_wrong_dtype",
            "linear_incompatible_shape",
            "norm_negative_epsilon",
            "norm_incompatible_affine",
            "attention_noncausal",
            "attention_dropout",
        ]
        and all(
            row["rejected"] is True and row["error_type"] == "ValueError" and row["message"]
            for row in result["malformed_inputs"]
        ),
        "Primitive malformed-input gate differs",
    )
    ids = [
        (operation, index)
        for operation in ("linear", "layer_norm", "scaled_dot_product_attention")
        for index in (0, 1)
    ]
    require(
        [(row["operation"], row["index"]) for row in result["cases"]] == ids,
        "Missing/reordered primitive cases",
    )
    rows, outputs = [], {}
    for row in result["cases"]:
        filename = f"{row['operation']}-{row['index']}.npz"
        with (
            RAW.raw_path(receipt_path.parent, filename, row["raw"]["sha256"], raw_store) as path,
            np.load(path, allow_pickle=False) as archive,
        ):
            arrays = {name: archive[name] for name in archive.files}
        require(len(arrays) == len(archive.files), "Duplicate primitive array member")
        validate_shapes(arrays, result["width"], row["operation"])
        inputs, checks = arrays_decision(arrays)
        require(checks == row["checks"], "Primitive decision differs from raw arrays")
        prior = outputs.get(row["operation"])
        current = arrays["actual_output"].tobytes()
        changed = prior is None or prior != current
        require(row["changed_output"] is changed and changed, "Missing changed-input output proof")
        outputs[row["operation"]] = current
        for execution in row["executions"]:
            if result["engine"] == "mlx":
                require(
                    execution["default_device"] == "gpu"
                    and execution["outputs_and_derivatives_dtype"] == "float32",
                    "MLX primitive device/dtype differs",
                )
            else:
                device = "mps" if result["engine"] == "mps" else "cpu"
                require(
                    execution["noncontiguous_operands"] is bool(row["index"])
                    and execution["saved_tensors"]
                    and all(
                        t["device"] == device
                        and t["dtype"] in {"torch.float32", "torch.bool", "torch.int64"}
                        for t in execution["saved_tensors"]
                    ),
                    "Primitive saved-state device/dtype differs",
                )
        require(len(row["executions"]) == 2, "Missing repeated primitive execution")
        passed = all(
            all(check[k] for k in ("agreement", "dual_oracle", "exact_replay"))
            for check in checks.values()
        )
        require(row["passed"] is passed, "Primitive row outcome differs")
        rows.append(
            dict(
                operation=row["operation"],
                index=row["index"],
                inputs=inputs,
                checks=checks,
                passed=passed,
            )
        )
    if result["engine"] == "mlx":
        require(
            set(result["compiled_graphs"]) == {operation for operation, _ in ids},
            "Missing compiled primitive graphs",
        )
        for operation, proof in result["compiled_graphs"].items():
            SUITE.binding(proof["file"]["path"], proof["file"])
            require(
                proof["file"]["path"] == str(receipt_path.parent / (operation + ".dot"))
                and type(proof["python_trace_count"]) is int
                and proof["python_trace_count"] == 1
                and type(proof["executions"]) is int
                and proof["executions"] == 4
                and "Compiled" in Path(proof["file"]["path"]).read_text(),
                "Primitive graph/retrace proof differs",
            )
    passed = all(row["passed"] for row in rows)
    require(result["passed"] is passed, "Primitive summary differs from raw")
    SUITE.pinned(receipt_path, receipt_sha)
    SUITE.pinned(capture_path, capture_sha)
    return dict(
        schema="maintained-library-independent-primitive-audit-v1",
        receipt=bound,
        capture=capture_binding,
        evidence_valid=True,
        numerics_passed=passed,
        engine=result["engine"],
        width=result["width"],
        cases=rows,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "capture", "raw-store"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("receipt-sha256", "capture-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.receipt, args.receipt_sha256, args.capture, args.capture_sha256, args.raw_store
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
