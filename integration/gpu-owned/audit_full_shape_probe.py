"""Audit closed B1/T32 and B16/T32 development evidence without a tensor framework.

This rechecks saved FP64 comparisons and checkpoint bytes. It does not rerun
equations or infer that unsaved trajectory arrays can be reconstructed from hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

PROTOCOL = "gpu-v4-owned-full-shape-development-v1"
MANIFEST_SHA = "1617562aa8cd1e04c358aa5e4261fb72a6fad529472115f1762462c3ed470954"
PRODUCER_SHA = "06a772b3e28819739fc63a3f782631840dcd22cb1eb8d71b9d09b002670b6277"
SHAPES = ((1, 32), (16, 32))
WIDTHS = (32, 64, 128, 256, 512, 1024)
ENGINES = ("cpu_torch", "mps", "mlx")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ordinary(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Expected a regular evidence file")
    require(not any(p.is_symlink() for p in path.parents), "Symlinked evidence ancestor")
    return path


def bound(path, expected, size=None):
    path = ordinary(path)
    require(
        isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
        "Invalid expected evidence digest",
    )
    before = path.stat()
    require(size is None or before.st_size == size, "Evidence size changed")
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    fingerprint = tuple(getattr(before, name) for name in fields)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        require(
            tuple(getattr(os.fstat(stream.fileno()), name) for name in fields) == fingerprint,
            "Evidence replaced before hashing",
        )
        require(
            hashlib.file_digest(stream, "sha256").hexdigest() == expected, "Evidence hash changed"
        )
        require(
            tuple(getattr(os.fstat(stream.fileno()), name) for name in fields) == fingerprint,
            "Evidence changed during hashing",
        )
    require(
        tuple(getattr(ordinary(path).stat(), name) for name in fields) == fingerprint,
        "Evidence changed during read",
    )
    return {"bytes": before.st_size, "sha256": expected}


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        ordinary(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def identity(value):
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def expected_tokens(batch, length):
    require((batch, length) in SHAPES, "Undeclared token shape")
    positions = np.arange(batch * length, dtype=np.int32).reshape(batch, length)
    sequences = positions // length
    return 65 + (positions * 7 + 2 * sequences) % 3, 65 + (positions * 11 + sequences + 1) % 3


def identity_record(value, shape, dtype):
    require(
        isinstance(value, dict)
        and set(value) == {"dtype", "shape", "sha256"}
        and value["dtype"] == dtype
        and value["shape"] == list(shape)
        and all(type(size) is int for size in value["shape"])
        and isinstance(value["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None,
        "Malformed array identity",
    )


def shapes(width, batch=1, length=32):
    require(width in WIDTHS and (batch, length) in SHAPES, "Undeclared model shape")
    parameters = {
        "token": (128, width),
        "position": (32, width),
        "qkv_weight": (3 * width, width),
        "qkv_bias": (3 * width,),
        "attention_weight": (width, width),
        "attention_bias": (width,),
        "ff1_weight": (2 * width, width),
        "ff1_bias": (2 * width,),
        "ff2_weight": (width, 2 * width),
        "ff2_bias": (width,),
        "head_weight": (128, width),
        "head_bias": (128,),
        **{f"ln{i}_{name}": (width,) for i in (1, 2, 3) for name in ("weight", "bias")},
    }
    outputs = {"logits": (batch, length, 128), "loss": ()}
    outputs.update({"gradient/" + name: shape for name, shape in parameters.items()})
    return outputs, 8 * width * width + 301 * width + 128


def arrays(folder, receipt, expected):
    name = receipt["path"]
    require(isinstance(name, str) and Path(name).name == name, "Unsafe array path")
    require(name.endswith(".npz"), "Expected NPZ evidence")
    path = Path(folder) / name
    bound(path, receipt["sha256"], receipt["bytes"])
    require(set(receipt["arrays"]) == set(expected), "Array inventory changed")
    with np.load(path, allow_pickle=False) as saved:
        require(
            len(saved.files) == len(expected) and set(saved.files) == set(expected),
            "Missing, duplicate or unexpected NPZ member",
        )
        values = {key: saved[key] for key in expected}
    for key, value in values.items():
        shape, dtype = expected[key]
        require(value.shape == shape and value.dtype.str == dtype, "Array shape/dtype changed")
        require(identity(value) == receipt["arrays"][key], "Array bytes changed")
        require(bool(np.isfinite(value).all()), "Nonfinite saved array")
    bound(path, receipt["sha256"], receipt["bytes"])
    return values


def comparison(actual, expected, atol, rtol, kind):
    require(actual.shape == expected.shape, "Comparison shape changed")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(e)
    error = np.abs(a - e)
    bad = ~finite | (error > atol + rtol * np.abs(e))
    return {
        "atol": atol,
        "rtol": rtol,
        "tolerance": kind,
        "failed_coordinates": int(bad.sum()),
        "nonfinite_coordinates": int((~finite).sum()),
        "max_absolute_error": float(error[finite].max()) if finite.any() else None,
        "passed": not bool(bad.any()),
    }


def exact_record(records, expected):
    require(set(records) == set(expected), "Exact-replay field inventory changed")
    for name, wanted in expected.items():
        row = records[name]
        require(
            set(row) == {"actual", "expected", "passed"}
            and row["passed"] is True
            and row["actual"] == wanted
            and row["expected"] == wanted,
            "Exact-replay identity differs",
        )


def cells():
    return [
        {
            "width": w,
            "batch": b,
            "length": t,
            "engine": e,
            "process_index": p,
            "id": f"w{w}-b{b}-t{t}-{e}-p{p}",
        }
        for w in WIDTHS
        for b, t in SHAPES
        for e in ENGINES
        for p in (0, 1)
    ]


def validate_manifest(plan):
    require(
        plan["schema"] == "gpu-owned-full-shape-plan-v1"
        and plan["protocol"] == PROTOCOL
        and canonical(plan["cells"]) == canonical(cells())
        and plan["planned_workers"] == 72
        and plan["shapes"] == [list(shape) for shape in SHAPES]
        and plan["development_seed"] == 20261103
        and plan["initial_replays"] == 10
        and plan["training_steps"] == 6
        and plan["trajectories_per_worker"] == 2
        and plan["warmup_steps"] == 2
        and plan["tolerances"]
        == {"forward": [2e-5, 2e-4], "gradient": [3e-5, 3e-3], "oracle": [1e-10, 1e-10]}
        and plan["formal_eligible"] is False
        and type(plan["formal_fits"]) is int
        and plan["formal_fits"] == 0
        and plan["admission_stream_opened"] is False,
        "Frozen full-shape plan changed",
    )
    require(
        plan["source_files"][
            "outputs/arm64-transformer/integration/gpu-owned/model_full_shape_probe.py"
        ]
        == PRODUCER_SHA
        and hashlib.sha256(canonical(plan["source_files"])).hexdigest() == plan["source_sha256"],
        "Frozen source identity inventory changed",
    )


def validate_summary(summary, expected):
    require(
        summary["schema"] == "gpu-owned-full-shape-summary-v1"
        and summary["protocol"] == PROTOCOL
        and summary["manifest_sha256"] == MANIFEST_SHA
        and canonical(summary["planned_workers"]) == canonical(expected)
        and canonical([r["cell"] for r in summary["workers"]]) == canonical(expected)
        and summary["passed"] is True
        and summary["complete"] is True
        and summary["broad_framework_suite_remains_blocking"] is True
        and summary["formal_eligible"] is False
        and type(summary["formal_fits"]) is int
        and summary["formal_fits"] == 0
        and summary["admission_stream_opened"] is False,
        "Incomplete or changed model execution",
    )
    require(
        canonical(summary["fresh_process_pairs"])
        == canonical(
            [
                {"width": w, "batch": b, "length": t, "engine": e, "passed": True}
                for w in WIDTHS
                for b, t in SHAPES
                for e in ENGINES
            ]
        ),
        "Parent fresh-process inventory differs",
    )
    require(
        canonical([(r["width"], r["batch"], r["length"]) for r in summary["common_fixture_checks"]])
        == canonical([(w, b, t) for w in WIDTHS for b, t in SHAPES])
        and all(r["passed"] is True for r in summary["common_fixture_checks"]),
        "Parent common-fixture inventory differs",
    )
    for row in summary["workers"]:
        require(
            type(row.get("pid")) is int
            and row["pid"] > 0
            and type(row.get("returncode")) is int
            and row["returncode"] == 0
            and row["passed"] is True,
            "Invalid launched worker identity or execution",
        )
        receipt = Path(row["receipt"]["path"])
        require(
            receipt.name == "result.json" and receipt.parent.name == row["cell"]["id"],
            "Parent worker receipt path substituted",
        )


def validate_pairs(rows, summary):
    require(
        canonical([row["cell"] for row in rows]) == canonical(cells()),
        "Audited worker inventory changed",
    )
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        require(first["fingerprints"] == second["fingerprints"], "Fresh-process array mismatch")
        require(
            first["trajectory_hashes"] == second["trajectory_hashes"],
            "Fresh-process trajectory mismatch",
        )
        require(
            first["checkpoint_predictions"] == second["checkpoint_predictions"],
            "Fresh-process checkpoint prediction mismatch",
        )
    fixtures = [(w, b, t) for w in WIDTHS for b, t in SHAPES]
    for (width, batch, length), record in zip(
        fixtures, summary["common_fixture_checks"], strict=True
    ):
        selected = [
            r
            for r in rows
            if (r["cell"]["width"], r["cell"]["batch"], r["cell"]["length"])
            == (width, batch, length)
        ]
        operands = [r["fingerprints"]["operands"] for r in selected]
        require(all(value == operands[0] for value in operands), "Engines used different operands")
        by_engine = {
            r["cell"]["engine"]: r["fingerprints"]["operands"]
            for r in selected
            if r["cell"]["process_index"] == 0
        }
        require(record["operands"] == by_engine, "Parent common-fixture evidence differs")
    for width in WIDTHS:
        parameters = [
            row["fingerprints"]["operands"]["parameters"]
            for row in rows
            if row["cell"]["width"] == width
        ]
        require(
            all(value == parameters[0] for value in parameters),
            "Initial parameters differ across batch shapes at the same width",
        )


def audit_worker(folder, worker, cell, launched_pid):
    expected_files = {
        "operands.npz",
        "initial.npz",
        "oracles.npz",
        "checkpoint-step3.npz",
        "checkpoint-step6.npz",
        "checkpoint-step3-logits.npz",
        "final-logits.npz",
    }
    require(
        {p.name for p in Path(folder).glob("*.npz")} == expected_files,
        "Raw archive coverage changed",
    )
    require(
        worker["schema"] == "gpu-owned-full-shape-worker-v1"
        and canonical(worker["cell"]) == canonical(cell)
        and type(launched_pid) is int
        and launched_pid > 0
        and type(worker.get("pid")) is int
        and worker["pid"] == launched_pid
        and worker["protocol"] == PROTOCOL
        and worker["manifest_sha256"] == MANIFEST_SHA
        and worker["complete"] is True
        and worker["passed"] is True
        and worker["formal_eligible"] is False
        and type(worker["formal_fits"]) is int
        and worker["formal_fits"] == 0
        and worker["admission_stream_opened"] is False
        and worker["postflight"]["passed"] is True
        and worker["postflight"]["all_declared_input_bytes_and_file_identities_unchanged"] is True,
        "Worker identity/completion changed",
    )
    require(
        worker["capabilities"]["baseline_class"] == "project_maintained_full_source_build"
        and worker["capabilities"]["standard_release_claim"] is False,
        "Owned framework provenance missing",
    )
    batch, length = cell["batch"], cell["length"]
    output_shapes, count = shapes(cell["width"], batch, length)
    for key, name in (
        ("operands", "operands.npz"),
        ("initial_arrays", "initial.npz"),
        ("oracle_arrays", "oracles.npz"),
        ("checkpoint_logits_array", "checkpoint-step3-logits.npz"),
        ("final_logits_array", "final-logits.npz"),
    ):
        require(worker[key]["path"] == name, "Array role substituted")
    flat = {name: ((count,), "<f4") for name in ("parameters", "m", "v")}
    state_spec = {**flat, "step": ((), "<i8")}
    inputs = arrays(
        folder,
        worker["operands"],
        {
            "parameters": flat["parameters"],
            "tokens": ((batch, length), "<i4"),
            "targets": ((batch, length), "<i4"),
        },
    )
    require(
        bool(((inputs["tokens"] >= 0) & (inputs["tokens"] < 128)).all())
        and bool(((inputs["targets"] >= 0) & (inputs["targets"] < 128)).all()),
        "Tokens outside vocabulary",
    )
    declared_tokens, declared_targets = expected_tokens(batch, length)
    require(
        np.array_equal(inputs["tokens"], declared_tokens)
        and np.array_equal(inputs["targets"], declared_targets),
        "Synthetic token formula changed",
    )
    initial = arrays(
        folder, worker["initial_arrays"], {k: (s, "<f4") for k, s in output_shapes.items()}
    )
    oracle = arrays(
        folder,
        worker["oracle_arrays"],
        {
            f"{engine}/{k}": (s, "<f8")
            for engine in ("numpy", "torch")
            for k, s in output_shapes.items()
        },
    )
    accuracy = {}
    require(set(worker["oracle_agreement"]) == set(output_shapes), "Oracle inventory changed")
    for key in output_shapes:
        check = comparison(oracle[f"torch/{key}"], oracle[f"numpy/{key}"], 1e-10, 1e-10, "oracle")
        require(
            check == worker["oracle_agreement"][key] and check["passed"], "FP64 agreement differs"
        )
        atol, rtol, kind = (
            (3e-5, 3e-3, "gradient") if key.startswith("gradient/") else (2e-5, 2e-4, "forward")
        )
        accuracy[key] = comparison(initial[key], oracle[f"torch/{key}"], atol, rtol, kind)
        require(accuracy[key]["passed"], "Saved initial FP64 accuracy fails")
    require(
        canonical([r["index"] for r in worker["initial_replays"]]) == canonical(list(range(10))),
        "Replay count changed",
    )
    initial_ids = {k: identity(v) for k, v in initial.items()}
    for row in worker["initial_replays"]:
        require(row["accuracy"] == accuracy, "Recorded initial accuracy differs")
        exact_record(row["exact"], initial_ids)
    zero = identity(np.zeros(count, dtype=np.float32))
    initial_state = {
        "parameters": identity(inputs["parameters"]),
        "m": zero,
        "v": zero,
        "step": identity(np.asarray(0, dtype=np.int64)),
    }
    exact_record(worker["warmup_restore"], initial_state)
    require(
        canonical([r["repeat"] for r in worker["trajectories"]]) == canonical([0, 1]),
        "Trajectory repeats changed",
    )
    raw_step3 = arrays(
        folder, worker["checkpoint_logits_array"], {"logits": ((batch, length, 128), "<f4")}
    )
    state_ids, checkpoint_predictions = {}, {3: identity(raw_step3["logits"])}
    for repeat, trajectory in enumerate(worker["trajectories"]):
        require(
            canonical([s["step"] for s in trajectory["steps"]]) == canonical(list(range(1, 7))),
            "Step budget changed",
        )
        for row in trajectory["steps"]:
            step = row["step"]
            require(
                set(row["arrays"]) == {*state_spec, "loss"}, "Trajectory array inventory changed"
            )
            for key, (shape, dtype) in {**state_spec, "loss": ((), "<f4")}.items():
                identity_record(row["arrays"][key], shape, dtype)
            require(
                row["arrays"]["step"] == identity(np.asarray(step, dtype=np.int64)),
                "Wrong optimizer counter",
            )
            if repeat:
                exact_record(row["exact"], state_ids[step])
                require(row["arrays"] == state_ids[step], "Repeat trajectory differs")
            else:
                state_ids[step] = row["arrays"]
            if step in (3, 6):
                wanted = {k: row["arrays"][k] for k in state_spec}
                if repeat == 0:
                    require(
                        row["checkpoint"]["path"] == f"checkpoint-step{step}.npz",
                        "Checkpoint path substituted",
                    )
                    checkpoint = arrays(folder, row["checkpoint"], state_spec)
                    require(
                        {k: identity(v) for k, v in checkpoint.items()} == wanted,
                        "Checkpoint substitution",
                    )
                exact_record(row["loaded_state_exact"], wanted)
                exact_record(row["serialized_state_exact"], wanted)
                prediction = row["prediction_exact"]["logits"]["expected"]
                identity_record(prediction, (batch, length, 128), "<f4")
                if repeat == 0 and step == 6:
                    checkpoint_predictions[step] = prediction
                else:
                    require(
                        checkpoint_predictions[step] == prediction,
                        "Checkpoint prediction differs between trajectories",
                    )
                exact_record(row["prediction_exact"], {"logits": prediction})
    final = arrays(folder, worker["final_logits_array"], {"logits": ((batch, length, 128), "<f4")})
    final_id = identity(final["logits"])
    for trajectory in worker["trajectories"]:
        require(trajectory["final_logits"] == final_id, "Final prediction differs")
        exact_record(trajectory["steps"][-1]["prediction_exact"], {"logits": final_id})
    fingerprints = {
        "operands": {k: identity(v) for k, v in inputs.items()},
        "initial": initial_ids,
        "final_state": state_ids[6],
        "final_logits": final_id,
        "checkpoint_predictions": {
            str(step): value for step, value in checkpoint_predictions.items()
        },
    }
    require(
        worker["fresh_process_fingerprints"] == fingerprints, "Fresh-process fingerprints differ"
    )
    return {
        "cell": cell,
        "pid": launched_pid,
        "accuracy": accuracy,
        "raw_archives_checked": 7,
        "fingerprints": fingerprints,
        "trajectory_hashes": state_ids,
        "checkpoint_predictions": checkpoint_predictions,
    }


def audit(root, manifest, summary_sha):
    root = Path(root)
    bound(manifest, MANIFEST_SHA)
    bound(root / "summary.json", summary_sha)
    plan, summary = read(manifest), read(root / "summary.json")
    expected = cells()
    validate_manifest(plan)
    validate_summary(summary, expected)
    rows, final_bindings = [], []
    for cell, row in zip(expected, summary["workers"], strict=True):
        require(row["returncode"] == 0 and row["passed"] is True, "Failed worker retained")
        receipt = root / cell["id"] / "result.json"
        bound(receipt, row["receipt"]["sha256"])
        worker = read(receipt)
        rows.append(audit_worker(receipt.parent, worker, cell, row["pid"]))
        bound(receipt, row["receipt"]["sha256"])
        final_bindings.append((receipt, row["receipt"]["sha256"], None))
        raw = [
            worker[name]
            for name in (
                "operands",
                "initial_arrays",
                "oracle_arrays",
                "checkpoint_logits_array",
                "final_logits_array",
            )
        ]
        raw.extend(
            step["checkpoint"]
            for step in worker["trajectories"][0]["steps"]
            if step["step"] in (3, 6)
        )
        final_bindings.extend(
            (receipt.parent / item["path"], item["sha256"], item["bytes"]) for item in raw
        )
    validate_pairs(rows, summary)
    for path, sha, size in final_bindings:
        bound(path, sha, size)
    bound(root / "summary.json", summary_sha)
    bound(manifest, MANIFEST_SHA)
    return {
        "schema": "gpu-owned-full-shape-array-audit-v1",
        "passed": True,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "summary_sha256": summary_sha,
        "manifest_sha256": MANIFEST_SHA,
        "declared_producer_sha256": PRODUCER_SHA,
        "auditor_sha256": digest(__file__),
        "workers": len(rows),
        "fresh_process_pairs": 36,
        "common_fixtures": 12,
        "raw_archives_checked": sum(r["raw_archives_checked"] for r in rows),
        "scope": "Saved initial FP64 comparisons and retained checkpoint bytes recomputed; "
        "full-equation execution not rerun",
        "limits": [
            "B1/T32 and B16/T32 synthetic development only; no corpus training or timing study",
            "Other initial replays, intermediate steps and warmup restoration have authenticated "
            "execution-time hash records, not separately retained raw arrays",
            "Raw checkpoint predictions are retained for steps 3 and 6; "
            "other intermediate predictions are not retained",
            "Post-training accuracy versus FP64, complete framework integration and formal "
            "admission remain unchecked here",
            "Runtime provenance is retained execution evidence, not reauthenticated "
            "by this array-only audit",
        ],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output already exists")
    try:
        result = audit(args.root, args.manifest, args.summary_sha256)
    except Exception as error:
        failed = {
            "schema": "gpu-owned-full-shape-array-audit-v1",
            "passed": False,
            "formal_eligible": False,
            "formal_fits": 0,
            "admission_stream_opened": False,
            "summary_sha256": args.summary_sha256,
            "manifest_sha256": MANIFEST_SHA,
            "auditor_sha256": digest(__file__),
            "error_type": type(error).__name__,
            "error": str(error),
            "scope": "Failed or incomplete saved-array audit",
        }
        with args.output.open("x") as stream:
            json.dump(failed, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        raise
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
