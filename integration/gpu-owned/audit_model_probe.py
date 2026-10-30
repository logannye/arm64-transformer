"""Audit saved owned-runtime model evidence using NumPy, without a tensor framework.

This rechecks saved FP64 comparisons and checkpoint bytes. It does not rerun
equations or infer that unsaved trajectory arrays can be reconstructed from hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

PROTOCOL = "gpu-v4-owned-model-compatibility-v2"
MANIFEST_SHA = "06c29116993f0adbaa2bfb6f5f04b79de152345c3549cfaf466c17a9f2e32a86"
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
    before = path.stat()
    require(size is None or before.st_size == size, "Evidence size changed")
    require(digest(path) == expected, "Evidence hash changed")
    after = path.stat()
    require(
        (before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns),
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


def shapes(width):
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
    outputs = {"logits": (3, 3, 128), "loss": ()}
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
        {"width": w, "engine": e, "process_index": p, "id": f"w{w}-{e}-p{p}"}
        for w in WIDTHS
        for e in ENGINES
        for p in (0, 1)
    ]


def validate_summary(summary, expected):
    require(
        summary["schema"] == "gpu-owned-model-summary-v2"
        and summary["protocol"] == PROTOCOL
        and summary["manifest_sha256"] == MANIFEST_SHA
        and summary["planned_workers"] == expected
        and [r["cell"] for r in summary["workers"]] == expected
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
        summary["fresh_process_pairs"]
        == [{"width": w, "engine": e, "passed": True} for w in WIDTHS for e in ENGINES],
        "Parent fresh-process inventory differs",
    )
    require(
        [r["width"] for r in summary["common_fixture_checks"]] == list(WIDTHS)
        and all(r["passed"] is True for r in summary["common_fixture_checks"]),
        "Parent common-fixture inventory differs",
    )


def validate_pairs(rows, summary):
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
    for width, record in zip(WIDTHS, summary["common_fixture_checks"], strict=True):
        selected = [r for r in rows if r["cell"]["width"] == width]
        operands = [r["fingerprints"]["operands"] for r in selected]
        require(all(value == operands[0] for value in operands), "Engines used different operands")
        by_engine = {
            r["cell"]["engine"]: r["fingerprints"]["operands"]
            for r in selected
            if r["cell"]["process_index"] == 0
        }
        require(record["operands"] == by_engine, "Parent common-fixture evidence differs")


def audit_worker(folder, worker, cell):
    expected_files = {
        "operands.npz",
        "initial.npz",
        "oracles.npz",
        "checkpoint-step3.npz",
        "checkpoint-step6.npz",
        "final-logits.npz",
    }
    require(
        {p.name for p in Path(folder).glob("*.npz")} == expected_files,
        "Raw archive coverage changed",
    )
    require(
        worker["schema"] == "gpu-owned-model-worker-v2"
        and worker["cell"] == cell
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
    output_shapes, count = shapes(cell["width"])
    for key, name in (
        ("operands", "operands.npz"),
        ("initial_arrays", "initial.npz"),
        ("oracle_arrays", "oracles.npz"),
        ("final_logits_array", "final-logits.npz"),
    ):
        require(worker[key]["path"] == name, "Array role substituted")
    flat = {name: ((count,), "<f4") for name in ("parameters", "m", "v")}
    state_spec = {**flat, "step": ((), "<i8")}
    inputs = arrays(
        folder,
        worker["operands"],
        {"parameters": flat["parameters"], "tokens": ((3, 3), "<i4"), "targets": ((3, 3), "<i4")},
    )
    require(
        bool(((inputs["tokens"] >= 0) & (inputs["tokens"] < 128)).all())
        and bool(((inputs["targets"] >= 0) & (inputs["targets"] < 128)).all()),
        "Tokens outside vocabulary",
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
        [r["index"] for r in worker["initial_replays"]] == list(range(10)), "Replay count changed"
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
    require([r["repeat"] for r in worker["trajectories"]] == [0, 1], "Trajectory repeats changed")
    state_ids, checkpoint_predictions = {}, {}
    for repeat, trajectory in enumerate(worker["trajectories"]):
        require(
            [s["step"] for s in trajectory["steps"]] == list(range(1, 7)), "Step budget changed"
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
                identity_record(prediction, (3, 3, 128), "<f4")
                if repeat == 0:
                    checkpoint_predictions[step] = prediction
                else:
                    require(
                        checkpoint_predictions[step] == prediction,
                        "Checkpoint prediction differs between trajectories",
                    )
                exact_record(row["prediction_exact"], {"logits": prediction})
    final = arrays(folder, worker["final_logits_array"], {"logits": ((3, 3, 128), "<f4")})
    final_id = identity(final["logits"])
    for trajectory in worker["trajectories"]:
        require(trajectory["final_logits"] == final_id, "Final prediction differs")
        exact_record(trajectory["steps"][-1]["prediction_exact"], {"logits": final_id})
    fingerprints = {
        "operands": {k: identity(v) for k, v in inputs.items()},
        "initial": initial_ids,
        "final_state": state_ids[6],
        "final_logits": final_id,
    }
    require(
        worker["fresh_process_fingerprints"] == fingerprints, "Fresh-process fingerprints differ"
    )
    return {
        "cell": cell,
        "accuracy": accuracy,
        "raw_archives_checked": 6,
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
    require(
        plan["cells"] == expected and plan["protocol"] == PROTOCOL, "Frozen plan inventory changed"
    )
    validate_summary(summary, expected)
    rows, final_bindings = [], []
    for cell, row in zip(expected, summary["workers"], strict=True):
        require(row["returncode"] == 0 and row["passed"] is True, "Failed worker retained")
        receipt = root / cell["id"] / "result.json"
        bound(receipt, row["receipt"]["sha256"])
        worker = read(receipt)
        rows.append(audit_worker(receipt.parent, worker, cell))
        bound(receipt, row["receipt"]["sha256"])
        final_bindings.append((receipt, row["receipt"]["sha256"], None))
        raw = [
            worker[name]
            for name in ("operands", "initial_arrays", "oracle_arrays", "final_logits_array")
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
        "schema": "gpu-owned-model-array-audit-v1",
        "passed": True,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
        "summary_sha256": summary_sha,
        "manifest_sha256": MANIFEST_SHA,
        "auditor_sha256": digest(__file__),
        "workers": len(rows),
        "fresh_process_pairs": 18,
        "raw_archives_checked": sum(r["raw_archives_checked"] for r in rows),
        "scope": "Saved initial FP64 comparisons and retained checkpoint bytes recomputed; "
        "full-equation execution not rerun",
        "limits": [
            "B3/T3 synthetic development only; not the B16/T32 training or B1/T32 inference task",
            "Other initial replays, intermediate steps and warmup restoration have authenticated "
            "execution-time hash records, not separately retained raw arrays",
            "Step-3 checkpoint predictions are checked by execution-time hashes only; "
            "raw logits are retained for step 6",
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
    result = audit(args.root, args.manifest, args.summary_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
