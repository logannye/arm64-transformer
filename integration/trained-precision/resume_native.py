"""Resume the declared policy-15 development state in a fresh process.

Check all parameter/Adam-state bytes against the uninterrupted run at steps
32–64. This is an exact within-engine continuation check, not cross-engine
accuracy, a training-quality result, timing or formal admission.
"""

import argparse
import ctypes
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def inputs(path, expected):
    require(sha(path) == expected, "Uninterrupted receipt changed")
    record = json.loads(Path(path).read_text())
    require(
        record["schema"] == "trained-precision-fresh-development-v2"
        and record["precision_policy"] == 15
        and record["complete"] is True
        and record["formal_eligible"] is False
        and record["admission_stream_opened"] is False
        and record["steps"] == 64
        and record["seed"] == 20261103
        and record["width"] in (512, 1024)
        and record["checkpoints"] == [0, 1, 8, 32, 64],
        "Wrong uninterrupted development run",
    )
    require([s["step"] for s in record["states"]] == list(range(65)), "Incomplete state history")
    rows = record["training_inputs"]["train.i32"]
    paths = {"rows": Path(rows["path"]), "schedule": path.parent / "schedule.i32"}
    bindings = {"rows": rows["sha256"], "schedule": record["schedule_sha256"]}
    for name, suffix in (("parameters", ""), ("m", "-m"), ("v", "-v")):
        filename = f"step-032{suffix}.f32"
        paths[name] = path.parent / filename
        saved = record["saved"] if name == "parameters" else record["saved_optimizer"]
        bindings[name] = saved[filename]
        require(bindings[name] == record["states"][32][name], "Saved state differs from history")
        require(
            paths[name].stat().st_size
            == (8 * record["width"] ** 2 + 301 * record["width"] + 128) * 4,
            "Saved state size differs",
        )
    require(paths["rows"].stat().st_size == 2048 * 33 * 4, "Wrong training corpus size")
    require(paths["schedule"].stat().st_size == 64 * 16 * 4, "Wrong schedule size")
    for name, file in paths.items():
        require(sha(file) == bindings[name], "Continuation input changed: " + name)
    return record, paths, bindings


def worker(args):
    record, paths, bindings = inputs(args.receipt, args.receipt_sha256)
    require(
        os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(record["width"]), "Set the width first"
    )
    require(sha(args.library) == record["library_sha256"], "Native library changed")
    require(os.getpid() != record["pid"], "Continuation must use a fresh process")
    require(
        sha(Path(__file__).with_name("fresh_states.py")) == record["producer_sha256"],
        "Producer changed",
    )
    sys.path.insert(0, str(ROOT))
    import numpy as np

    from transformer import spec
    from transformer.trained_precision import native_evaluate, source_inventory
    from transformer.validate import Native

    require(source_inventory() == record["source_sha256"], "Model source changed")
    own_sha = sha(Path(__file__))
    native = Native(args.library)
    native.lib.tf_precision_policy.argtypes = []
    native.lib.tf_precision_policy.restype = ctypes.c_uint64
    require(native.lib.tf_precision_policy() == 15, "Wrong native precision policy")
    before = native.backend_identity
    require(
        before["selected_kind"] == record["backend_before"]["selected_kind"],
        "Native dispatch differs",
    )
    flat, m, v = (np.fromfile(paths[name], "f4") for name in ("parameters", "m", "v"))
    rows = np.fromfile(paths["rows"], "i4").reshape(2048, 33)
    schedule = np.fromfile(paths["schedule"], "i4").reshape(64, 16)
    require(np.all((rows >= 0) & (rows < 128)), "Invalid tokens")
    require(np.all((schedule >= 0) & (schedule < len(rows))), "Invalid schedule")
    states = []
    for step in range(32, 65):
        if step > 32:
            batch = rows[schedule[step - 1]]
            result = native_evaluate(
                native,
                flat,
                np.ascontiguousarray(batch[:, :-1]),
                np.ascontiguousarray(batch[:, 1:]),
                True,
            )
            require(
                all(np.isfinite(value).all() for value in result.values()),
                "Nonfinite forward/backward",
            )
            native.call("adam", flat, result["gradient"], m, v, step, spec.PARAM_COUNT)
        require(
            all(np.isfinite(value).all() for value in (flat, m, v)), "Nonfinite optimizer state"
        )
        state = {
            "step": step,
            **{
                name: hashlib.sha256(value.tobytes()).hexdigest()
                for name, value in (("parameters", flat), ("m", m), ("v", v))
            },
        }
        states.append(state)
        require(state == record["states"][step], f"Checkpoint continuation differs at step {step}")
        print(json.dumps({"step": step, "exact_continuation": True}), flush=True)
    after = native.backend_identity
    for kind in ("sme", "neon"):
        for direction in ("forward", "backward"):
            key = f"{kind}_{direction}_calls"
            selected = "sme" if before["selected_kind"] == 1 else "neon"
            require(
                after[key] - before[key] == (32 * 16 * 3 if kind == selected else 0),
                "Wrong continuation dispatch",
            )
    require(after["selected_kind"] == before["selected_kind"], "Dispatch changed")
    require(
        source_inventory() == record["source_sha256"] and sha(Path(__file__)) == own_sha,
        "Source changed during continuation",
    )
    require(sha(args.library) == record["library_sha256"], "Binary changed during continuation")
    require(
        inputs(args.receipt, args.receipt_sha256)[2] == bindings,
        "Inputs changed during continuation",
    )
    return {
        "schema": "trained-precision-native-continuation-v1",
        "complete": True,
        "formal_eligible": False,
        "admission_stream_opened": False,
        "pid": os.getpid(),
        "width": record["width"],
        "precision_policy": 15,
        "receipt_sha256": args.receipt_sha256,
        "source_sha256": record["source_sha256"],
        "producer_sha256": own_sha,
        "library_sha256": record["library_sha256"],
        "input_sha256": bindings,
        "backend_before": before,
        "backend_after": after,
        "states": states,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = worker(args)
    except BaseException:
        result = {
            "complete": False,
            "formal_eligible": False,
            "admission_stream_opened": False,
            "error": traceback.format_exc(),
        }
        raise
    finally:
        with (args.output / "result.json").open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
