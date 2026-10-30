"""Create bounded development trajectories from authenticated training-only data.

No validation/test corpus, reserved admission seed, timing or formal fit. Run one
width in a fresh process. A second process must reproduce every state digest.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261103
STEPS = 64
CHECKPOINTS = (0, 1, 8, 32, 64)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def worker(args):
    policy = args.precision_policy
    if os.environ.get("ARM64_TRANSFORMER_WIDTH") != str(args.width):
        raise ValueError("Set the width before importing model code")
    if sha(args.library) != args.library_sha256:
        raise ValueError("Unbound native binary")
    sys.path.insert(0, str(ROOT))
    import ctypes

    import numpy as np
    import torch

    from transformer import spec
    from transformer.trained_precision import native_evaluate, source_inventory
    from transformer.validate import Native

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before = source_inventory()
    producer_sha = sha(Path(__file__))
    output = args.output
    required = len(CHECKPOINTS) * spec.PARAM_COUNT * 4 * (3 if policy == 15 else 1)
    storage_root = output.parent
    while not storage_root.exists():
        storage_root = storage_root.parent
    if shutil.disk_usage(storage_root).free < required + 5 * 1024**3:
        raise ValueError("Insufficient space for saved states and 5 GiB reserve")
    output.mkdir(parents=True, exist_ok=False)
    report = args.evidence_root / f"experiments/width-sweep-v6/width-{args.width}/summary.json"
    if sha(report) != args.summary_sha256:
        raise ValueError("Original fixture authority changed")
    run = next(r for r in json.loads(report.read_text())["runs"] if r["fixture"] == "seed-42")
    fixture = args.evidence_root / f"results/inputs-v6/width-{args.width}/seed-42"
    inputs = {}
    for name in ("initial.f32", "train.i32"):
        path = fixture / name
        if sha(path) != run["fixture_sha256"][name]:
            raise ValueError("Training input changed")
        inputs[name] = {"path": str(path), "sha256": sha(path)}
    flat = np.fromfile(fixture / "initial.f32", "f4")
    rows = np.fromfile(fixture / "train.i32", "i4").reshape(-1, 33)
    if flat.shape != (spec.PARAM_COUNT,) or rows.shape != (2048, 33):
        raise ValueError("Original fixture dimensions changed")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, args.width, 73]))
    schedule = rng.integers(0, len(rows), (STEPS, 16), dtype=np.int32)
    schedule.tofile(output / "schedule.i32")
    evaluation = np.empty((32, 33), "i4")
    evaluation[:16] = rows[rng.integers(0, len(rows), 16)]
    evaluation[16:20] = 0
    evaluation[20:24] = 127
    evaluation[24:28] = np.arange(33, dtype=np.int32) % 2 * 127
    evaluation[28:] = rng.integers(0, 128, (4, 33), dtype=np.int32)
    evaluation.tofile(output / "rows.i32")
    native = Native(args.library)
    native.lib.tf_precision_policy.argtypes = []
    native.lib.tf_precision_policy.restype = ctypes.c_uint64
    if native.lib.tf_precision_policy() != policy:
        raise ValueError("Fresh trajectory binary differs from the declared precision policy")
    m, v = np.zeros_like(flat), np.zeros_like(flat)
    record = {
        "schema": "trained-precision-fresh-development-v1"
        if policy == 1
        else "trained-precision-fresh-development-v2",
        "seed": SEED,
        "width": args.width,
        "steps": STEPS,
        "pid": os.getpid(),
        "executable": str(Path(sys.executable).resolve(strict=True)),
        "checkpoints": list(CHECKPOINTS),
        "source_sha256": before,
        "producer_sha256": producer_sha,
        "library_sha256": args.library_sha256,
        "backend_before": native.backend_identity,
        "training_inputs": inputs,
        "summary_sha256": args.summary_sha256,
        "formal_eligible": False,
        "admission_stream_opened": False,
        "complete": False,
        "states": [],
        "saved": {},
    }
    if policy == 15:
        record.update(precision_policy=policy, saved_optimizer={})
    for step in range(STEPS + 1):
        if shutil.disk_usage(output).free < 5 * 1024**3:
            raise ValueError("Development trajectory reached the 5 GiB disk reserve")
        if step:
            batch = rows[schedule[step - 1]]
            result = native_evaluate(
                native,
                flat,
                np.ascontiguousarray(batch[:, :-1]),
                np.ascontiguousarray(batch[:, 1:]),
                True,
            )
            if not all(np.isfinite(a).all() for a in result.values()):
                raise ValueError("Nonfinite development forward/backward")
            native.call("adam", flat, result["gradient"], m, v, step, spec.PARAM_COUNT)
        if not all(np.isfinite(a).all() for a in (flat, m, v)):
            raise ValueError("Nonfinite development optimizer state")
        state = {
            name: hashlib.sha256(array.tobytes()).hexdigest()
            for name, array in (("parameters", flat), ("m", m), ("v", v))
        }
        record["states"].append({"step": step, **state})
        if step in CHECKPOINTS:
            path = output / f"step-{step:03d}.f32"
            flat.tofile(path)
            record["saved"][path.name] = sha(path)
            if policy == 15:
                for name, values in (("m", m), ("v", v)):
                    path = output / f"step-{step:03d}-{name}.f32"
                    values.tofile(path)
                    record["saved_optimizer"][path.name] = sha(path)
        print(json.dumps({"width": args.width, "step": step, "finite": True}), flush=True)
    if before != source_inventory() or producer_sha != sha(Path(__file__)):
        raise ValueError("Model source changed during development training")
    if sha(args.library) != args.library_sha256 or native.lib.tf_precision_policy() != policy:
        raise ValueError("Native binary changed during development training")
    record.update(
        complete=True,
        backend_after=native.backend_identity,
        rows_sha256=sha(output / "rows.i32"),
        schedule_sha256=sha(output / "schedule.i32"),
    )
    with (output / "result.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(512, 1024), required=True)
    parser.add_argument("--precision-policy", type=int, choices=(1, 15), default=1)
    for name in ("evidence-root", "output", "library"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("library-sha256", "summary-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        worker(args)
    except BaseException:
        if args.output.is_dir() and not (args.output / "result.json").exists():
            with (args.output / "result.json").open("x") as stream:
                json.dump(
                    {
                        "complete": False,
                        "formal_eligible": False,
                        "admission_stream_opened": False,
                        "error": traceback.format_exc(),
                    },
                    stream,
                    indent=2,
                )
        raise
