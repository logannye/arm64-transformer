"""All-width captured-library model and identical-gradient optimizer accuracy.

Use already exposed CPU development states as identical operands for each arm.
These comparisons are separate from engine-specific training repeatability.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCER = "integration/trained-precision/library_common_state.py"
SPEC = importlib.util.spec_from_file_location(
    "library_common_capture", ROOT / "integration/trained-precision/captured_replay.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)
SUITE, require = REPLAY.SUITE, REPLAY.SUITE.require
FLOOR = 16 * 1024**3
MODEL_STEPS = (0, 8, 64)
ADAM_STEPS = (0, 64)
ADAM_COUNTS = (1, 2, 10)


def cases():
    return [f"model/step-{s}/b{b}-t32" for s in MODEL_STEPS for b in (1, 3, 16)] + [
        f"adam/step-{s}/count-{n}" for s in ADAM_STEPS for n in ADAM_COUNTS
    ]


def checkpoint(np, trajectory, step, count):
    row = next(r for r in trajectory["saved"] if r["step"] == step)
    require(set(row["files"]) == {"parameters", "m", "v"}, "Incomplete common state")
    state = dict(step=step)
    for key, bound in row["files"].items():
        SUITE.binding(bound["path"], bound)
        require(bound["bytes"] == count * 4, "Wrong common-state size")
        state[key] = np.fromfile(bound["path"], "<f4")
        require(np.isfinite(state[key]).all(), "Nonfinite common state")
    require((state["v"] >= 0).all(), "Negative second moment")
    return state


def exact_state(np, left, right):
    return left["step"] == right["step"] and all(
        np.asarray(left[k]).dtype == np.asarray(right[k]).dtype
        and np.asarray(left[k]).shape == np.asarray(right[k]).shape
        and np.asarray(left[k]).tobytes() == np.asarray(right[k]).tobytes()
        for k in ("parameters", "m", "v")
    )


def compare(np, actual, expected, kind):
    atol, rtol = {
        "forward": (2e-5, 2e-4),
        "gradient": (3e-5, 3e-3),
        "optimizer": (3e-6, 3e-4),
        "oracle": (1e-10, 1e-10),
        "exact": (0, 0),
    }[kind]
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, "Common-state comparison shape differs")
    finite = np.isfinite(a) & np.isfinite(b)
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    failed = ~finite | (difference > atol + rtol * np.abs(b))
    return dict(
        kind=kind,
        atol=atol,
        rtol=rtol,
        passed=not bool(failed.any()),
        failed_coordinates=int(failed.sum()),
        max_absolute_error=float(difference[np.isfinite(difference)].max(initial=0)),
    )


def run(args):
    require(os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(args.width), "Select width first")
    require(not args.output.exists(), "Fresh common-state output required")
    sys.pycache_prefix = str(args.output / "unused-bytecode-cache")
    SUITE.check_controls()
    capture, plan = REPLAY.captured_plan(args.capture, args.capture_sha256)
    authority, native, inventory, _ = SUITE.bootstrap(plan)
    receipt, _ = authority.verify()
    SUITE.output_path(args.output, receipt["project"])
    SUITE.pinned(ROOT / PRODUCER, receipt["project"]["files"][PRODUCER])
    SUITE.pinned(args.trajectory, args.trajectory_sha256)
    trajectory = SUITE.read_json(args.trajectory)
    require(
        trajectory["schema"] == "maintained-library-optimizer-development-v1"
        and trajectory["engine"] == "cpu_torch"
        and trajectory["width"] == args.width
        and trajectory["seed"] == 20261103
        and trajectory["start_step"] == 0
        and trajectory["complete"]
        and trajectory["passed"]
        and trajectory["integrity_after"],
        "Complete exposed CPU development trajectory required",
    )
    previous_capture = trajectory["capture"]
    SUITE.binding(previous_capture["path"], previous_capture)
    previous = SUITE.read_json(previous_capture["path"])
    require(
        previous["runtime_identity_sha256"] == capture["runtime_identity_sha256"],
        "Common-state operands came from different runtime bytes",
    )
    for framework in ("pytorch", "mlx"):
        old, new = (c["precision_api_derivations"][framework] for c in (previous, capture))
        require(
            old["receipt"] == new["receipt"]
            and old["helper"]["sha256"] == new["helper"]["sha256"]
            and old["helper"]["bytes"] == new["helper"]["bytes"],
            "Common-state operands came from different package derivations",
        )
    import numpy as np
    import torch

    from transformer import library_training, spec
    from transformer.gpu_admission_checks import adam_reference, numpy_model
    from transformer.gpu_repeatability import _torch_reference
    from transformer.gpu_study_cpu_admission import _compute

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(
        shutil.disk_usage(args.output.parent).free > FLOOR + 1536 * 1024**2,
        "Insufficient common-state output budget",
    )
    SUITE.binding(trajectory["fixture"]["path"], trajectory["fixture"])
    fixture = SUITE.read_json(trajectory["fixture"]["path"])
    SUITE.binding(fixture["training"]["path"], fixture["training"])
    rows = np.fromfile(fixture["training"]["path"], "<i4").reshape(2048, 33)
    rng = np.random.default_rng(np.random.SeedSequence([20261103, args.width, 91]))
    fixed_gradient = rng.normal(0, 0.01, spec.PARAM_COUNT).astype(np.float32)
    args.output.mkdir()
    record = dict(
        schema="maintained-library-common-state-development-v1",
        engine=args.engine,
        width=args.width,
        pid=os.getpid(),
        seed=20261103,
        capture=SUITE.binding(args.capture),
        producer=SUITE.binding(ROOT / PRODUCER),
        trajectory=SUITE.binding(args.trajectory),
        expected_cases=cases(),
        cases=[],
        complete=False,
        passed=False,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )
    backend = None
    try:
        api = library_training.selected_api(args.engine)
        framework = "mlx" if args.engine == "mlx" else "pytorch"
        require(
            receipt["frameworks"][framework]["selection"]["precision_api"]["module"]
            == api.__name__,
            "Unselected maintained API",
        )
        record["selected_api"] = dict(module=api.__name__, path=api.__file__, identity=api.IDENTITY)
        record["origins_before"] = REPLAY.origins(args.engine, authority, native, inventory)
        if args.engine != "cpu_torch":
            backend = library_training.create_gpu_backend(
                args.engine, checkpoint(np, trajectory, 0, spec.PARAM_COUNT)["parameters"]
            )
            record["capabilities"] = backend.capabilities()
        for name in cases():
            require(shutil.disk_usage(args.output).free > FLOOR, "Disk reserve reached")
            family, step_text, shape = name.split("/")
            state = checkpoint(np, trajectory, int(step_text.split("-")[1]), spec.PARAM_COUNT)
            before = {
                k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in state.items()
            }
            if family == "model":
                batch = int(shape.split("-")[0][1:])
                x, y = (
                    np.ascontiguousarray(rows[:batch, :-1]),
                    np.ascontiguousarray(rows[:batch, 1:]),
                )
                expected = numpy_model(state["parameters"], x, y)
                double = _torch_reference(state["parameters"], x, y, backward=True)
                oracle = {k: compare(np, double[k], expected[k], "oracle") for k in expected}
                outputs = []
                for _ in range(2):
                    if backend is None:
                        value = _compute(
                            "cpu_torch",
                            dict(state, mode=np.asarray("forward_gradient"), tokens=x, targets=y),
                            library_api=True,
                        )
                        actual = {k: value[k] for k in ("logits", "loss", "gradient")}
                    else:
                        backend.load_state(state)
                        device_batch = backend.upload_inputs(x, y)
                        logits = backend.to_numpy(backend.forward(device_batch)).copy()
                        loss, grad = backend.loss_and_grad(device_batch)
                        actual = dict(
                            logits=logits,
                            loss=backend.to_numpy(loss).copy(),
                            gradient=backend.to_numpy(grad).copy(),
                        )
                        require(
                            exact_state(np, backend.export_state(), state),
                            "Forward/backward changed common state",
                        )
                    outputs.append(actual)
                require(
                    all(
                        np.asarray(outputs[0][k]).dtype == np.asarray(outputs[1][k]).dtype
                        and np.asarray(outputs[0][k]).tobytes()
                        == np.asarray(outputs[1][k]).tobytes()
                        for k in expected
                    ),
                    "Common-state repetition differs",
                )
                actual = outputs[0]
                repetitions = {
                    k: [
                        dict(
                            dtype=np.asarray(o[k]).dtype.str,
                            shape=list(np.asarray(o[k]).shape),
                            sha256=hashlib.sha256(np.asarray(o[k]).tobytes()).hexdigest(),
                        )
                        for o in outputs
                    ]
                    for k in expected
                }
                checks = {
                    k: compare(
                        np, actual[k], expected[k], "gradient" if k == "gradient" else "forward"
                    )
                    for k in expected
                }
            else:
                count = int(shape.split("-")[1])
                expected = adam_reference(state, fixed_gradient, count)
                if backend is None:
                    actual = {k: np.asarray(v).copy() for k, v in state.items()}
                    for _ in range(count):
                        value = _compute(
                            "cpu_torch",
                            dict(
                                actual,
                                mode=np.asarray("adam"),
                                step=np.asarray(int(actual["step"]) + 1),
                                gradient=fixed_gradient,
                            ),
                            library_api=True,
                        )
                        actual = {k: value[k] for k in state}
                else:
                    backend.load_state(state)
                    for _ in range(count):
                        backend.apply_gradient(fixed_gradient)
                    actual = backend.export_state()
                checks = {
                    k: compare(np, actual[k], expected[k], "exact" if k == "step" else "optimizer")
                    for k in expected
                }
                oracle = {}
                repetitions = None
            require(
                before
                == {
                    k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in state.items()
                },
                "Common operands mutated",
            )
            path = args.output / (name.replace("/", "-") + ".npz")
            with path.open("xb") as stream:
                np.savez_compressed(stream, **actual)
            record["cases"].append(
                dict(
                    name=name,
                    actual=SUITE.binding(path),
                    comparisons=checks,
                    oracle=oracle,
                    repetitions=repetitions,
                    passed=all(c["passed"] for c in [*checks.values(), *oracle.values()]),
                )
            )
            print(json.dumps(dict(case=name, passed=record["cases"][-1]["passed"])), flush=True)
        record.update(complete=True, passed=all(c["passed"] for c in record["cases"]))
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        try:
            if backend is not None:
                backend.close()
            SUITE.check_controls()
            SUITE.verify_import_resolution(inventory, sys.path)
            record["origins_after"] = REPLAY.origins(args.engine, authority, native, inventory)
            SUITE.pinned(args.capture, args.capture_sha256)
            SUITE.pinned(args.trajectory, args.trajectory_sha256)
            require(not Path(sys.pycache_prefix).exists(), "Bytecode cache appeared")
            record["integrity_after"] = True
        except BaseException:
            record.update(passed=False, integrity_error=traceback.format_exc())
        SUITE.write_json(args.output / "result.json", record)
    return int(not record["passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("cpu_torch", "mps", "mlx"), required=True)
    parser.add_argument("--width", type=int, choices=(32, 64, 128, 256, 512, 1024), required=True)
    for name in ("capture", "trajectory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("capture", "trajectory"):
        parser.add_argument("--" + name + "-sha256", required=True)
    raise SystemExit(run(parser.parse_args()))
