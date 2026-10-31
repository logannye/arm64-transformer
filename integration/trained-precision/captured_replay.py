"""Replay unchanged development equations in an explicitly captured runtime.

The original worker receipt and raw arrays remain unchanged. A separate receipt
binds the capture, actual loaded origins and pre/postflight authentication. This
does not qualify a framework repair or authorize a formal fit.
"""

import argparse
import importlib.util
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCER = "integration/trained-precision/captured_replay.py"
WORKER = "transformer/trained_precision.py"
SPEC = importlib.util.spec_from_file_location(
    "captured_replay_bootstrap", ROOT / "integration/gpu-owned/owned_mlx_suites.py"
)
SUITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUITE)


def worker_arguments(args):
    """Construct one ordinary worker invocation without runtime overrides."""
    SUITE.require(args.engine in {"cpu_torch", "mps", "mlx"}, "Framework engine required")
    SUITE.require(args.width in {512, 1024}, "Wide development model required")
    SUITE.require(
        not args.python_candidate or args.engine == "cpu_torch", "Python candidate is CPU-only"
    )
    SUITE.require(
        args.framework_policy is None or args.engine in {"mps", "mlx"},
        "Framework candidate is GPU-only",
    )
    SUITE.require(
        not args.framework_compiled or (args.engine == "mlx" and args.framework_policy is not None),
        "Compilation requires an explicit MLX candidate",
    )
    command = [
        str(ROOT / WORKER),
        "worker",
        "--engine",
        args.engine,
        "--width",
        str(args.width),
        "--inventory",
        str(args.inventory),
        "--inventory-sha256",
        args.inventory_sha256,
        "--evidence-root",
        str(args.evidence_root),
        "--batch-size",
        str(args.batch_size),
        "--output",
        str(args.output),
        "--raw-store",
        str(args.raw_store),
    ]
    if args.gradients:
        command.append("--gradients")
    if args.python_candidate:
        command.extend(["--python-candidate", "--python-candidate-version", "2"])
    if args.framework_policy is not None:
        command.extend(["--framework-policy", str(args.framework_policy)])
    if args.framework_compiled:
        command.append("--framework-compiled")
    command.extend(["--case-offset", str(args.case_offset)])
    if args.case_limit is not None:
        command.extend(["--case-limit", str(args.case_limit)])
    return command


def captured_plan(path, expected_sha):
    SUITE.pinned(path, expected_sha)
    capture = SUITE.read_json(path)
    SUITE.require(
        capture["schema"] == "gpu-owned-runtime-capture-v1"
        and capture["status"] == "identity_captured_not_qualified"
        and capture["formal_eligible"] is False
        and capture["formal_fits"] == 0
        and capture["admission_stream_opened"] is False,
        "Not a development runtime capture",
    )
    inputs = {
        name: capture[key]
        for name, key in (
            ("inventory", "runtime_inventory"),
            ("selection", "framework_selection"),
            ("receipt", "runtime_receipt"),
        )
    }
    for value in inputs.values():
        SUITE.binding(value["path"], value)
    receipt = SUITE.read_json(inputs["receipt"]["path"])
    SUITE.require(Path(receipt["project"]["root"]) == ROOT, "Captured project root differs")
    for relative in (PRODUCER, WORKER):
        SUITE.require(relative in receipt["project"]["files"], "Replay producer is unbound")
        SUITE.pinned(ROOT / relative, receipt["project"]["files"][relative])
    SUITE.require(
        capture["runtime_identity_sha256"] == receipt["runtime_identity_sha256"],
        "Capture runtime identity differs",
    )
    return capture, {"runtime_root": capture["runtime_root"], "authority_inputs": inputs}


def origins(engine, authority, backend, inventory):
    receipt, _ = verified = authority.verify()
    import numpy as np
    import torch

    SUITE.require(
        np.__version__
        == receipt["frameworks"]["pytorch"]["selection"].get("numpy_abi_version", "1.26.4")
        and str(Path(np.__file__).resolve()) in inventory["layout"]["packages"]["numpy"],
        "Captured NumPy version or origin differs",
    )
    SUITE.require(not torch.ops.loaded_libraries, "Process-local operator library loaded")
    result = {
        "numpy": {"version": np.__version__, "path": str(Path(np.__file__).resolve())},
        "pytorch": backend._loaded_identity("cpu_torch", authority, verified),
    }
    if engine == "mlx":
        import mlx.core as mx

        from transformer.gpu_mlx_compile import verify_compilation

        SUITE.require(mx.default_device() == mx.gpu and mx.metal.is_available(), "MLX GPU required")
        result["mlx"] = backend._loaded_identity("mlx", authority, verified)
        result["compilation"] = backend.validate_compilation_probe(verify_compilation(mx))
    backend.verify_loaded_project_modules(receipt["project"])
    result["loaded_python_files"] = SUITE.verify_loaded_files(
        inventory, authority, receipt["project"]
    )
    result["runtime_identity_sha256"] = receipt["runtime_identity_sha256"]
    return result


def run(args):
    command = worker_arguments(args)
    SUITE.require(not args.output.exists(), "Replay output must be fresh")
    SUITE.require(
        os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(args.width),
        "Set width before starting the interpreter",
    )
    sys.pycache_prefix = str(args.output / "unused-bytecode-cache")
    SUITE.check_controls()
    capture, plan = captured_plan(args.capture, args.capture_sha256)
    authority, backend, inventory, _ = SUITE.bootstrap(plan)
    receipt, _ = authority.verify()
    SUITE.output_path(args.output, receipt["project"])
    record = {
        "schema": "trained-precision-captured-runtime-replay-v1",
        "capture": SUITE.binding(args.capture),
        "producer": SUITE.binding(Path(__file__).resolve()),
        "worker_source": SUITE.binding(ROOT / WORKER),
        "runtime_receipt": plan["authority_inputs"]["receipt"],
        "runtime_identity_sha256": capture["runtime_identity_sha256"],
        "engine": args.engine,
        "width": args.width,
        "command": command,
        "pid": os.getpid(),
        "complete": False,
        "passed": False,
        "formal_eligible": False,
        "formal_fits": 0,
        "admission_stream_opened": False,
    }
    try:
        record["origins_before"] = origins(args.engine, authority, backend, inventory)
        previous_argv = sys.argv
        try:
            sys.argv = command
            try:
                runpy.run_path(command[0], run_name="__main__")
                code = 0
            except SystemExit as error:
                code = error.code if isinstance(error.code, int) else 0 if error.code is None else 1
        finally:
            sys.argv = previous_argv
        record["worker_returncode"] = code
        result = args.output / "result.json"
        record["worker_receipt"] = SUITE.binding(result)
        worker = SUITE.read_json(result)
        SUITE.require(
            worker["engine"] == args.engine
            and worker["width"] == args.width
            and worker["pid"] == os.getpid()
            and "owned_runtime" not in worker,
            "Worker differs from the declared captured-runtime invocation",
        )
        SUITE.require(code in (0, 1), "Worker invocation failed")
        SUITE.require(worker["complete"] is True, "Numerical worker did not complete")
        SUITE.require(code == int(not worker["passed"]), "Worker exit status differs")
        record["complete"] = True
        record["passed"] = worker["passed"]
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        try:
            SUITE.check_controls()
            SUITE.verify_import_resolution(inventory, sys.path)
            record["origins_after"] = origins(args.engine, authority, backend, inventory)
            SUITE.pinned(args.capture, args.capture_sha256)
            SUITE.require(not Path(sys.pycache_prefix).exists(), "Bytecode cache was created")
            if "worker_receipt" in record:
                SUITE.binding(record["worker_receipt"]["path"], record["worker_receipt"])
            record["postflight_passed"] = True
        except BaseException:
            record["postflight_error"] = traceback.format_exc()
            record["postflight_passed"] = False
            record["complete"] = record["passed"] = False
        args.output.mkdir(parents=True, exist_ok=True)
        SUITE.write_json(args.output / "captured-runtime.json", record)
    return record


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "inventory", "evidence-root", "output", "raw-store"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--capture-sha256", required=True)
    result.add_argument("--inventory-sha256", required=True)
    result.add_argument("--engine", choices=("cpu_torch", "mps", "mlx"), required=True)
    result.add_argument("--width", type=int, choices=(512, 1024), required=True)
    result.add_argument("--batch-size", type=int, choices=(1, 16), default=16)
    result.add_argument("--gradients", action="store_true")
    result.add_argument("--python-candidate", action="store_true")
    result.add_argument("--framework-policy", type=int, choices=(7, 15))
    result.add_argument("--framework-compiled", action="store_true")
    result.add_argument("--case-offset", type=int, default=0)
    result.add_argument("--case-limit", type=int)
    return result


if __name__ == "__main__":
    answer = run(parser().parse_args())
    print(json.dumps({key: answer[key] for key in ("complete", "passed", "postflight_passed")}))
    raise SystemExit(0 if answer["passed"] else 1)
