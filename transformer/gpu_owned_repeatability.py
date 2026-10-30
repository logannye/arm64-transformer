"""Complete owned-runtime development repeatability, never formal admission.

Uses the unchanged retained diagnostic operands, comparisons, model and optimizer.
The worker body is explicit so its backend dependency is injected without changing
or monkeypatching the historical standard producer. All six widths, both engines,
two processes, ten repetitions and full/partial shapes are mandatory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from transformer.gpu_owned_study import FRAMEWORKS
from transformer.gpu_study import digest, identity, read_json, require, write_new
from transformer.gpu_study_context import OWNED, child_controls, python_command

WIDTHS = (32, 64, 128, 256, 512, 1024)
STAGES = ("embedding", "full_model", "training")
SCHEMA = "gpu-owned-repeatability-plan-v1"
LIMITS = {
    "output_bytes": 32 * 1024**3,
    "minimum_free_bytes": 5 * 1024**3,
    "minimum_start_free_bytes": 37 * 1024**3,
    "worker_seconds": 14400,
}


def check_storage(output, *, initial=False):
    path = Path(output)
    existing = path
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    floor = LIMITS["minimum_start_free_bytes"] if initial else LIMITS["minimum_free_bytes"]
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0
    require(free >= floor, "Owned repeatability disk reserve exhausted")
    require(size <= LIMITS["output_bytes"], "Owned repeatability output exceeded bound")
    return {"free_bytes": free, "output_bytes": size, "limits": dict(LIMITS)}


def bounded_worker(command, *, cwd, environment, log, output):
    result = {
        "command": command,
        "started_unix": time.time(),
        "returncode": None,
        "timed_out": False,
    }
    with Path(log).open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        result["pid"] = child.pid
        try:
            while child.poll() is None:
                check_storage(output)
                if time.time() - result["started_unix"] > LIMITS["worker_seconds"]:
                    result["timed_out"] = True
                    raise TimeoutError("Owned repeatability worker exceeded time limit")
                time.sleep(1)
        except BaseException:
            result["error"] = traceback.format_exc()
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        finally:
            result.update(returncode=child.returncode, ended_unix=time.time())
    return result


def validate_capabilities(worker, manifest, receipt):
    """Validate both actual loaded-origin snapshots using authenticated inventories."""
    from transformer.gpu_owned_backend import (
        validate_compilation_probe,
        validate_loaded_origin_proof,
    )

    binding, selected = manifest["bindings"], manifest["owned_frameworks"]
    engine = worker["engine"]
    require(engine in ("mps", "mlx"), "Unknown repeatability backend")
    framework = "pytorch" if engine == "mps" else "mlx"
    runtime_root = Path(worker["runtime_root"])
    interpreter = Path(binding["runtime_layout"]["interpreter"])
    require(
        runtime_root.is_absolute()
        and interpreter.is_relative_to(runtime_root)
        and interpreter.relative_to(runtime_root).as_posix() in binding["runtime"],
        "Repeatability runtime root differs",
    )
    for key in ("capabilities", "capabilities_after"):
        capabilities = worker.get(key, {})
        require(
            capabilities.get("baseline_class") == "project_maintained_full_source_build"
            and capabilities.get("standard_release_claim") is False
            and capabilities.get("numerical_admission_claim") is False
            and capabilities.get("runtime_identity_sha256") == identity(binding["runtime"])
            and capabilities.get("framework_identity") == selected["frameworks"][framework]
            and capabilities.get("runtime_receipt_sha256")
            == binding["baseline"][selected["runtime_receipt"]]
            and capabilities.get("storage_dtype") == "float32"
            and capabilities.get("cpu_tensor_fallback") is False,
            "Missing, stale, or incomplete owned capability snapshot",
        )
        if engine == "mlx":
            require(
                capabilities.get("compiled") is True
                and capabilities.get("MLX_DISABLE_COMPILE", "missing") is None,
                "MLX compile control changed",
            )
            validate_compilation_probe(capabilities["compilation_probe"])
        else:
            require(
                capabilities.get("deterministic_algorithms") is True, "MPS strict controls changed"
            )
        proof = capabilities["loaded_origin_proof"]
        origins = (
            proof.get("framework_module_origins", {}).get(framework, {}).get("module_origins", {})
        )
        require(
            proof.get("module_origins")
            == (
                [origins.get("torch"), origins.get("torch._C")]
                if engine == "mps"
                else [origins.get("mlx.core")]
            )
            and {
                "transformer",
                "transformer.spec",
                "transformer.reference",
                "transformer.gpu_owned_backend",
                f"transformer.gpu_{engine}",
            }
            <= set(proof.get("project_module_origins", {})),
            "Selected backend or numerical project module origins are absent",
        )
        validate_loaded_origin_proof(
            proof,
            inventory={"files": binding["runtime"], "layout": binding["runtime_layout"]},
            runtime_root=str(runtime_root),
            receipt=receipt,
        )
    return True


def source_names(engine):
    require(engine in ("mps", "mlx"), "Unknown owned repeatability engine")
    names = {
        "transformer/__init__.py",
        "transformer/gpu_environment.py",
        "transformer/gpu_repeatability.py",
        "transformer/gpu_owned_repeatability.py",
        "transformer/gpu_owned_backend.py",
        "transformer/gpu_owned_origins.py",
        "transformer/gpu_owned_study.py",
        "transformer/gpu_study.py",
        "transformer/gpu_study_runtime.py",
        "transformer/gpu_study_context.py",
        "transformer/reference.py",
        "transformer/spec.py",
        f"transformer/gpu_{engine}.py",
    }
    if engine == "mlx":
        names.add("transformer/gpu_mlx_compile.py")
    return sorted(names)


def source_identity(engine=None):
    root = Path(__file__).resolve().parents[1]
    names = (
        set(source_names("mps") + source_names("mlx")) if engine is None else source_names(engine)
    )
    return {name: digest(root / name) for name in sorted(names)}


def worker_inventory():
    return [
        {"engine": engine, "width": width, "process": process}
        for engine in ("mps", "mlx")
        for width in WIDTHS
        for process in range(2)
    ]


def validate_plan(plan):
    require(
        plan.get("schema") == SCHEMA
        and plan.get("protocol") == OWNED.protocol
        and plan.get("contract") == OWNED.repeatability_contract,
        "Wrong owned repeatability namespace",
    )
    require(
        plan.get("scope") == "development_only"
        and plan.get("seed") == 20261103
        and plan.get("formal_admission_opened") is False,
        "Formal data is forbidden here",
    )
    require(
        plan.get("workers") == worker_inventory()
        and plan.get("stages") == list(STAGES)
        and plan.get("repetitions") == 10
        and plan.get("processes") == 2,
        "Missing, duplicated, reordered or shortened diagnostic inventory",
    )
    require(plan.get("selection") == FRAMEWORKS, "Unselected framework candidate")
    require(plan.get("limits") == LIMITS, "Repeatability resource contract changed")
    require(plan.get("source_sha256") == source_identity(), "Repeatability source bytes changed")
    require(
        plan.get("runtime_identity_sha256") == identity(plan["runtime"]["files"]),
        "Runtime inventory identity mismatch",
    )
    require(
        set(plan)
        == {
            "schema",
            "protocol",
            "contract",
            "scope",
            "seed",
            "workers",
            "stages",
            "repetitions",
            "processes",
            "formal_admission_opened",
            "selection",
            "source_sha256",
            "runtime_root",
            "runtime",
            "runtime_identity_sha256",
            "runtime_receipt",
            "limits",
        },
        "Unexpected repeatability authority field",
    )
    return plan


def authenticate_plan(path, expected_sha256):
    from transformer.gpu_owned_backend import authenticate_runtime

    require(digest(path) == expected_sha256, "Repeatability plan trust anchor mismatch")
    plan = validate_plan(read_json(path))
    receipt = plan["runtime_receipt"]
    authority = authenticate_runtime(
        plan["runtime_root"], plan["runtime"], plan["selection"], receipt["path"], receipt["sha256"]
    )
    return plan, authority


def prepare(runtime_root, python_paths, runtime_receipt, runtime_receipt_sha256, output):
    from transformer.gpu_owned_backend import authenticate_runtime
    from transformer.gpu_study_runtime import discover_runtime

    runtime = discover_runtime(runtime_root, python_paths)
    authority = authenticate_runtime(
        runtime_root, runtime, FRAMEWORKS, runtime_receipt, runtime_receipt_sha256
    )
    plan = {
        "schema": SCHEMA,
        "protocol": OWNED.protocol,
        "contract": OWNED.repeatability_contract,
        "scope": "development_only",
        "seed": 20261103,
        "workers": worker_inventory(),
        "stages": list(STAGES),
        "repetitions": 10,
        "processes": 2,
        "formal_admission_opened": False,
        "selection": FRAMEWORKS,
        "source_sha256": source_identity(),
        "runtime_root": str(Path(runtime_root).resolve()),
        "runtime": runtime,
        "runtime_identity_sha256": identity(runtime["files"]),
        "runtime_receipt": {"path": authority.receipt_path, "sha256": authority.receipt_sha256},
        "limits": dict(LIMITS),
    }
    validate_plan(plan)
    write_new(output, plan)
    return plan


def run_worker(engine, width, destination, *, authority):
    """The retained complete diagnostic, using authenticated unchanged backends."""
    authority.verify()
    check_storage(Path(destination).parent)
    import numpy as np

    from transformer.gpu_environment import require_environment
    from transformer.gpu_owned_backend import create_runtime_backend
    from transformer.gpu_repeatability import (
        BATCHES,
        LENGTHS,
        PINS,
        REPETITIONS,
        SEED,
        _numpy_forward,
        _save_arrays,
        _torch_reference,
        _write_json,
        accuracy,
        array_identity,
        embedding_cases,
        embedding_operands,
        embedding_oracle,
        exact_comparison,
        expected_case_names,
        make_embedding_runner,
        repeat_case,
        tokens_for,
    )

    stages = list(STAGES)
    authority.verify()
    runtime_identity_sha256 = identity(json.loads(authority.inventory_json)["files"])
    require_environment(PINS)
    from transformer import spec

    if width != spec.WIDTH:
        raise ValueError("Width must be selected before Python starts")
    import torch

    torch.set_num_threads(1)
    from transformer.reference import Transformer

    initial = Transformer(seed=SEED).flat_parameters()
    backend = create_runtime_backend(engine, initial, authority=authority)
    report = {
        "contract": OWNED.repeatability_contract,
        "protocol": OWNED.protocol,
        "runtime_receipt_sha256": authority.receipt_sha256,
        "scope": "development_only",
        "engine": engine,
        "width": width,
        "stages": stages,
        "repetitions": REPETITIONS,
        "initial": array_identity(initial),
        "source_sha256": source_identity(engine),
        "runtime_identity_sha256": runtime_identity_sha256,
        "runtime_root": str(authority.root),
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {key: os.environ.get(key) for key in PINS},
        "capabilities": backend.capabilities(),
        "cases": [],
        "unsupported_controls": [],
        "cleanup_errors": [],
        "expected_cases": expected_case_names(width, stages),
    }
    initial_state = backend.export_state()
    _save_arrays(destination / "initial-state.npz", initial_state)

    def retain(result):
        nonlocal backend
        report["cases"].append(result)
        if not result["complete"]:
            recovery = {"attempted": True, "fresh_backend": False}
            try:
                backend.close()
            except Exception as error:
                recovery["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
                report["cleanup_errors"].append(
                    {"case": result["name"], **recovery["cleanup_error"]}
                )
            result["recovery"] = recovery
            # A rejected lazy graph can poison its backend's pending queue. A
            # fresh ordinary backend isolates the next declared case; its failed
            # predecessor and skipped repetitions remain explicit failures.
            backend = None
            backend = create_runtime_backend(engine, initial, authority=authority)
            backend.load_state(initial_state)
            recovery["fresh_backend"] = True
            _write_json(destination / result["name"] / "result.json", result)
        _write_json(destination / "worker.json", report)
        check_storage(Path(destination).parent)
        print(
            json.dumps(
                {
                    "case": result["name"],
                    "exact": result["exact_passed"],
                    "accuracy": result["accuracy_passed"],
                    "complete": result["complete"],
                }
            ),
            flush=True,
        )

    try:
        if "embedding" in stages:
            if engine == "metal":
                raise ValueError(
                    "Metal has no isolated embedding probe; select full_model/training explicitly"
                )
            for case in embedding_cases(width):
                if not case["supported"]:
                    report["unsupported_controls"].append(
                        {**case, "reason": "B*T exceeds vocabulary 128"}
                    )
                    continue
                tokens, table, dy = embedding_operands(width, case)
                gold = embedding_oracle(tokens, table, dy)
                weight = torch.tensor(table, dtype=torch.float64, requires_grad=True)
                output = torch.nn.functional.embedding(
                    torch.from_numpy(tokens.astype(np.int64)), weight
                )
                (output * torch.tensor(dy, dtype=torch.float64)).sum().backward()
                independent = accuracy(weight.grad.numpy(), gold["gradient"], "oracle")

                def oracle(actual, expected=gold, independent=independent):
                    return {
                        "output": accuracy(actual["output"], expected["output"], "forward"),
                        "gradient": accuracy(actual["gradient"], expected["gradient"], "gradient"),
                        "independent_embedding_oracle": independent,
                    }

                # Construction itself may reject a supported framework shape.
                operation = None

                def embedding_run():
                    nonlocal operation
                    if operation is None:
                        operation = make_embedding_runner(engine, tokens, table, dy)
                    return operation()

                result = repeat_case(case["name"], embedding_run, destination, oracle)
                result["fixture"] = {
                    "tokens": array_identity(tokens),
                    "table": array_identity(table),
                    "cotangent": array_identity(dy),
                }
                _save_arrays(
                    destination / case["name"] / "fixture.npz",
                    {"tokens": tokens, "table": table, "cotangent": dy},
                )
                retain(result)
        for batch in BATCHES:
            for length in LENGTHS:
                tokens = tokens_for(batch, length, "skewed")
                targets = np.roll(tokens, -1, axis=1).copy()

                def restore():
                    backend.load_state(initial_state)
                    restored = backend.export_state()
                    if not all(
                        exact_comparison(restored[key], initial_state[key])["passed"]
                        for key in initial_state
                    ):
                        raise RuntimeError("Exact initial-state restoration failed")

                if "full_model" in stages:
                    gold = _torch_reference(initial, tokens, targets, backward=True)
                    oracle_check = accuracy(
                        gold["logits"], _numpy_forward(initial, tokens), "oracle"
                    )

                    def full_model():
                        restore()
                        device_batch = backend.upload_inputs(tokens, targets)
                        logits = backend.to_numpy(backend.forward(device_batch)).copy()
                        loss, gradient = backend.loss_and_grad(device_batch)
                        result = {
                            "logits": logits,
                            "loss": backend.to_numpy(loss),
                            "gradient": backend.to_numpy(gradient),
                        }
                        restored = backend.export_state()
                        if not all(
                            exact_comparison(restored[key], initial_state[key])["passed"]
                            for key in initial_state
                        ):
                            raise RuntimeError(
                                "Forward/backward changed parameters or optimizer state"
                            )
                        return result

                    def full_oracle(actual):
                        return {
                            **{
                                key: accuracy(
                                    actual[key],
                                    gold[key],
                                    "gradient" if key == "gradient" else "forward",
                                )
                                for key in gold
                            },
                            "independent_forward_oracle": oracle_check,
                        }

                    result = repeat_case(
                        f"full_model/b{batch}-t{length}", full_model, destination, full_oracle
                    )
                    result["fixture"] = {
                        "tokens": array_identity(tokens),
                        "targets": array_identity(targets),
                    }
                    retain(result)
                if "training" in stages:

                    def training():
                        restore()
                        device_batch = backend.upload_inputs(tokens, targets)
                        losses = []
                        for _ in range(6):
                            losses.append(float(backend.to_numpy(backend.train_step(device_batch))))
                            if not backend.check_finite():
                                raise RuntimeError("Nonfinite state during six-step training")
                        state = backend.export_state()
                        logits = backend.to_numpy(backend.forward(device_batch)).copy()
                        # Exercise exact final-checkpoint restoration and replay in each execution.
                        backend.load_state(state)
                        replay = backend.to_numpy(backend.forward(device_batch)).copy()
                        if not exact_comparison(replay, logits)["passed"]:
                            raise RuntimeError("Final checkpoint forward replay changed")
                        return {**state, "logits": logits, "losses": np.asarray(losses, np.float32)}

                    def checkpoint_oracle(actual):
                        expected = _torch_reference(
                            actual["parameters"], tokens, targets, backward=False
                        )
                        numpy_logits = _numpy_forward(actual["parameters"], tokens)
                        return {
                            "checkpoint_logits": accuracy(
                                actual["logits"], numpy_logits, "forward"
                            ),
                            "independent_forward_oracle": accuracy(
                                expected["logits"], numpy_logits, "oracle"
                            ),
                        }

                    result = repeat_case(
                        f"training/b{batch}-t{length}", training, destination, checkpoint_oracle
                    )
                    result["fixture"] = {
                        "tokens": array_identity(tokens),
                        "targets": array_identity(targets),
                    }
                    retain(result)
    except Exception as exc:
        report["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if backend is not None:
            try:
                report["capabilities_after"] = backend.capabilities()
                authority.verify()
                report["storage_after"] = check_storage(Path(destination).parent)
            except Exception as error:
                report["cleanup_errors"].append(
                    {
                        "case": "identity_postflight",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
            try:
                backend.close()
            except Exception as error:
                report["cleanup_errors"].append(
                    {
                        "case": "worker_finalization",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
        report["source_sha256_after"] = source_identity(engine)
        report["source_unchanged"] = report["source_sha256"] == report["source_sha256_after"]
        report["inventory_complete"] = {case["name"] for case in report["cases"]} == set(
            report["expected_cases"]
        )
        report["complete"] = (
            report["inventory_complete"]
            and "fatal_error" not in report
            and not report["cleanup_errors"]
            and all(case["complete"] for case in report["cases"])
        )
        report["exact_passed"] = (
            report["complete"]
            and report["source_unchanged"]
            and all(case["exact_passed"] for case in report["cases"])
        )
        report["accuracy_passed"] = report["complete"] and all(
            case["accuracy_passed"] for case in report["cases"]
        )
        _write_json(destination / "worker.json", report)
    return report


def run(plan_path, plan_sha256, output):
    from transformer.gpu_environment import ENVIRONMENT, child_environment

    plan, authority = authenticate_plan(plan_path, plan_sha256)
    from transformer.gpu_repeatability import _write_json, compare_processes

    output = Path(output).resolve()
    storage = check_storage(output, initial=True)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    summary = {
        "contract": OWNED.repeatability_contract,
        "protocol": OWNED.protocol,
        "scope": "development_only",
        "engines": ["mps", "mlx"],
        "widths": list(WIDTHS),
        "stages": list(STAGES),
        "workers": [],
        "cross_process": [],
        "source_sha256": source_identity(),
        "runtime_identity_sha256": plan["runtime_identity_sha256"],
        "runtime_receipt_sha256": authority.receipt_sha256,
        "plan_sha256": plan_sha256,
        "full_declared_inventory": True,
        "formal_admission_opened": False,
        "complete": False,
        "exact_passed": False,
        "accuracy_passed": False,
        "storage_before": storage,
        "limits": dict(LIMITS),
    }
    paths = [str(root)] + [
        str(Path(plan["runtime_root"]) / name) for name in plan["runtime"]["python_paths"]
    ]
    try:
        for row in plan["workers"]:
            authenticate_plan(plan_path, plan_sha256)
            check_storage(output)
            engine, width, process = row["engine"], row["width"], row["process"]
            name = f"{engine}-w{width}-process{process}"
            destination = output / name
            env = child_controls(
                {"protocol": OWNED.protocol},
                child_environment(
                    ENVIRONMENT,
                    ARM64_TRANSFORMER_WIDTH=str(width),
                    PYTHONPATH=os.pathsep.join(paths),
                ),
            )
            launch_manifest = {
                "protocol": OWNED.protocol,
                "bindings": {
                    "source": plan["source_sha256"],
                    "runtime": plan["runtime"]["files"],
                    "runtime_python_paths": plan["runtime"]["python_paths"],
                },
            }
            command = python_command(
                launch_manifest,
                destination,
                roots={"source": root, "runtime": plan["runtime_root"]},
            ) + [
                "-m",
                "transformer.gpu_owned_repeatability",
                "worker",
                "--plan",
                str(Path(plan_path).resolve()),
                "--plan-sha256",
                plan_sha256,
                "--worker-index",
                str(plan["workers"].index(row)),
                "--output",
                str(destination),
            ]
            execution = bounded_worker(
                command, cwd=root, environment=env, log=output / (name + ".log"), output=output
            )
            result = destination / "worker.json"
            record = {
                **row,
                "returncode": execution["returncode"],
                "execution": execution,
                "result": name + "/worker.json",
                "command": command,
                "complete": False,
                "exact_passed": False,
                "accuracy_passed": False,
            }
            if result.is_file():
                value = read_json(result)
                record.update(
                    {
                        key: value.get(key) is True
                        for key in ("complete", "exact_passed", "accuracy_passed")
                    }
                )
            record["artifact_sha256"] = (
                {
                    path.relative_to(destination).as_posix(): digest(path)
                    for path in destination.rglob("*")
                    if path.is_file()
                }
                if destination.is_dir()
                else {}
            )
            record["log_sha256"] = digest(output / (name + ".log"))
            summary["workers"].append(record)
            _write_json(output / "summary.json", summary)
            require(
                "error" not in execution and not execution["timed_out"],
                "Owned repeatability worker stopped; partial evidence retained",
            )
            if process == 1 and all(
                (output / f"{engine}-w{width}-process{index}" / "worker.json").is_file()
                for index in range(2)
            ):
                compared = compare_processes(
                    output / f"{engine}-w{width}-process0",
                    destination,
                    output / f"{engine}-w{width}-cross-process.json",
                )
                summary["cross_process"].append(
                    {"engine": engine, "width": width, "passed": compared["passed"]}
                )
        from transformer.gpu_study import verify_files

        for row in summary["workers"]:
            directory = output / Path(row["result"]).parent
            require(
                {
                    path.relative_to(directory).as_posix()
                    for path in directory.rglob("*")
                    if path.is_file()
                }
                == set(row["artifact_sha256"]),
                "Retained worker evidence inventory changed",
            )
            verify_files(directory, row["artifact_sha256"])
            require(
                digest(output / (directory.name + ".log")) == row["log_sha256"],
                "Retained worker log changed",
            )
        summary["complete"] = (
            len(summary["workers"]) == 24
            and len(summary["cross_process"]) == 12
            and all(row["complete"] and row["returncode"] == 0 for row in summary["workers"])
        )
        summary["exact_passed"] = (
            summary["complete"]
            and all(row["exact_passed"] for row in summary["workers"])
            and all(row["passed"] for row in summary["cross_process"])
        )
        summary["accuracy_passed"] = summary["complete"] and all(
            row["accuracy_passed"] for row in summary["workers"]
        )
        authenticate_plan(plan_path, plan_sha256)
        summary["storage_after"] = check_storage(output)
    except BaseException:
        summary.update(complete=False, exact_passed=False, error=traceback.format_exc())
        raise
    finally:
        _write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker"))
    parser.add_argument("--runtime-root")
    parser.add_argument("--python-path", action="append", default=[])
    parser.add_argument("--runtime-receipt")
    parser.add_argument("--runtime-receipt-sha256")
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        require(
            all((args.runtime_root, args.runtime_receipt, args.runtime_receipt_sha256)),
            "Prepare requires full runtime and external receipt identity",
        )
        prepare(
            args.runtime_root,
            args.python_path,
            args.runtime_receipt,
            args.runtime_receipt_sha256,
            args.output,
        )
        return 0
    require(
        args.plan is not None and args.plan_sha256 is not None, "External plan identity required"
    )
    if args.action == "worker":
        plan, authority = authenticate_plan(args.plan, args.plan_sha256)
        require(type(args.worker_index) is int and 0 <= args.worker_index < 24, "Unknown worker")
        row = plan["workers"][args.worker_index]
        require(
            os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(row["width"]), "Worker width differs"
        )
        args.output.mkdir(parents=True, exist_ok=False)
        result = run_worker(row["engine"], row["width"], args.output, authority=authority)
        authenticate_plan(args.plan, args.plan_sha256)
    else:
        result = run(args.plan, args.plan_sha256, args.output)
    return int(not (result["complete"] and result["exact_passed"] and result["accuracy_passed"]))


if __name__ == "__main__":
    raise SystemExit(main())
