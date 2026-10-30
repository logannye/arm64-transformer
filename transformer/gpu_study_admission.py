"""Exact admission inventory and independently evaluated raw-array receipts.

This module does not promote development results to admission. Producer coverage
is checked without opening fixtures. The run command requires a frozen contract,
current authenticated baseline repeatability and complete producer coverage before
opening seed 20261104. Every numerical receipt is re-evaluated from raw arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from transformer.gpu_environment import child_environment, require_environment
from transformer.gpu_study import (
    ENGINES,
    ENVIRONMENT,
    WIDTHS,
    digest,
    disk_preflight,
    identity,
    policy_path,
    python_path,
    read_json,
    regular,
    require,
    require_separate_output,
    validate_manifest,
    verify_bindings,
    write_new,
)

TOLERANCES = {
    "forward": (2e-5, 2e-4),
    "gradient": (3e-5, 3e-3),
    "math": (2e-7, 2e-6),
    "oracle": (1e-10, 1e-10),
    "optimizer": (3e-6, 3e-4),
    "trajectory": (2e-4, 2e-3),
    "exact": (0, 0),
}
ATTENTION = {
    "output": "forward",
    "probabilities": "math",
    "dqkv": "gradient",
    "same_saved_probabilities_gradient": "gradient",
    "probability_mass": "math",
}
NORM = {
    "normalized": "forward",
    "inverse_std": "math",
    "ln": "forward",
    "qkv": "forward",
    "dx": "gradient",
    "dgamma": "gradient",
    "dbeta": "gradient",
    "dweight": "gradient",
    "dbias": "gradient",
    "dln": "gradient",
    "same_operands_output": "forward",
    "same_operands_dln": "gradient",
    "same_operands_dweight": "gradient",
    "same_operands_dbias": "gradient",
    "same_incoming_dx": "gradient",
    "same_incoming_dgamma": "gradient",
    "same_incoming_dbeta": "gradient",
}
PARAMETERS = (
    "token",
    "position",
    "ln1_weight",
    "ln1_bias",
    "qkv_weight",
    "qkv_bias",
    "attention_weight",
    "attention_bias",
    "ln2_weight",
    "ln2_bias",
    "ff1_weight",
    "ff1_bias",
    "ff2_weight",
    "ff2_bias",
    "ln3_weight",
    "ln3_bias",
    "head_weight",
    "head_bias",
)
CONTROLS = (
    "causal_prefix",
    "sequence_isolation",
    "repeat_embedding_gradient",
    "repeat_forward",
    "repeat_training_same_process",
    "repeat_training_fresh_process",
    "warmup_restore",
    "partial_batch",
    "tail_bounds",
    "input_immutability",
    "checkpoint_first_tie",
    "checkpoint_replay",
    "finite_state",
    "malformed_input",
    "output_collision",
    "pending_work_cleanup",
    "device_execution",
    "state_restore",
    "runtime_failure",
)


def gate_inventory():
    from transformer.gpu_precision import attention_case_specs, layernorm_qkv_case_specs

    result = []

    def gate(name, members, *, blocking=True, family="runtime"):
        result.append({"id": name, "family": family, "blocking": blocking, "members": members})

    for width in WIDTHS:
        for specification in attention_case_specs(width):
            gate(f"metal/{width}/attention/{specification[0]}", dict(ATTENTION), family="precision")
        for specification in layernorm_qkv_case_specs(width):
            gate(f"metal/{width}/layernorm_qkv/{specification[0]}", dict(NORM), family="precision")
        for engine in ("mps", "mlx"):
            for specification in attention_case_specs(width):
                gate(
                    f"{engine}/{width}/framework_attention/{specification[0]}",
                    {"output": "forward", "dqkv": "gradient"},
                    blocking=False,
                    family="diagnostic_precision",
                )
            for specification in layernorm_qkv_case_specs(width):
                gate(
                    f"{engine}/{width}/framework_layernorm_qkv/{specification[0]}",
                    {
                        key: "forward" if key in ("ln", "qkv") else "gradient"
                        for key in ("ln", "qkv", "dx", "dgamma", "dbeta", "dweight", "dbias")
                    },
                    blocking=False,
                    family="diagnostic_precision",
                )
        for engine in ENGINES:
            for batch in (1, 3, 16):
                gate(
                    f"{engine}/{width}/full/B{batch}T32",
                    {
                        "logits": "forward",
                        "loss": "forward",
                        **{f"gradient/{p}": "gradient" for p in PARAMETERS},
                    },
                    blocking=engine == "metal",
                    family="accuracy",
                )
            for control in CONTROLS:
                gate(f"{engine}/{width}/control/{control}", {"observations": "exact"})
            for state in ("initial", "evolved"):
                gate(
                    f"{engine}/{width}/common_state/{state}",
                    {"logits": "forward", "gradient": "gradient"},
                )
            for source in ENGINES:
                for step in (1, 2, 10):
                    gate(
                        f"{engine}/{width}/adam/from-{source}/step-{step}",
                        {
                            "parameters": "optimizer",
                            "m": "optimizer",
                            "v": "optimizer",
                            "step": "exact",
                        },
                    )
            gate(
                f"{engine}/{width}/trajectory",
                {name: "trajectory" for name in ("parameters", "loss", "gradient", "m", "v")},
                blocking=width <= 256,
                family="trajectory",
            )
        gate(
            f"oracle/{width}/independent_fp64",
            {"logits": "oracle", "gradient": "oracle"},
            family="oracle",
        )
        for name in ("forward", "dx", "dw", "db", "embedding", "relu", "cross_entropy"):
            gate(
                f"metal/{width}/primitive/{name}",
                {"values": "gradient" if name in ("dx", "dw", "db", "embedding") else "forward"},
            )
        for control in (
            "guards",
            "barriers",
            "borrowed_result_lifetime",
            "borrowed_batch_lifetime",
            "finite_flag",
        ):
            gate(f"metal/{width}/control/{control}", {"observations": "exact"})
    require(len({g["id"] for g in result}) == len(result), "Duplicate admission gate")
    require(sum(g["family"] == "precision" for g in result) == 270, "Wrong precision case count")
    return result


def execution_readiness():
    """Read producer coverage without generating fixtures or initializing devices."""
    try:
        from transformer.gpu_admission_checks import producer_inventory

        handled = set(producer_inventory())
    except ImportError:
        handled = set()
    required = {g["id"] for g in gate_inventory()}
    missing = sorted(required - handled)
    return {
        "ready": not missing,
        "admission_seed_opened": False,
        "reason": "Missing prospective execution handlers"
        if missing
        else "All producers registered; admission still requires frozen bindings",
        "missing_handlers": missing,
        "inventory_gates": len(gate_inventory()),
        "precision_cases": 270,
    }


def binding_identity(manifest):
    return identity(
        {
            key: manifest[key]
            for key in (
                "protocol",
                "workload",
                "timing",
                "environment",
                "cells",
                "bindings",
                "gate_inventory_sha256",
                "admission_seed",
            )
        }
    )


def verify_baseline_repeatability(manifest, roots):
    """Authenticate full baseline inventory and recompute exact equality from proofs."""
    import numpy as np

    from transformer.gpu_repeatability import (
        CONTRACT,
        PINS,
        PROCESSES,
        REPETITIONS,
        STAGES,
        array_identity,
        expected_case_names,
    )

    root = Path(roots["baseline"])
    records = manifest["bindings"]["baseline"]
    require("summary.json" in records, "Missing baseline summary binding")
    from transformer.gpu_study import verify_files

    verify_files(root, records)
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(actual_files == set(records), "Baseline evidence inventory is incomplete or changed")
    summary = read_json(root / "summary.json")
    runtime_hash = identity(manifest["bindings"]["runtime"])
    require(
        summary.get("contract") == CONTRACT
        and summary.get("scope") == "development_only"
        and summary.get("engines") == ["mps", "mlx"]
        and summary.get("widths") == list(WIDTHS)
        and summary.get("stages") == list(STAGES)
        and summary.get("full_declared_inventory") is True,
        "Incomplete baseline repetition protocol",
    )
    require(
        summary.get("runtime_identity_sha256") == runtime_hash,
        "Baseline runtime identity is absent or stale",
    )
    require(
        summary.get("complete") is True and summary.get("exact_passed") is True,
        "Standard baseline repeatability still fails; reserved admission stays closed",
    )
    expected_workers = [
        (engine, width, process)
        for engine in ("mps", "mlx")
        for width in WIDTHS
        for process in range(PROCESSES)
    ]
    workers = summary.get("workers", [])
    require(
        [(w.get("engine"), w.get("width"), w.get("process")) for w in workers] == expected_workers,
        "Missing, duplicate, or reordered baseline worker",
    )
    retained = {}
    for entry in workers:
        engine, width, process = entry["engine"], entry["width"], entry["process"]
        worker_path = f"{engine}-w{width}-process{process}/worker.json"
        require(
            entry.get("returncode") == 0 and entry.get("result") == worker_path,
            "Failed or redirected baseline worker",
        )
        worker = read_json(regular(root, worker_path))
        require(
            worker.get("runtime_identity_sha256") == runtime_hash
            and worker.get("environment") == PINS,
            "Baseline worker runtime/environment differs",
        )
        source = worker.get("source_sha256", {})
        require(
            bool(source)
            and worker.get("source_sha256_after") == source
            and all(
                manifest["bindings"]["source"].get(name) == sha for name, sha in source.items()
            ),
            "Baseline source bytes are stale",
        )
        required_sources = {
            "transformer/gpu_repeatability.py",
            "transformer/gpu_environment.py",
            "scripts/check_gpu_repeatability.py",
            "transformer/reference.py",
            "transformer/spec.py",
            f"transformer/gpu_{engine}.py",
        }
        if engine == "mlx":
            required_sources.add("transformer/gpu_mlx_compile.py")
            capabilities = worker.get("capabilities", {})
            probe = capabilities.get("compilation_probe", {})
            require(
                capabilities.get("compiled") is True
                and capabilities.get("MLX_DISABLE_COMPILE", "missing") is None
                and probe.get("status") == "passed"
                and probe.get("mode_override_used") is False
                and probe.get("python_trace_count") == 1
                and probe.get("executions") == 4
                and probe.get("compiled_primitive_present") is True,
                "Baseline has no passing MLX compilation execution proof",
            )
        require(set(source) == required_sources, "Incomplete baseline executed source identity")
        expected = expected_case_names(width, STAGES)
        cases = worker.get("cases", [])
        require(
            len(cases) == len(expected)
            and {c.get("name") for c in cases} == set(expected)
            and worker.get("expected_cases") == expected,
            "Baseline worker omitted cases",
        )
        for case in cases:
            require(case.get("complete") is True, "Incomplete baseline case")
            first = regular(root / Path(worker_path).parent, case["first_arrays"])
            with np.load(first, allow_pickle=False) as arrays:
                values = {name: arrays[name] for name in arrays.files}
            require(
                bool(values) and all(np.isfinite(a).all() for a in values.values()),
                "Invalid first baseline output",
            )
            fingerprints = {name: array_identity(array) for name, array in values.items()}
            executions = case.get("executions", [])
            require(
                [x.get("execution") for x in executions] == list(range(REPETITIONS)),
                "Missing baseline repetitions",
            )
            for execution in executions:
                require(
                    execution.get("finite") is True
                    and set(execution.get("exact", {})) == set(fingerprints),
                    "Incomplete repetition fingerprints",
                )
                for name, fingerprint in fingerprints.items():
                    proof = execution["exact"][name]
                    require(
                        proof.get("actual") == proof.get("expected") == fingerprint
                        and proof.get("different_coordinates") == 0
                        and proof.get("passed") is True,
                        "Baseline exact repetition failed",
                    )
            key = (engine, width, case["name"])
            if process == 0:
                retained[key] = fingerprints
            else:
                require(retained[key] == fingerprints, "Baseline fresh-process repetition failed")
    return True


def array_artifact(root, record):
    import numpy as np

    require(
        isinstance(record, dict) and set(record) == {"path", "sha256"},
        "Array evidence must specify path and digest",
    )
    path = regular(root, record["path"])
    require(digest(path) == record["sha256"], "Array evidence changed")
    with path.open("rb") as handle:
        array = np.load(handle, allow_pickle=False)
        require(not handle.read(1), "Trailing bytes after array evidence")
    require(
        array.dtype.kind in "biuf" and array.size > 0, "Evidence must be a nonempty numeric array"
    )
    require(np.isfinite(array).all(), "Nonfinite admission evidence")
    return array


def evaluate_member(root, record, tolerance):
    import numpy as np

    require(
        isinstance(record, dict) and set(record) == {"actual", "expected"},
        "Missing raw comparison evidence",
    )
    actual, expected = (array_artifact(root, record[key]) for key in ("actual", "expected"))
    require(actual.shape == expected.shape, "Admission array shapes differ")
    if tolerance == "exact":
        passed = actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    else:
        require(
            actual.dtype.kind == expected.dtype.kind == "f",
            "Numerical evidence must be floating point",
        )
        atol, rtol = TOLERANCES[tolerance]
        passed = bool(
            np.all(
                np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                <= atol + rtol * np.abs(expected.astype(np.float64))
            )
        )
    return bool(passed)


def evaluate_gate(root, specification, record, manifest):
    require(record.get("id") == specification["id"], "Gate identity mismatch")
    if record.get("status") == "runtime_error":
        require(
            specification["family"] == "diagnostic_precision"
            and specification["blocking"] is False,
            "Runtime failures cannot waive blocking admission",
        )
        require(
            record.get("members") == {} and record.get("passed") is False,
            "A failed diagnostic cannot claim numerical outputs",
        )
        proof = record.get("error_artifact", {})
        require(set(proof) == {"path", "sha256"}, "Missing authenticated runtime error")
        path = regular(root, proof["path"])
        require(digest(path) == proof["sha256"], "Runtime error evidence changed")
        error = read_json(path)
        require(
            error.get("gate") == specification["id"]
            and error.get("binding_sha256") == binding_identity(manifest)
            and isinstance(error.get("exception_type"), str)
            and bool(error["exception_type"])
            and isinstance(error.get("message"), str)
            and bool(error["message"]),
            "Incomplete runtime error evidence",
        )
        return False
    require(
        set(record.get("members", {})) == set(specification["members"]),
        "Missing named check member",
    )
    outcomes = [
        evaluate_member(root, record["members"][name], tolerance)
        for name, tolerance in specification["members"].items()
    ]
    return all(outcomes)


def validate_receipt(manifest, receipt, root):
    """Re-evaluate all authenticated arrays; ignore claimed passed booleans."""
    require(
        receipt.get("schema") == 1 and receipt.get("scope") == "formal_admission",
        "Not a formal admission receipt",
    )
    require(receipt.get("seed") == 20261104, "Wrong admission stream")
    require(
        receipt.get("binding_sha256") == binding_identity(manifest),
        "Admission source/runtime/input/policy binding is stale",
    )
    require(
        receipt.get("inventory_sha256") == manifest["gate_inventory_sha256"],
        "Admission inventory changed",
    )
    require(
        receipt.get("producers_complete") is True and receipt.get("missing_handlers") == [],
        "Admission execution is incomplete",
    )
    records = receipt.get("gates")
    require(isinstance(records, list), "Missing gate evidence")
    require(
        [g.get("id") for g in records] == [g["id"] for g in manifest["gate_inventory"]],
        "Missing, duplicate, or reordered gate",
    )
    failures = []
    for specification, record in zip(manifest["gate_inventory"], records, strict=True):
        passed = evaluate_gate(root, specification, record, manifest)
        require(record.get("passed") is passed, "Claimed gate status disagrees with raw evidence")
        if specification["blocking"] and not passed:
            failures.append(record["id"])
    require(not failures, f"Blocking admission failures: {failures[:8]}")
    require(receipt.get("status") == "passed", "Admission receipt is not complete")
    return True


def run_admission(manifest_path, expected_hash, roots, output):
    """Open the reserved stream only after source freeze and full handler coverage."""
    require(digest(manifest_path) == expected_hash, "Manifest trust anchor mismatch")
    manifest = validate_manifest(read_json(manifest_path))
    require(
        manifest["phase"] == "frozen" and manifest["admission"] is None,
        "Admission needs a frozen pre-admission manifest, without a recycled receipt",
    )
    verify_bindings(manifest, roots)
    verify_baseline_repeatability(manifest, roots)
    readiness = execution_readiness()
    require(
        readiness["ready"],
        f"Admission remains unopened: {len(readiness['missing_handlers'])} "
        "missing execution handlers",
    )
    output = require_separate_output(output, roots)
    disk = disk_preflight(output, "admission")
    output.mkdir(parents=True)
    result = {
        "schema": 1,
        "scope": "formal_admission",
        "seed": 20261104,
        "binding_sha256": binding_identity(manifest),
        "inventory_sha256": manifest["gate_inventory_sha256"],
        "producers_complete": True,
        "missing_handlers": [],
        "gates": [],
        "status": "running",
        "disk_preflight": disk,
    }
    try:
        for width in WIDTHS:
            verify_bindings(manifest, roots)
            env = child_environment(ENVIRONMENT, ARM64_TRANSFORMER_WIDTH=str(width))
            env["PYTHONPATH"] = python_path(manifest, roots)
            env["ARM64_TRANSFORMER_METAL_POLICY"] = str(policy_path(manifest, roots))
            command = [
                sys.executable,
                "-m",
                "transformer.gpu_study_admission",
                "worker",
                "--manifest",
                str(Path(manifest_path).resolve()),
                "--manifest-sha256",
                expected_hash,
                "--roots",
                json.dumps({k: str(Path(v).resolve()) for k, v in roots.items()}),
                "--output",
                str(output.resolve()),
                "--width",
                str(width),
            ]
            process = subprocess.run(
                command,
                cwd=roots["source"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=14400,
            )
            write_new(
                output / f"worker-width-{width}.json",
                {
                    "command": command,
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                },
            )
            require(
                process.returncode == 0, f"Admission worker failed at width {width}; logs retained"
            )
            records = read_json(output / f"width-{width}.json")
            expected = [
                g for g in manifest["gate_inventory"] if g["id"].split("/")[1] == str(width)
            ]
            require(
                [r.get("id") for r in records] == [g["id"] for g in expected],
                "Incomplete admission worker",
            )
            result["gates"].extend(records)
            write_new(output / f"progress-width-{width}.json", result)
        result["status"] = (
            "passed"
            if all(
                not g["blocking"] or r["passed"]
                for g, r in zip(manifest["gate_inventory"], result["gates"], strict=True)
            )
            else "failed"
        )
        verify_bindings(manifest, roots)
    except Exception as exc:
        result.update(status="execution_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_new(output / "admission.json", result)
    return result


def produce_width(manifest, roots, output, width):
    require_environment(ENVIRONMENT)
    from transformer.gpu_admission_checks import close_contexts, produce_gate

    require(
        os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(width),
        "Admission width was not set before imports",
    )
    verify_bindings(manifest, roots)
    verify_baseline_repeatability(manifest, roots)
    records = []
    try:
        for specification in manifest["gate_inventory"]:
            if specification["id"].split("/")[1] != str(width):
                continue
            record = produce_gate(specification, manifest, roots, output)
            record["passed"] = evaluate_gate(output, specification, record, manifest)
            records.append(record)
            write_new(Path(output) / f"gate-{width}-{len(records):04d}.json", record)
    finally:
        close_contexts()
    verify_bindings(manifest, roots)
    write_new(Path(output) / f"width-{width}.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "run", "seal", "worker"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--roots")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, choices=WIDTHS)
    args = parser.parse_args()
    if args.action == "inventory":
        print(json.dumps(execution_readiness(), indent=2))
        return
    require(
        all((args.manifest, args.manifest_sha256, args.roots, args.output)),
        "Manifest/hash/roots/output required",
    )
    require(digest(args.manifest) == args.manifest_sha256, "Manifest trust anchor mismatch")
    roots = json.loads(args.roots)
    if args.action == "worker":
        require(args.width is not None, "Worker width required")
        manifest = validate_manifest(read_json(args.manifest))
        require(
            manifest["phase"] == "frozen" and manifest["admission"] is None,
            "Admission worker requires a fresh frozen contract",
        )
        require(execution_readiness()["ready"], "Missing admission execution handlers")
        produce_width(manifest, roots, args.output, args.width)
    elif args.action == "run":
        result = run_admission(args.manifest, args.manifest_sha256, roots, args.output)
        if result["status"] != "passed":
            raise SystemExit(1)
    else:
        require(args.receipt is not None, "Receipt is required to seal measurement manifest")
        manifest = validate_manifest(read_json(args.manifest))
        require(
            manifest["phase"] == "frozen" and manifest["admission"] is None,
            "Only a fresh frozen manifest can be sealed",
        )
        verify_bindings(manifest, roots)
        verify_baseline_repeatability(manifest, roots)
        validate_receipt(manifest, read_json(args.receipt), args.receipt.parent)
        manifest["admission"] = {"sha256": digest(args.receipt)}
        write_new(args.output, manifest)


if __name__ == "__main__":
    main()
