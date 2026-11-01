"""Independently re-evaluate selected native/Metal development raw comparisons."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "library_development_audit_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, E, N, RAW, ORIGINS = (
    load(n) for n in ("authority", "execution", "native_assets", "evidence", "audit")
)
C = A.C


def producer_role(auditor, root=None, revision=None):
    """Permit the audit-only repair without relabelling an existing worker."""
    C.require((root is None) == (revision is None), "Both producer source arguments are required")
    if root is None:
        return auditor
    producer = A.source_role(root, revision, control=True)
    audit_only = {A.CONTROL_PREFIX + n for n in ("audit_development.py", "qualify.py")}
    C.require(
        {k: v for k, v in producer["files"].items() if k not in audit_only}
        == {k: v for k, v in auditor["files"].items() if k not in audit_only},
        "Audit-only reuse requires unchanged worker and numerical dependencies",
    )
    return producer


def validate_capabilities(capabilities):
    # The frozen Adam gates use gradients from all five engines. MPS and MLX
    # therefore execute even though the gates target only ARM64 and Metal.
    C.require(
        set(capabilities) == {"metal", "mps", "mlx"},
        "Missing or foreign cross-engine gradient capability proof",
    )
    for engine, cap in capabilities.items():
        E.validate_capabilities(engine, cap)


def worker_identity(record, job, runtime_identity, roles):
    C.require(
        record["schema"] == "frozen-library-development-worker-v1"
        and record["complete"] is True
        and record["integrity_after"] is True
        and record["formal_eligible"] is False
        and type(record["formal_fits"]) is int
        and record["formal_fits"] == 0
        and record["admission_stream_opened"] is False
        and "error" not in record,
        "Incomplete or foreign development worker",
    )
    C.require(
        type(record["pid"]) is int
        and record["pid"] > 0
        and type(record["parent_pid"]) is int
        and record["parent_pid"] == job["parent_pid"]
        and record["pid"] != record["parent_pid"],
        "Wrong worker process identity",
    )
    for phase in ("before", "after"):
        proof = record["origins_" + phase]
        C.require(
            proof["runtime_identity_sha256"] == runtime_identity
            and proof["source_roles_sha256"] == C.identity(roles),
            "Foreign worker authority",
        )


def runtime_validation(record, job, capture, inventory, roles):
    """Require full final authentication for workers using the new guard."""
    if A.CONTROL_PREFIX + "runtime_guard.py" not in roles["control"]["files"]:
        return
    for phase in ("before", "after"):
        ORIGINS.validate_origins(
            record["origins_" + phase],
            roles,
            capture,
            inventory,
            job.get("engine", "cpu_torch"),
            pid=record["pid"],
        )
    ORIGINS.validate_validation(
        record.get("runtime_validation_final"),
        roles,
        capture,
        inventory,
        pid=record["pid"],
        final=True,
    )


def execute(args):
    C.require(sys.flags.isolated and sys.flags.dont_write_bytecode, "Use audit Python -I -B")
    C.require(
        not any(n.split(".", 1)[0] in {"transformer", "torch", "mlx"} for n in sys.modules),
        "Independent audit requires fresh source/framework imports",
    )
    control = A.source_role(HERE.parents[1], args.control_revision, control=True)
    producer = producer_role(control, args.producer_control_root, args.producer_control_revision)
    math = A.source_role(args.math_root, C.MATH_REVISION)
    path = C.regular(args.receipt)
    record = C.read(path, args.receipt_sha256)
    C.bound(record["job"])
    job = C.read(record["job"]["path"], record["job"]["sha256"])
    worker_module = load("worker")
    worker_module.validate_job(job)
    roles = dict(control=producer, math=math)
    C.require(
        job["kind"] == "native_metal_gates"
        and job["roles"] == roles
        and Path(job["output"]) == path.parent
        and job["capture"] == C.binding(args.capture),
        "Wrong development source or output",
    )
    capture = C.read(args.capture, args.capture_sha256)
    C.bound(capture["runtime_receipt"])
    receipt = C.read(capture["runtime_receipt"]["path"], capture["runtime_receipt"]["sha256"])
    C.require(
        receipt["project"] == dict(root=math["root"], files=math["files"]), "Wrong math source"
    )
    N.verify(job["native_assets"], math)
    C.bound(capture["runtime_inventory"])
    inventory = C.read(capture["runtime_inventory"]["path"], capture["runtime_inventory"]["sha256"])
    sys.path.insert(0, math["root"])
    from transformer.gpu_owned_backend import validate_loaded_origin_proof

    expected = [
        g
        for g in C.comparison()["gate_inventory"]
        if g["id"].split("/")[:2]
        in [[e, str(job["width"])] for e in ("cpu_arm64", "metal", "oracle")]
    ]
    worker_identity(record, job, capture["runtime_identity_sha256"], roles)
    runtime_validation(record, job, capture, inventory, roles)
    result = record["outputs"]
    C.require(
        result["expected_inventory"] == expected
        and [g["id"] for g in result["groups"]] == [g["id"] for g in expected]
        and result["raw_store"] == job["raw_store"],
        "Incomplete development gate denominator",
    )
    decisions = []
    for specification, row in zip(expected, result["groups"], strict=True):
        passed = RAW.audit_gate(specification, row, job["raw_store"], math["root"])
        C.require(passed or not specification["blocking"], "Blocking raw development gate failed")
        decisions.append(dict(id=row["id"], passed=passed, blocking=specification["blocking"]))
    seen = {record["pid"]}
    child_kinds = []
    for child in result["child_proofs"]:
        for name in ("receipt", "job", "log"):
            C.bound(child[name])
        saved = C.read(child["receipt"]["path"], child["receipt"]["sha256"])
        task = C.read(child["job"]["path"], child["job"]["sha256"])
        worker_module.validate_job(task)
        C.require(
            task["roles"] == roles
            and task["capture"] == job["capture"]
            and task["native_assets"] == job["native_assets"]
            and task["parent_pid"] == record["pid"]
            and task["width"] == job["width"]
            and child["returncode"] == 0
            and child["pid"] == saved["pid"]
            and saved["job"] == child["job"]
            and saved["pid"] not in seen,
            "Stale, repeated or foreign development child",
        )
        seen.add(saved["pid"])
        worker_identity(saved, task, capture["runtime_identity_sha256"], roles)
        runtime_validation(saved, task, capture, inventory, roles)
        child_kinds.append((task["engine"], task["kind"]))
        for binding in saved["outputs"].values():
            if isinstance(binding, dict) and set(binding) == {"path", "sha256", "bytes"}:
                C.bound(binding)
        for binding in saved["outputs"].get("files", []):
            C.bound(binding)
        if task["engine"] == "metal":
            E.validate_capabilities("metal", saved["outputs"]["capabilities"])
            C.require(
                saved["outputs"]["metal_library_origin"]
                == C.binding(Path(producer["root"]) / A.METAL_LIBRARY),
                "Fresh Metal worker used another native library",
            )
    C.require(
        sorted(child_kinds)
        == sorted(
            [
                ("cpu_arm64", "repeat_training"),
                ("cpu_arm64", "repeat_training"),
                ("metal", "repeat_training"),
                ("metal", "repeat_training"),
                ("cpu_arm64", "checkpoint_tie"),
            ]
        ),
        "Fresh-process/tie denominator changed",
    )
    for phase in ("before", "after"):
        proof = record["origins_" + phase]
        validate_loaded_origin_proof(
            proof["pytorch"],
            inventory=inventory,
            runtime_root=capture["runtime_root"],
            receipt=receipt,
        )
    validate_capabilities(result["capabilities"])
    C.require(
        result["metal_library_origin"] == C.binding(Path(producer["root"]) / A.METAL_LIBRARY),
        "Wrong retained Metal origin",
    )
    origins = ORIGINS.project_origins(math, dict(sys.modules))
    C.require(
        "torch.nn.precise_fp32" not in sys.modules and "mlx.core" not in sys.modules,
        "Selected candidate must not replace independent audit arithmetic",
    )
    A.verify_role(control, control=True)
    A.verify_role(producer, control=True)
    A.verify_role(math)
    C.require(C.binding(path)["sha256"] == args.receipt_sha256, "Worker receipt changed")
    return dict(
        schema="frozen-library-native-metal-audit-v1",
        evidence_valid=True,
        numerics_passed=True,
        receipt=C.binding(path),
        capture=C.binding(args.capture),
        width=job["width"],
        groups=decisions,
        child_pids=sorted(seen - {record["pid"]}),
        auditor_control_role=control,
        worker_control_role=producer,
        auditor_math_revision=math["revision"],
        auditor_project_origins=origins,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-revision", required=True)
    parser.add_argument("--producer-control-root", type=Path)
    parser.add_argument("--producer-control-revision")
    parser.add_argument("--math-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args), allow_nan=False))
