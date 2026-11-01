"""Re-evaluate every frozen admission member and fresh-process control from raw bytes."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "library_admission_audit_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, F, RAW, E, WORKER, ORIGINS = (
    load(n) for n in ("authority", "frozen", "evidence", "execution", "admission_worker", "audit")
)
C = A.C


def worker_identity(record, job, manifest, capture, inventory):
    C.require(
        record["schema"] == "frozen-library-admission-worker-v1"
        and record["complete"] is True
        and record["integrity_after"] is True
        and record["formal_eligible"] is False
        and type(record["formal_fits"]) is int
        and record["formal_fits"] == 0
        and record["admission_stream_opened"] is True
        and record["manifest"] == job["manifest"]
        and "error" not in record,
        "Incomplete or foreign admission worker",
    )
    C.require(
        type(record["pid"]) is int
        and type(record["parent_pid"]) is int
        and record["pid"] > 0
        and record["parent_pid"] == job["parent_pid"]
        and record["pid"] != record["parent_pid"],
        "Foreign admission process",
    )
    from transformer.gpu_owned_backend import (
        validate_compilation_probe,
        validate_loaded_origin_proof,
    )

    receipt = F.PLAN.load(capture["runtime_receipt"])
    roles = manifest["bindings"]["roles"]
    ORIGINS.validate_validation(
        record.get("runtime_validation_final"),
        roles,
        capture,
        inventory,
        pid=record["pid"],
        final=True,
    )
    imports = load("qualify_suites").imports
    for phase in ("before", "after"):
        proof = record["origins_" + phase]
        ORIGINS.validate_validation(
            proof.get("runtime_validation"), roles, capture, inventory, pid=record["pid"]
        )
        C.require(
            proof["runtime_identity_sha256"] == capture["runtime_identity_sha256"]
            and proof["source_roles_sha256"] == C.identity(roles),
            "Foreign admission runtime/source",
        )
        for engine in ("pytorch", "mlx"):
            if engine in proof:
                validate_loaded_origin_proof(
                    proof[engine],
                    inventory=inventory,
                    runtime_root=capture["runtime_root"],
                    receipt=receipt,
                )
        if "mlx" in proof:
            validate_compilation_probe(proof["compilation"])
        imports(
            proof["loaded_python_files"],
            dict(inventory, root=capture["runtime_root"]),
            A.merged_project(roles),
        )


def audit_children(record, job, manifest, capture, inventory, np):
    roles, outputs = manifest["bindings"]["roles"], record["outputs"]
    children, pids = {}, {record["pid"]}
    for child in outputs["child_proofs"]:
        for name in ("receipt", "job", "log"):
            C.bound(child[name])
        saved, task = F.PLAN.load(child["receipt"]), F.PLAN.load(child["job"])
        WORKER.validate_job(task)
        C.require(
            task["manifest"] == job["manifest"]
            and task["width"] == job["width"]
            and task["parent_pid"] == record["pid"]
            and task["raw_store"] == job["raw_store"]
            and saved["job"] == child["job"]
            and saved["pid"] == child["pid"]
            and child["returncode"] == 0
            and saved["pid"] not in pids,
            "Stale or repeated admission child",
        )
        pids.add(saved["pid"])
        worker_identity(saved, task, manifest, capture, inventory)
        children.setdefault((task["engine"], task["kind"]), []).append(saved)
        for item in saved["outputs"].get("files", []):
            C.bound(item)
    expected = Counter({(e, "repeat_training"): 2 for e in WORKER.ENGINES})
    expected.update({(e, "checkpoint_tie"): 1 for e in WORKER.ENGINES[:2]})
    C.require(
        Counter({k: len(v) for k, v in children.items()}) == expected,
        "Admission fresh-process/tie denominator differs",
    )
    count = 8 * job["width"] ** 2 + 301 * job["width"] + 128
    for engine in WORKER.ENGINES:
        states = []
        for saved in children[(engine, "repeat_training")]:
            if engine == "mlx":
                task = F.PLAN.load(saved["job"])
                ORIGINS.training_graph(
                    roles["math"], saved["outputs"]["compiled_training"], task["output"]
                )
            bound = saved["outputs"]["state"]
            C.bound(bound)
            with Path(bound["path"]).open("rb") as stream:
                state = RAW.decode_arrays(stream, ("parameters", "m", "v", "step", "logits"))
            C.require(
                all(
                    state[k].dtype == np.float32
                    and state[k].shape == (count,)
                    and np.isfinite(state[k]).all()
                    for k in ("parameters", "m", "v")
                )
                and state["step"].dtype == np.int64
                and state["step"].shape == ()
                and int(state["step"]) == 6
                and (state["v"] >= 0).all()
                and state["logits"].dtype == np.float32
                and state["logits"].shape == (3, 32, 128)
                and np.isfinite(state["logits"]).all(),
                "Wrong fresh admission state",
            )
            states.append(state)
        C.require(
            all(
                states[0][k].dtype == states[1][k].dtype
                and states[0][k].shape == states[1][k].shape
                and states[0][k].tobytes() == states[1][k].tobytes()
                for k in states[0]
            ),
            "Raw fresh-process training differs",
        )
    for engine in WORKER.ENGINES[:2]:
        bound = children[(engine, "checkpoint_tie")][0]["outputs"]["observations"]
        C.bound(bound)
        with Path(bound["path"]).open("rb") as stream:
            values = RAW.decode_arrays(stream, ("actual", "expected"))
        expected = np.array([1, 2, 1, 1, 1, 1, 1], np.float64)
        C.require(
            all(
                v.dtype == expected.dtype
                and v.shape == expected.shape
                and v.tobytes() == expected.tobytes()
                for v in values.values()
            ),
            "CPU first-tie control failed",
        )
    C.require(
        set(outputs["capabilities"]) == {"metal", "mps", "mlx"},
        "Missing GPU admission capabilities",
    )
    for engine, cap in outputs["capabilities"].items():
        E.validate_capabilities(engine, cap)
    C.require(
        outputs["metal_library_origin"]
        == C.binding(Path(roles["control"]["root"]) / A.METAL_LIBRARY),
        "Admission Metal host origin differs",
    )
    return sorted(pids - {record["pid"]})


def audit_device_repair(record, job, manifest):
    """Reaudit the one identified failed observation without repeating trajectories."""
    prior = F.PLAN.load(job["device_repair"])
    previous = F.PLAN.load(prior["manifest"])
    old, new = previous["bindings"], manifest["bindings"]
    producer = A.verify_role(old["roles"]["control"], control=True)
    C.require(
        producer["revision"] == "b521fa2ba14ddad4faf13f8a1a762f027b9e9ff3"
        and old["roles"]["math"] == new["roles"]["math"]
        and all(old[k] == new[k] for k in ("capture", "fixtures", "native_assets", "hardware"))
        and all(
            previous[k] == manifest[k]
            for k in previous
            if k not in ("bindings", "source_roles", "frozen_utc")
        ),
        "Device repair cannot transfer numerical or runtime identity",
    )
    allowed = {
        "integration/library-study-v1/" + name
        for name in (
            "context.py",
            "qualify.py",
            "admission.py",
            "admission_worker.py",
            "audit_admission.py",
        )
    }
    current = new["roles"]["control"]
    C.require(
        set(producer["files"]) == set(current["files"])
        and all(
            producer["files"][k] == current["files"][k]
            for k in producer["files"]
            if k not in allowed
        ),
        "Device repair changed unrelated execution controls",
    )
    argv = [
        sys.executable,
        "-I",
        "-B",
        str(Path(producer["root"]) / "integration/library-study-v1/audit_admission.py"),
        "--manifest",
        prior["manifest"]["path"],
        "--manifest-sha256",
        prior["manifest"]["sha256"],
        "--receipt",
        job["device_repair"]["path"],
        "--receipt-sha256",
        job["device_repair"]["sha256"],
    ]
    audit = subprocess.run(argv, capture_output=True, text=True, timeout=3600, check=False)
    C.require(audit.returncode == 1 and not audit.stderr, "Original device failure reaudit differs")
    audited = json.loads(audit.stdout)
    C.require(
        audited["evidence_valid"] is True
        and audited["blocking_passed"] is False
        and audited["width"] == 32
        and audited["receipt"] == job["device_repair"]
        and audited["manifest"] == prior["manifest"]
        and [g["id"] for g in audited["groups"] if g["blocking"] and not g["passed"]]
        == [WORKER.DEVICE_REPAIR_ID],
        "Original admission has another blocking failure",
    )
    before, after = prior["outputs"], record["outputs"]
    C.require(
        before["expected_inventory"] == after["expected_inventory"]
        and before["raw_store"] == after["raw_store"] == job["raw_store"]
        and before["child_proofs"] == after["child_proofs"]
        and before["metal_library_origin"] == after["metal_library_origin"]
        and set(after["capabilities"]) == {"metal", "mps", "mlx"}
        and all(before["capabilities"][e] == after["capabilities"][e] for e in ("metal", "mps"))
        and record["pid"] not in {prior["pid"], *audited["child_pids"]},
        "Device repair changed retained provenance",
    )
    for first, second in zip(before["groups"], after["groups"], strict=True):
        if first["id"] != WORKER.DEVICE_REPAIR_ID:
            C.require(first == second, "Device repair changed another raw admission group")
        else:
            import numpy as np

            actual, expected = RAW.load_pair(
                first["members"]["observations"], job["raw_store"], new["roles"]["math"]["root"]
            )
            C.require(
                actual.dtype == expected.dtype == np.int32
                and actual.shape == expected.shape == (8,)
                and np.array_equal(actual, [0, 1, 1, 1, 1, 1, 1, 1])
                and np.array_equal(expected, np.ones(8, np.int32)),
                "Original device observation differs from the diagnosed string mismatch",
            )
            repaired, target = RAW.load_pair(
                second["members"]["observations"], job["raw_store"], new["roles"]["math"]["root"]
            )
            C.require(
                repaired.dtype == target.dtype == expected.dtype
                and repaired.shape == target.shape == expected.shape
                and target.tobytes() == expected.tobytes()
                and repaired[1:].tobytes() == actual[1:].tobytes(),
                "Device repair changed another observation or its denominator",
            )
    E.validate_capabilities("mlx", after["capabilities"]["mlx"])
    C.bound(job["device_repair"])
    return dict(
        receipt=job["device_repair"],
        manifest=prior["manifest"],
        replaced_group=WORKER.DEVICE_REPAIR_ID,
        reused_groups=len(before["groups"]) - 1,
        original_audit=audited,
        audit_argv=argv,
        audit_exit_code=audit.returncode,
    )


def execute(args):
    C.require_execution_support()
    C.require(
        sys.flags.isolated and sys.flags.dont_write_bytecode,
        "Use independent admission auditor -I -B",
    )
    C.require(
        not any(n.split(".", 1)[0] in {"transformer", "torch", "mlx"} for n in sys.modules),
        "Fresh independent admission audit required",
    )
    manifest = F.read(args.manifest, args.manifest_sha256)
    roles = manifest["bindings"]["roles"]
    C.require(Path(roles["control"]["root"]) == HERE.parents[1], "Admission auditor source differs")
    record = C.read(args.receipt, args.receipt_sha256)
    C.bound(record["job"])
    job = C.read(record["job"]["path"], record["job"]["sha256"])
    WORKER.validate_job(job)
    C.require(
        job["kind"] == "width"
        and job["manifest"] == C.binding(args.manifest)
        and Path(job["output"]) == Path(args.receipt).parent
        and os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(job["width"]),
        "Foreign admission width/job",
    )
    capture = F.PLAN.load(manifest["bindings"]["capture"])
    inventory = F.PLAN.load(capture["runtime_inventory"])
    sys.path.insert(0, roles["math"]["root"])
    import numpy as np

    worker_identity(record, job, manifest, capture, inventory)
    specs = [
        s for s in C.comparison()["gate_inventory"] if int(s["id"].split("/")[1]) == job["width"]
    ]
    outputs = record["outputs"]
    C.require(
        outputs["expected_inventory"] == specs
        and [r["id"] for r in outputs["groups"]] == [s["id"] for s in specs]
        and outputs["raw_store"] == job["raw_store"],
        "Incomplete admission denominator",
    )
    checked = []
    for row, spec in zip(outputs["groups"], specs, strict=True):
        if row.get("status") == "runtime_error":
            C.require(
                spec["family"] == "diagnostic_precision"
                and spec["blocking"] is False
                and row["passed"] is False
                and row["members"] == {}
                and row["decisions"] == {},
                "Runtime error cannot waive a blocking admission group",
            )
            error = F.PLAN.load(row["error"])
            C.require(
                error["gate"] == row["id"] and error["exception_type"] and error["traceback"],
                "Missing diagnostic failure evidence",
            )
            passed = False
        else:
            passed = RAW.audit_gate(spec, row, job["raw_store"], roles["math"]["root"])
        checked.append(dict(id=row["id"], passed=passed, blocking=spec["blocking"]))
    repair = None
    if "device_repair" in job:
        repair = audit_device_repair(record, job, manifest)
        child_pids = repair["original_audit"]["child_pids"]
    else:
        child_pids = audit_children(record, job, manifest, capture, inventory, np)
    blocking = all(r["passed"] for r in checked if r["blocking"])
    C.require(
        outputs["blocking_passed"] is blocking, "Blocking admission result differs from raw members"
    )
    proof = ORIGINS.project_origins(roles["math"], dict(sys.modules))
    C.require(
        "torch.nn.precise_fp32" not in sys.modules and "mlx.core" not in sys.modules,
        "Candidate must not replace independent admission arithmetic",
    )
    F.verify(manifest)
    C.require(C.binding(args.receipt)["sha256"] == args.receipt_sha256, "Admission receipt changed")
    return dict(
        schema="frozen-library-admission-width-audit-v1",
        evidence_valid=True,
        blocking_passed=blocking,
        width=job["width"],
        groups=checked,
        manifest=C.binding(args.manifest),
        receipt=C.binding(args.receipt),
        child_pids=child_pids,
        device_repair=repair,
        auditor_project_origins=proof,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    value = execute(args)
    print(json.dumps(value, allow_nan=False))
    raise SystemExit(0 if value["blocking_passed"] else 1)
