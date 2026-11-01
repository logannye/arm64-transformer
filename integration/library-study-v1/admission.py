"""Serial six-width admission and verification for the explicit library successor."""

import argparse
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "library_admission_controller_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, F, Q = (load(n) for n in ("authority", "frozen", "qualify"))
C = A.C
WIDTHS = C.WIDTHS
COMPLETED_WIDTH_BINDINGS = {
    32: "17d9aee5125310e6a7301b6eb74a948e0604f8d809652204e53f7055f1b8eb63",
    64: "26c7dd11e683824f7300dcc7d1b64c07aa70b28788057a316adea006f0f66b18",
    128: "051c209df1148d8ee0ddbea7580f13944b828427689997bab7489aec91a19622",
}


def reuse_source(manifest, width, binding):
    """Recover the three completed workers preceding the diagnosed width-256 abort."""
    C.require(
        width in COMPLETED_WIDTH_BINDINGS and binding["sha256"] == COMPLETED_WIDTH_BINDINGS[width],
        "Only the three recorded complete admission workers may be reused",
    )
    original = F.PLAN.load(binding)
    worker = F.PLAN.load(original["receipt"])
    previous = F.PLAN.load(worker["manifest"])
    old, new = previous["bindings"], manifest["bindings"]
    producer = A.verify_role(old["roles"]["control"], control=True)
    C.require(
        original["width"] == width
        and worker["complete"] is True
        and worker["integrity_after"] is True
        and worker["outputs"]["blocking_passed"] is True
        and producer["revision"] == "143c58c400151ef4ef834bf9c56da5ce91e7eaed"
        and old["roles"]["math"] == new["roles"]["math"]
        and all(old[k] == new[k] for k in ("capture", "fixtures", "native_assets", "hardware"))
        and previous["protocol"] == C.ORIGINAL_PROTOCOL
        and manifest["protocol"] == C.PROTOCOL
        and previous["comparison"] == C.original_comparison()
        and manifest["comparison"] == C.comparison()
        and all(
            previous[k] == manifest[k]
            for k in previous
            if k not in ("bindings", "source_roles", "frozen_utc", "protocol", "comparison")
        ),
        "Completed admission reuse changed scientific, runtime or source identity",
    )
    allowed = {
        "integration/library-study-v1/" + name
        for name in (
            "admission.py",
            "admission_worker.py",
            "audit_admission.py",
            "contract.py",
            "qualification_plan.py",
            "campaign.py",
            "audit_campaign.py",
            "report.py",
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
        "Completed admission reuse changed unrelated execution controls",
    )
    return original, worker["manifest"], producer


def reuse_audit_argv(python, producer, manifest_binding, receipt):
    return [
        str(python),
        "-I",
        "-B",
        str(Path(producer["root"]) / "integration/library-study-v1/audit_admission.py"),
        "--manifest",
        manifest_binding["path"],
        "--manifest-sha256",
        manifest_binding["sha256"],
        "--receipt",
        receipt["path"],
        "--receipt-sha256",
        receipt["sha256"],
    ]


def verify(manifest, manifest_binding, binding):
    F.verify(manifest)
    value = F.PLAN.load(binding)
    C.require(
        value["schema"] == "frozen-library-admission-v1"
        and value["manifest"] == manifest_binding
        and value["complete"] is True
        and value["formal_eligible"] is True
        and type(value["formal_fits"]) is int
        and value["formal_fits"] == 0
        and value["admission_stream_opened"] is True
        and value["seed"] == C.ADMISSION_SEED
        and value["gate_inventory_sha256"] == C.INVENTORY_SHA256
        and [r["width"] for r in value["widths"]] == list(WIDTHS),
        "Admission is incomplete, foreign or ineligible",
    )
    observed = []
    for row in value["widths"]:
        producer_manifest = manifest_binding
        if "reused_from" in row:
            original, producer_manifest, producer = reuse_source(
                manifest, row["width"], row["reused_from"]
            )
            execution = row["audit_execution"]
            for name in ("stdout", "stderr"):
                C.bound(execution[name])
            C.require(
                row["receipt"] == original["receipt"]
                and row["worker_execution"] == original["worker_execution"]
                and execution["exit_code"] == 0
                and execution["stdout"] == row["audit"]
                and execution["argv"]
                == reuse_audit_argv(
                    execution["argv"][0], producer, producer_manifest, row["receipt"]
                ),
                "Completed worker requires an independent original-source re-audit",
            )
        worker, audit = F.PLAN.load(row["receipt"]), F.PLAN.load(row["audit"])
        C.require(
            worker["complete"] is True
            and worker["integrity_after"] is True
            and worker["manifest"] == producer_manifest
            and worker["outputs"]["blocking_passed"] is True
            and audit["schema"] == "frozen-library-admission-width-audit-v1"
            and audit["manifest"] == producer_manifest
            and audit["receipt"] == row["receipt"]
            and audit["evidence_valid"] is True
            and audit["blocking_passed"] is True
            and audit["width"] == row["width"],
            "Width admission audit is stale or failed",
        )
        specs = [
            g
            for g in C.comparison()["gate_inventory"]
            if int(g["id"].split("/")[1]) == row["width"]
        ]
        C.require(
            worker["outputs"]["expected_inventory"] == specs
            and [g["id"] for g in audit["groups"]] == [g["id"] for g in specs]
            and all(
                g["blocking"] is s["blocking"] and (g["passed"] is True or not s["blocking"])
                for g, s in zip(audit["groups"], specs, strict=True)
            ),
            "Admission group denominator/decision changed",
        )
        observed.extend(g["id"] for g in audit["groups"])
    C.require(
        observed == [g["id"] for g in C.comparison()["gate_inventory"]],
        "Admission groups reordered or omitted",
    )
    return value


def execute(args):
    C.require_execution_support()
    manifest = F.read(args.manifest, args.manifest_sha256)
    manifest_binding = C.binding(args.manifest)
    roles, capture_binding = manifest["bindings"]["roles"], manifest["bindings"]["capture"]
    capture = F.PLAN.load(capture_binding)
    inventory = F.PLAN.load(capture["runtime_inventory"])
    prerequisite = F.PLAN.load(manifest["bindings"]["raw_prerequisites"])
    request = F.PLAN.load(prerequisite["request"])
    C.require(
        Path(args.raw_store).absolute() == Path(request["raw_store"]),
        "Unfrozen admission raw store",
    )
    output = Path(args.output).absolute()
    C.require(
        output.resolve() == output and not output.exists() and not output.is_symlink(),
        "Fresh admission output required",
    )
    protected = [Path(r["root"]) for r in roles.values()] + [Path(args.raw_store)]
    protected += [Path(p) for p in inventory["layout"]["scan_roots"]]
    C.require(
        all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
        "Admission output overlaps protected source/runtime",
    )
    C.require(
        shutil.disk_usage(output.parent).free > C.MIN_FREE_BYTES + 4 * 1024**3,
        "Insufficient admission disk reserve",
    )
    output.mkdir()
    path = Path(roles["math"]["root"]) / "integration/gpu-owned/owned_mlx_suites.py"
    spec = importlib.util.spec_from_file_location("admission_controller_environment", path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    environment = A.execution_environment(suite.controls(dict(os.environ)))
    rows = []
    initial_raw = (
        int(
            subprocess.check_output(["/usr/bin/du", "-sk", str(args.raw_store)], text=True).split()[
                0
            ]
        )
        * 1024
    )
    C.write(
        output / "opened.json",
        dict(
            schema="frozen-library-admission-opened-v1",
            manifest=manifest_binding,
            seed=C.ADMISSION_SEED,
            formal_fits=0,
            admission_stream_opened=True,
            formal_eligible=False,
        ),
    )
    for width in WIDTHS:
        F.verify(manifest)
        env = dict(environment, ARM64_TRANSFORMER_WIDTH=str(width))
        if args.reuse_admission is not None and width in COMPLETED_WIDTH_BINDINGS:
            prior_binding = C.binding(args.reuse_admission / f"width-{width}-binding.json")
            original, producer_manifest, producer = reuse_source(manifest, width, prior_binding)
            saved = F.PLAN.load(original["receipt"])
            C.require(
                saved["outputs"]["raw_store"] == str(Path(args.raw_store).absolute()),
                "Reused admission raw store differs",
            )
            argv = reuse_audit_argv(
                args.audit_python, producer, producer_manifest, original["receipt"]
            )
            audited, execution = Q.run(argv, f"width-{width}-audit", output, env)
            C.require(
                audited["evidence_valid"] is True and audited["blocking_passed"] is True,
                "Completed admission worker failed independent re-audit",
            )
            row = dict(
                original,
                audit=execution["stdout"],
                audit_execution=execution,
                reused_from=prior_binding,
            )
            rows.append(row)
            C.write(output / f"width-{width}-binding.json", row)
            continue
        destination = output / f"width-{width}"
        job = dict(
            schema="frozen-library-admission-job-v1",
            kind="width",
            manifest=manifest_binding,
            width=width,
            output=str(destination),
            raw_store=str(Path(args.raw_store).absolute()),
            raw_start_bytes=initial_raw,
            parent_pid=os.getpid(),
        )
        if width == 32 and args.device_repair_receipt is not None:
            job["device_repair"] = C.binding(args.device_repair_receipt)
            load("admission_worker").validate_job(job)
        job_path = output / f"width-{width}-job.json"
        C.write(job_path, job)
        argv = [
            inventory["layout"]["interpreter"],
            "-P",
            "-S",
            "-B",
            str(HERE / "admission_worker.py"),
            "--job",
            str(job_path),
            "--job-sha256",
            C.binding(job_path)["sha256"],
        ]
        _, execution = Q.run(argv, f"width-{width}-worker", output, env)
        receipt = C.binding(destination / "result.json")
        argv = [
            str(args.audit_python),
            "-I",
            "-B",
            str(HERE / "audit_admission.py"),
            "--manifest",
            str(args.manifest),
            "--manifest-sha256",
            args.manifest_sha256,
            "--receipt",
            receipt["path"],
            "--receipt-sha256",
            receipt["sha256"],
        ]
        audited, audit = Q.run(argv, f"width-{width}-audit", output, env)
        C.require(
            audited["evidence_valid"] is True and audited["blocking_passed"] is True,
            "Blocking admission failed; preserve exposed stream and raw evidence",
        )
        row = dict(width=width, receipt=receipt, audit=audit["stdout"], worker_execution=execution)
        rows.append(row)
        C.write(output / f"width-{width}-binding.json", row)
    result = dict(
        schema="frozen-library-admission-v1",
        manifest=manifest_binding,
        widths=rows,
        gate_inventory_sha256=C.INVENTORY_SHA256,
        seed=C.ADMISSION_SEED,
        complete=True,
        formal_eligible=True,
        formal_fits=0,
        admission_stream_opened=True,
    )
    C.write(output / "result.json", result)
    verify(manifest, manifest_binding, C.binding(output / "result.json"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--audit-python", type=Path, required=True)
    parser.add_argument("--raw-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-repair-receipt", type=Path)
    parser.add_argument("--reuse-admission", type=Path)
    execute(parser.parse_args())
