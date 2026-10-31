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
WIDTHS = (32, 64, 128, 256, 512, 1024)


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
        worker, audit = F.PLAN.load(row["receipt"]), F.PLAN.load(row["audit"])
        C.require(
            worker["complete"] is True
            and worker["integrity_after"] is True
            and worker["manifest"] == manifest_binding
            and worker["outputs"]["blocking_passed"] is True
            and audit["schema"] == "frozen-library-admission-width-audit-v1"
            and audit["manifest"] == manifest_binding
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
    environment = suite.controls(dict(os.environ))
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
        job_path = output / f"width-{width}-job.json"
        C.write(job_path, job)
        env = dict(environment, ARM64_TRANSFORMER_WIDTH=str(width))
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
    execute(parser.parse_args())
