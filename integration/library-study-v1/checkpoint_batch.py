"""Untimed same-engine, same-width replay batches; every checkpoint is retained.

Only interpreter/runtime authentication is amortized. Each checkpoint still
loads its bound parameters and fixture, creates/closes its own selected backend,
and executes the unchanged full replay. Independent FP64 workers stay fresh.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "library_batched_checkpoint_replay", HERE / "checkpoint_replay.py"
)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
C = R.C
MAX_JOBS = 60


def jobs(value):
    C.require(
        set(value) - {"audit_recovery"}
        == {"schema", "manifest", "admission", "width", "engine", "jobs", "output", "parent_pid"}
        and value["schema"] == "frozen-library-checkpoint-batch-job-v1"
        and value["engine"] in R.ENGINES
        and type(value["width"]) is int
        and value["width"] in (32, 64, 128, 256, 512, 1024)
        and type(value["parent_pid"]) is int
        and value["parent_pid"] == os.getppid()
        and isinstance(value["jobs"], list)
        and 0 < len(value["jobs"]) <= MAX_JOBS,
        "Invalid checkpoint batch",
    )
    paths, outputs, checkpoints = set(), set(), set()
    for binding in value["jobs"]:
        source = C.read(C.bound(binding))
        C.require(
            source["engine"] == value["engine"]
            and source["manifest"] == value["manifest"]
            and source["admission"] == value["admission"]
            and source.get("audit_recovery") == value.get("audit_recovery")
            and source["parent_pid"] == value["parent_pid"],
            "Checkpoint batch mixes engine, authority or parent",
        )
        cell = C.read(C.bound(source["cell_receipt"]))["cell"]
        C.require(
            type(cell["width"]) is int and cell["width"] == value["width"],
            "Checkpoint batch mixes model widths",
        )
        C.require(
            binding["path"] not in paths and source["output"] not in outputs,
            "Duplicate checkpoint batch job or output",
        )
        key = (source["cell_receipt"]["path"], source["checkpoint"])
        C.require(key not in checkpoints, "Duplicate checkpoint batch parameters")
        checkpoints.add(key)
        paths.add(binding["path"])
        outputs.add(source["output"])
    return value["jobs"]


def execute(path, digest):
    value = C.read(path, digest)
    bindings = jobs(value)
    C.require(
        os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(value["width"]), "Select batch width first"
    )
    output = Path(value["output"])
    C.require(
        output.is_absolute() and output.resolve() == output and not output.exists(),
        "Fresh canonical batch output required",
    )
    context = R.recovery_context(value)
    manifest = (
        context.manifest
        if context
        else R.F.read(value["manifest"]["path"], value["manifest"]["sha256"])
    )
    if context is None:
        R.AD.verify(manifest, value["manifest"], value["admission"])
    from types import SimpleNamespace

    capture = R.F.PLAN.load(manifest["bindings"]["capture"])
    inventory = R.F.PLAN.load(capture["runtime_inventory"])
    first = C.read(C.bound(bindings[0]))
    R.W.protected_output(
        dict(output=str(output), raw_store=first["raw_store"]),
        SimpleNamespace(
            roles=context.roles if context else manifest["bindings"]["roles"], inventory=inventory
        ),
    )
    if context:
        context.protect(output)
    output.mkdir()
    runtime, records = None, []
    for binding in bindings:
        C.bound(binding)
        reply, runtime = R.execute(
            Path(binding["path"]),
            binding["sha256"],
            runtime=runtime,
            defer_full_postflight=True,
            quiet=True,
        )
        C.require(reply["complete"] is True, "Incomplete batched checkpoint")
        records.append(dict(job=binding, receipt=reply["receipt"]))
        C.write(output / f"progress-{len(records):03d}.json", dict(records=records, complete=False))
    final = runtime.finish()
    if context:
        context.verify()
    C.require(C.binding(path)["sha256"] == digest, "Checkpoint batch job changed")
    for binding in bindings:
        C.bound(binding)
    result = dict(
        schema="frozen-library-checkpoint-batch-v1",
        job=C.binding(path),
        manifest=value["manifest"],
        admission=value["admission"],
        engine=value["engine"],
        width=value["width"],
        pid=os.getpid(),
        parent_pid=os.getppid(),
        records=records,
        runtime_validation_final=final,
        complete=True,
    )
    C.write(output / "receipt.json", result)
    print(json.dumps(dict(complete=True, receipt=C.binding(output / "receipt.json"))))


def validate(binding, replay, replay_binding, manifest_binding, admission, engine):
    value = C.read(C.bound(binding))
    source = C.read(C.bound(value["job"]))
    C.require(
        value["schema"] == "frozen-library-checkpoint-batch-v1"
        and value["complete"] is True
        and value["manifest"] == manifest_binding
        and value["admission"] == admission
        and value["engine"] == engine
        and value["pid"] == replay["pid"]
        and value["parent_pid"] == replay["parent_pid"] == source["parent_pid"]
        and value["width"] == source["width"]
        and value["engine"] == source["engine"]
        and source["manifest"] == manifest_binding
        and source["admission"] == admission
        and source.get("audit_recovery") == replay.get("audit_recovery"),
        "Foreign checkpoint batch proof",
    )
    C.require(
        len(value["records"]) == len(source["jobs"]) <= MAX_JOBS
        and [r["job"] for r in value["records"]] == source["jobs"]
        and sum(
            r["receipt"] == replay_binding and r["job"] == replay["job"] for r in value["records"]
        )
        == 1,
        "Missing, duplicate or foreign checkpoint batch member",
    )
    return value["runtime_validation_final"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    args = parser.parse_args()
    execute(args.job, args.job_sha256)
