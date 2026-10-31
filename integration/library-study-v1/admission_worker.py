"""Complete frozen successor admission; never dispatch through a legacy study."""

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "library_admission_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, F, W, CONTEXT, RAW = (load(n) for n in ("authority", "frozen", "worker", "context", "evidence"))
C = A.C
SCHEMA = "frozen-library-admission-job-v1"
ENGINES = ("cpu_arm64", "cpu_torch", "metal", "mps", "mlx")


def validate_job(job):
    C.require_execution_support()
    C.require(
        job["schema"] == SCHEMA and job["kind"] in ("width", "repeat_training", "checkpoint_tie"),
        "Unknown frozen admission job",
    )
    keys = {
        "schema",
        "kind",
        "manifest",
        "width",
        "output",
        "raw_store",
        "parent_pid",
        "raw_start_bytes",
    }
    if job["kind"] != "width":
        keys.add("engine")
        C.require(
            job["engine"] in (ENGINES if job["kind"] == "repeat_training" else ENGINES[:2]),
            "Unknown admission child engine",
        )
    if job["kind"] == "repeat_training":
        keys.add("operands")
    C.require(
        type(job["raw_start_bytes"]) is int and job["raw_start_bytes"] >= 0,
        "Invalid admission raw-store baseline",
    )
    C.require(
        set(job) == keys
        and type(job["width"]) is int
        and job["width"] in W.N.WIDTHS
        and type(job["parent_pid"]) is int
        and job["parent_pid"] > 0,
        "Frozen admission job fields or width differ",
    )
    return job


def child(context, engine, kind, job):
    output = context.path(kind + "-admission-child").with_suffix("")
    task = {k: job[k] for k in ("schema", "manifest", "width", "raw_store", "raw_start_bytes")}
    task.update(kind=kind, engine=engine, output=str(output), parent_pid=os.getpid())
    if kind == "repeat_training":
        from transformer.gpu_admission_checks import _npz

        operands = context.path("fresh-admission-input")
        _npz(
            operands,
            {**context.initial, "tokens": context.tokens[:3], "targets": context.targets[:3]},
        )
        task["operands"] = C.binding(operands)
    path, log = output.with_suffix(".job.json"), output.with_suffix(".log")
    C.write(path, task)
    argv = [
        context.runtime.inventory["layout"]["interpreter"],
        "-P",
        "-S",
        "-B",
        str(HERE / "admission_worker.py"),
        "--job",
        str(path),
        "--job-sha256",
        C.binding(path)["sha256"],
    ]
    with log.open("x") as stream:
        process = subprocess.Popen(
            argv,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
            start_new_session=True,
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                C.require(time.monotonic() - started < 3600, "Admission child time budget reached")
                C.require(
                    shutil.disk_usage(output.parent).free > C.MIN_FREE_BYTES,
                    "Admission disk reserve reached",
                )
                time.sleep(2)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    C.require(process.returncode == 0, "Fresh admission child failed; retain its evidence")
    result_path = output / "result.json"
    result = C.read(result_path)
    C.require(
        result["complete"] is True
        and result["integrity_after"] is True
        and result["pid"] == process.pid
        and result["parent_pid"] == os.getpid()
        and result["job"] == C.binding(path)
        and result["formal_fits"] == 0
        and result["admission_stream_opened"] is True,
        "Foreign/incomplete admission child",
    )
    context.child_proofs.append(
        dict(
            receipt=C.binding(result_path),
            job=C.binding(path),
            pid=process.pid,
            returncode=process.returncode,
            argv=argv,
            log=C.binding(log),
        )
    )
    return result["outputs"]


def width(job, runtime, output, manifest):
    import numpy as np

    def fresh(context, engine):
        result = child(context, engine, "repeat_training", job)
        C.bound(result["state"])
        with Path(result["state"]["path"]).open("rb") as stream:
            return RAW.decode_arrays(stream, ("parameters", "m", "v", "step", "logits"))

    def checkpoint(context, engine):
        result = child(context, engine, "checkpoint_tie", job)
        C.bound(result["observations"])
        with Path(result["observations"]["path"]).open("rb") as stream:
            values = RAW.decode_arrays(stream, ("actual", "expected"))
        return values["actual"], values["expected"]

    context = CONTEXT.create(
        runtime,
        output,
        job["width"],
        manifest["bindings"]["native_assets"],
        job["raw_store"],
        fresh_training=fresh,
        checkpoint_tie=checkpoint,
        admission=job["manifest"],
    )
    context.child_proofs = []
    inventory = [
        s for s in C.comparison()["gate_inventory"] if int(s["id"].split("/")[1]) == job["width"]
    ]
    rows = []
    try:
        for specification in inventory:
            size = (
                int(
                    subprocess.check_output(
                        ["/usr/bin/du", "-sk", job["raw_store"]], text=True
                    ).split()[0]
                )
                * 1024
            )
            C.require(
                size - job["raw_start_bytes"] < C.MAX_RAW_GROWTH_BYTES,
                "Admission raw-store growth budget reached",
            )
            engine, _, family, *suffix = specification["id"].split("/")
            first_cpu = len(context.cpu_operations)
            try:
                pairs = context.produce(engine, family, "/".join(suffix))
            except Exception as error:
                C.require(
                    specification["family"] == "diagnostic_precision"
                    and specification["blocking"] is False,
                    "Blocking admission operation failed",
                )
                failed = context.backends.pop(engine, None)
                evidence = dict(
                    gate=specification["id"],
                    exception_type=type(error).__name__,
                    message=str(error) or repr(error),
                    traceback=traceback.format_exc(),
                )
                if failed is not None:
                    try:
                        failed.close()
                    except Exception as cleanup:
                        evidence["cleanup_error"] = repr(cleanup)
                path = output / "errors" / (specification["id"].replace("/", "--") + ".json")
                C.write(path, evidence)
                row = dict(
                    id=specification["id"],
                    status="runtime_error",
                    members={},
                    decisions={},
                    passed=False,
                    error=C.binding(path),
                )
            else:
                C.require(
                    set(pairs) == set(specification["members"]),
                    "Admission member denominator differs",
                )
                row = dict(id=specification["id"], members={}, decisions={})
                for name, (actual, expected) in pairs.items():
                    budget = np.asarray(actual).nbytes + np.asarray(expected).nbytes + 8 * 1024**2
                    C.require(
                        shutil.disk_usage(output).free > C.MIN_FREE_BYTES + budget,
                        "Preserve admission raw pair and disk reserve",
                    )
                    row["decisions"][name] = RAW.decision(
                        actual, expected, specification["members"][name]
                    )
                    row["members"][name] = RAW.save_pair(
                        actual, expected, job["raw_store"], runtime.roles["math"]["root"]
                    )
                row["passed"] = all(d["passed"] for d in row["decisions"].values())
            row["cpu_operations"] = context.cpu_operations[first_cpu:]
            C.write(output / "groups" / (row["id"].replace("/", "--") + ".json"), row)
            rows.append(row)
        # Diagnostic errors may retire a backend. Re-authenticate the selected
        # factories before final origins without turning a diagnostic into a
        # missing-capability failure or waiving any earlier blocking result.
        selected_backends = {e: context.backend(e) for e in ("metal", "mps", "mlx")}
        return dict(
            groups=rows,
            expected_inventory=inventory,
            raw_store=job["raw_store"],
            blocking_passed=all(
                r["passed"] for r, s in zip(rows, inventory, strict=True) if s["blocking"]
            ),
            child_proofs=context.child_proofs,
            capabilities={
                e: W.E.validate_capabilities(e, b.capabilities())
                for e, b in selected_backends.items()
            },
            metal_library_origin=selected_backends["metal"]._selected_native_origin,
        )
    except BaseException:
        C.write(
            output / "admission-stopped.json",
            dict(
                completed_groups=rows,
                child_proofs=context.child_proofs,
                cpu_operations=context.cpu_operations,
                error=traceback.format_exc(),
                formal_fits=0,
                admission_stream_opened=True,
            ),
        )
        raise
    finally:
        context.close()


def execute(path, expected):
    C.require_execution_support()
    job = validate_job(C.read(path, expected))
    C.require(
        job["parent_pid"] == os.getppid()
        and os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(job["width"]),
        "Admission process/width differs",
    )
    C.bound(job["manifest"])
    manifest = F.read(job["manifest"]["path"], job["manifest"]["sha256"])
    bindings = manifest["bindings"]
    sys.pycache_prefix = str(Path(job["output"]) / "unused-bytecode-cache")
    runtime = A.activate(bindings["roles"], bindings["capture"])
    derived = dict(
        job,
        roles=runtime.roles,
        capture=runtime.capture_binding,
        native_assets=bindings["native_assets"],
    )
    output = W.protected_output(derived, runtime)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir()
    record = dict(
        schema="frozen-library-admission-worker-v1",
        job=C.binding(path),
        manifest=job["manifest"],
        pid=os.getpid(),
        parent_pid=os.getppid(),
        started_unix=time.time(),
        complete=False,
        integrity_after=False,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=True,
    )
    origin_engine = "mlx" if job["kind"] == "width" else job["engine"]
    try:
        record["origins_before"] = runtime.origins(origin_engine)
        if job["kind"] == "width":
            result = width(job, runtime, output, manifest)
        elif job["kind"] == "repeat_training":
            result = W.repeat(derived, runtime, output)
        else:
            result = W.tie(derived, runtime, output, admission=job["manifest"])
        record["outputs"] = result
        record["origins_after"] = runtime.origins(origin_engine)
        runtime.recheck()
        C.require(C.binding(path)["sha256"] == expected, "Admission job changed")
        record.update(complete=True, integrity_after=True)
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        record["ended_unix"] = time.time()
        C.write(output / "result.json", record)
    print(json.dumps(dict(complete=record["complete"], integrity_after=record["integrity_after"])))
    return 0 if record["complete"] and record["integrity_after"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.job, args.job_sha256))
