"""Complete frozen successor admission; never dispatch through a legacy study."""

import argparse
import importlib.util
import json
import os
import resource
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
IO = load("artifact_io")
C = A.C
SCHEMA = "frozen-library-admission-job-v1"
ENGINES = ("cpu_arm64", "cpu_torch", "metal", "mps", "mlx")
DEVICE_REPAIR_ID = "mlx/32/control/device_execution"
DEVICE_REPAIR_RECEIPT_SHA256 = "7345e16db79fe944bee4a27cd7e5bb354b152d818ad66df8a567048a74f5e477"
CHILD_RECOVERY_SHA256 = "b9036ba939a4e053ab7059707c8ec3488b895fdf46785669944456b57695bbd1"


def child_recovery(job, manifest):
    """Reuse only the twelve guarded children of the recorded receipt-read failure."""
    binding = job["child_recovery"]
    C.require(binding["sha256"] == CHILD_RECOVERY_SHA256, "Unknown child recovery evidence")
    recovery = F.PLAN.load(binding)
    failed = F.PLAN.load(recovery["failed_parent"])
    stopped = F.PLAN.load(recovery["stopped"])
    previous = F.PLAN.load(recovery["manifest"])
    old, new = previous["bindings"], manifest["bindings"]
    producer = A.verify_role(old["roles"]["control"], control=True)
    C.require(
        job["kind"] == "width"
        and job["width"] == recovery["width"] == 256
        and recovery["evidence_valid"] is True
        and recovery["children_complete"] == 12
        and failed["complete"] is False
        and failed["integrity_after"] is False
        and failed["manifest"] == recovery["manifest"]
        and failed["pid"] == recovery["parent_pid"] == 28700
        and len(stopped["completed_groups"]) == 303
        and stopped["child_proofs"] == recovery["child_proofs"][:11]
        and "ValueError: JSON changed during read" in stopped["error"]
        and F.PLAN.load(failed["job"])["raw_store"] == job["raw_store"]
        and producer["revision"] == "6af9acb32fb766ae1e5fc2cdb2ff80b1f7f05a89"
        and old["roles"]["math"] == new["roles"]["math"]
        and all(old[k] == new[k] for k in ("capture", "fixtures", "native_assets", "hardware"))
        and all(
            previous[k] == manifest[k]
            for k in previous
            if k not in ("bindings", "source_roles", "frozen_utc")
        ),
        "Child recovery changed failure, scientific, runtime or source identity",
    )
    allowed = {
        "integration/library-study-v1/" + name
        for name in ("admission.py", "admission_worker.py", "audit_admission.py")
    }
    current = new["roles"]["control"]
    C.require(
        set(producer["files"]) == set(current["files"])
        and all(
            producer["files"][k] == current["files"][k]
            for k in producer["files"]
            if k not in allowed
        ),
        "Child recovery changed unrelated execution controls",
    )
    return recovery, previous


def read_child_receipt(path, anchor):
    """Retry output metadata only against the child's source-exact export hash."""
    C.require(anchor["path"] == str(path), "Foreign child receipt path")
    for attempt in range(3):
        try:
            value = C.read(path, anchor["sha256"])
            C.require(C.binding(path) == anchor, "Child receipt bytes differ from export")
            return value
        except ValueError as error:
            if str(error) not in ("JSON changed during read", "Bound file changed during read"):
                raise
            if attempt == 2:
                raise
    raise AssertionError("Unreachable receipt retry")


def isolated_diagnostic(gate):
    return gate in {f"mlx/{width}/framework_attention/random_t1" for width in (256, 512, 1024)}


def native_diagnostic_abort(proof, width):
    """Recognize only the reproduced SDPA view-offset assertion, never other exits."""
    C.bound(proof["log"])
    log = Path(proof["log"]["path"]).read_text()
    return proof["returncode"] == -signal.SIGABRT and (
        f"failed assertion `offset({16 * width}) must be < [buffer length]({12 * width}).'" in log
    )


def validate_job(job):
    C.require_execution_support()
    C.require(
        job["schema"] == SCHEMA
        and job["kind"] in ("width", "repeat_training", "checkpoint_tie", "diagnostic_probe"),
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
    if job["kind"] == "diagnostic_probe":
        keys.update(("engine", "gate"))
        C.require(
            job["engine"] == "mlx"
            and job["gate"] == f"mlx/{job['width']}/framework_attention/random_t1"
            and isolated_diagnostic(job["gate"]),
            "Only the diagnosed nonblocking MLX probe may be isolated",
        )
    elif job["kind"] != "width":
        keys.add("engine")
        C.require(
            job["engine"] in (ENGINES if job["kind"] == "repeat_training" else ENGINES[:2]),
            "Unknown admission child engine",
        )
    if job["kind"] == "repeat_training":
        keys.add("operands")
    if "device_repair" in job:
        keys.add("device_repair")
        C.require(
            job["kind"] == "width"
            and job["width"] == 32
            and job["device_repair"]["sha256"] == DEVICE_REPAIR_RECEIPT_SHA256,
            "Only the recorded width-32 device observation may be repaired",
        )
    if "child_recovery" in job:
        keys.add("child_recovery")
        C.require(
            job["kind"] == "width"
            and job["width"] == 256
            and "device_repair" not in job
            and job["child_recovery"]["sha256"] == CHILD_RECOVERY_SHA256,
            "Only the recorded width-256 child workers may be recovered",
        )
    C.require(
        type(job["raw_start_bytes"]) is int and job["raw_start_bytes"] >= 0,
        "Invalid admission raw-store baseline",
    )
    C.require(
        set(job) == keys
        and type(job["width"]) is int
        and job["width"] in C.WIDTHS
        and type(job["parent_pid"]) is int
        and job["parent_pid"] > 0,
        "Frozen admission job fields or width differ",
    )
    return job


def child(context, engine, kind, job, *, gate=None):
    if "child_recovery" in job and kind != "diagnostic_probe":
        for index, proof in enumerate(context.recovered_children):
            task = F.PLAN.load(proof["job"])
            if (task["engine"], task["kind"]) != (engine, kind):
                continue
            result = F.PLAN.load(proof["receipt"])
            C.require(
                result["complete"] is True
                and result["integrity_after"] is True
                and result["job"] == proof["job"],
                "Incomplete recovered admission child",
            )
            if kind == "repeat_training":
                expected = dict(
                    context.initial, tokens=context.tokens[:3], targets=context.targets[:3]
                )
                C.bound(task["operands"])
                with Path(task["operands"]["path"]).open("rb") as stream:
                    saved = RAW.decode_arrays(stream, tuple(expected))
                import numpy as np

                C.require(
                    all(
                        saved[k].dtype == np.asarray(v).dtype
                        and saved[k].shape == np.asarray(v).shape
                        and saved[k].tobytes() == np.asarray(v).tobytes()
                        for k, v in expected.items()
                    ),
                    "Recovered child operands differ from the frozen inputs",
                )
            context.child_proofs.append(context.recovered_children.pop(index))
            return result["outputs"]
        raise ValueError("Missing recovered admission child")
    output = context.path(kind + "-admission-child").with_suffix("")
    task = {k: job[k] for k in ("schema", "manifest", "width", "raw_store", "raw_start_bytes")}
    task.update(kind=kind, engine=engine, output=str(output), parent_pid=os.getpid())
    if kind == "diagnostic_probe":
        task["gate"] = gate
    if kind == "repeat_training":
        from transformer.gpu_admission_checks import _npz

        operands = context.path("fresh-admission-input")
        _npz(
            operands,
            {**context.initial, "tokens": context.tokens[:3], "targets": context.targets[:3]},
        )
        task["operands"] = C.binding(operands)
    validate_job(task)
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
    proof = dict(
        job=C.binding(path),
        pid=process.pid,
        returncode=process.returncode,
        argv=argv,
        log=C.binding(log),
    )
    if kind == "diagnostic_probe":
        proof["started"] = C.binding(output / "started.json")
        started_record = F.PLAN.load(proof["started"])
        C.require(
            started_record["job"] == proof["job"]
            and started_record["pid"] == process.pid
            and started_record["parent_pid"] == os.getpid()
            and started_record["diagnostic_gate"] == gate,
            "Diagnostic did not reach the authenticated primitive call",
        )
        if native_diagnostic_abort(proof, job["width"]):
            C.require(not (output / "result.json").exists(), "Aborted probe has a final receipt")
            return None, proof
    C.require(process.returncode == 0, "Fresh admission child failed; retain its evidence")
    result_path = output / "result.json"
    summary = json.loads(log.read_text().splitlines()[-1])
    result = read_child_receipt(result_path, summary["receipt"])
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
    proof["receipt"] = C.binding(result_path)
    if kind == "diagnostic_probe":
        return result["outputs"], proof
    context.child_proofs.append(proof)
    return result["outputs"]


def diagnostic(job, runtime, output, manifest, record):
    from transformer.gpu_admission_checks import _fixture_family, _npz

    context = CONTEXT.create(
        runtime,
        output,
        job["width"],
        manifest["bindings"]["native_assets"],
        job["raw_store"],
        fresh_training=None,
        checkpoint_tie=None,
        admission=job["manifest"],
    )
    try:
        case = _fixture_family(context, "attention")["random_t1"]
        context.backend("mlx")
        operands = output / "diagnostic-operands.npz"
        _npz(operands, {k: case[k] for k in ("qkv", "dy")})
        record.update(
            diagnostic_gate=job["gate"],
            diagnostic_operands=C.binding(operands),
            origins_before=runtime.origins("mlx"),
        )
        C.write(output / "started.json", record)
        try:
            pairs = context.produce("mlx", "framework_attention", "random_t1")
        except Exception as error:
            return dict(
                diagnostic_error=dict(
                    gate=job["gate"],
                    exception_type=type(error).__name__,
                    message=str(error) or repr(error),
                    traceback=traceback.format_exc(),
                )
            )
        # Storage errors and failed final guards remain blocking infrastructure failures.
        return dict(
            members={
                name: RAW.save_pair(
                    actual, expected, job["raw_store"], runtime.roles["math"]["root"]
                )
                for name, (actual, expected) in pairs.items()
            }
        )
    finally:
        context.close()


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
    if "child_recovery" in job:
        recovery, _ = child_recovery(job, manifest)
        context.recovered_children = list(recovery["child_proofs"])
    inventory = [
        s for s in C.comparison()["gate_inventory"] if int(s["id"].split("/")[1]) == job["width"]
    ]
    prior = None
    if "device_repair" in job:
        prior = F.PLAN.load(job["device_repair"])["outputs"]
        C.require(
            prior["expected_inventory"] == inventory and prior["raw_store"] == job["raw_store"],
            "Device repair cannot change inventory or raw store",
        )
    rows = []
    try:
        for index, specification in enumerate(inventory):
            if prior is not None and specification["id"] != DEVICE_REPAIR_ID:
                rows.append(prior["groups"][index])
                continue
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
            diagnostic_proof = None
            if isolated_diagnostic(specification["id"]):
                C.require(
                    specification["family"] == "diagnostic_precision"
                    and specification["blocking"] is False,
                    "Cannot isolate a required numerical gate as a diagnostic",
                )
                # Setup, disk, provenance and unexpected process errors must escape
                # the suppressible diagnostic-operation handler below.
                diagnosed, diagnostic_proof = child(
                    context, engine, "diagnostic_probe", job, gate=specification["id"]
                )
                if diagnosed is not None and "members" in diagnosed:
                    isolated_pairs = {
                        name: RAW.load_pair(bound, job["raw_store"], runtime.roles["math"]["root"])
                        for name, bound in diagnosed["members"].items()
                    }
            try:
                if diagnostic_proof is None:
                    pairs = context.produce(engine, family, "/".join(suffix))
                elif diagnosed is None:
                    raise RuntimeError("Isolated MLX SDPA native buffer-offset assertion (SIGABRT)")
                elif "diagnostic_error" in diagnosed:
                    raise RuntimeError(diagnosed["diagnostic_error"]["message"])
                else:
                    pairs = isolated_pairs
            except Exception as error:
                C.require(
                    specification["family"] == "diagnostic_precision"
                    and specification["blocking"] is False,
                    "Blocking admission operation failed",
                )
                failed = context.backends.pop(engine, None) if diagnostic_proof is None else None
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
            if diagnostic_proof is not None:
                row["diagnostic_child"] = diagnostic_proof
            C.write(output / "groups" / (row["id"].replace("/", "--") + ".json"), row)
            rows.append(row)
        # Diagnostic errors may retire a backend. Re-authenticate the selected
        # factories before final origins without turning a diagnostic into a
        # missing-capability failure or waiving any earlier blocking result.
        selected_backends = {
            e: context.backend(e)
            for e in (("mlx",) if prior is not None else ("metal", "mps", "mlx"))
        }
        capabilities = {} if prior is None else dict(prior["capabilities"])
        capabilities.update(
            {
                e: W.E.validate_capabilities(e, b.capabilities())
                for e, b in selected_backends.items()
            }
        )
        C.require(not getattr(context, "recovered_children", []), "Unused recovered children")
        return dict(
            groups=rows,
            expected_inventory=inventory,
            raw_store=job["raw_store"],
            blocking_passed=all(
                r["passed"] for r, s in zip(rows, inventory, strict=True) if s["blocking"]
            ),
            child_proofs=context.child_proofs if prior is None else prior["child_proofs"],
            capabilities=capabilities,
            metal_library_origin=(
                selected_backends["metal"]._selected_native_origin
                if prior is None
                else prior["metal_library_origin"]
            ),
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
    if job["kind"] == "diagnostic_probe":
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
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
        elif job["kind"] == "diagnostic_probe":
            result = diagnostic(job, runtime, output, manifest, record)
        else:
            result = W.tie(derived, runtime, output, admission=job["manifest"])
        record["outputs"] = result
        record["origins_after"] = runtime.origins(origin_engine)
        record["runtime_validation_final"] = runtime.finish()
        C.require(C.binding(path)["sha256"] == expected, "Admission job changed")
        record.update(complete=True, integrity_after=True)
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        record["ended_unix"] = time.time()
        receipt, export = IO.write_exact(
            output / "result.json",
            (json.dumps(record, indent=2, allow_nan=False) + "\n").encode(),
            C,
        )
    print(
        json.dumps(
            dict(
                complete=record["complete"],
                integrity_after=record["integrity_after"],
                receipt=receipt,
                receipt_export=export,
            )
        )
    )
    return 0 if record["complete"] and record["integrity_after"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.job, args.job_sha256))
