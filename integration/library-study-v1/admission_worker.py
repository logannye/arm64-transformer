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
C = A.C
SCHEMA = "frozen-library-admission-job-v1"
ENGINES = ("cpu_arm64", "cpu_torch", "metal", "mps", "mlx")
DEVICE_REPAIR_ID = "mlx/32/control/device_execution"
DEVICE_REPAIR_RECEIPT_SHA256 = "7345e16db79fe944bee4a27cd7e5bb354b152d818ad66df8a567048a74f5e477"


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
        C.write(output / "result.json", record)
    print(json.dumps(dict(complete=record["complete"], integrity_after=record["integrity_after"])))
    return 0 if record["complete"] and record["integrity_after"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.job, args.job_sha256))
