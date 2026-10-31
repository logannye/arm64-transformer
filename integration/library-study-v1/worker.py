"""Authenticated development qualification; formal admission remains closed."""

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
    spec = importlib.util.spec_from_file_location("library_worker_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A, E, N, RAW, CONTEXT = (
    load(n) for n in ("authority", "execution", "native_assets", "evidence", "context")
)
C = A.C
SCHEMA = "frozen-library-development-job-v1"
KINDS = {"native_metal_gates", "repeat_training", "checkpoint_tie"}


def validate_job(job):
    C.require(
        type(job) is dict and job.get("schema") == SCHEMA and job.get("kind") in KINDS,
        "Unknown development job",
    )
    keys = {
        "schema",
        "kind",
        "roles",
        "capture",
        "width",
        "native_assets",
        "raw_store",
        "output",
        "parent_pid",
    }
    if job["kind"] != "native_metal_gates":
        keys.add("engine")
        engines = (
            ("cpu_arm64", "cpu_torch")
            if job["kind"] == "checkpoint_tie"
            else ("cpu_arm64", "cpu_torch", "metal", "mps", "mlx")
        )
        C.require(job.get("engine") in engines, "Unknown selected engine")
    if job["kind"] == "repeat_training":
        keys.add("operands")
    C.require(set(job) == keys, "Development job fields differ; formal operands are unavailable")
    C.require(
        type(job["width"]) is int
        and job["width"] in N.WIDTHS
        and type(job["parent_pid"]) is int
        and job["parent_pid"] > 0,
        "Invalid development width or parent process",
    )
    return job


def protected_output(job, runtime):
    output = Path(job["output"])
    C.require(
        output.is_absolute()
        and output.resolve() == output
        and not output.exists()
        and not output.is_symlink(),
        "Fresh canonical worker output required",
    )
    protected = [Path(r["root"]) for r in runtime.roles.values()]
    protected += [Path(p) for p in runtime.inventory["layout"]["scan_roots"]]
    protected += [Path(job["raw_store"])]
    C.require(
        all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
        "Worker output overlaps a protected source, runtime or raw store",
    )
    C.require(
        shutil.disk_usage(output.parent).free > C.MIN_FREE_BYTES + 2 * 1024**3,
        "Preserve disk reserve and bounded worker outputs",
    )
    return output


def supplied_arrays(row):
    import numpy as np

    C.bound(row)
    C.require(row["bytes"] <= 256 * 1024**2, "Oversized repeat operands")
    with Path(row["path"]).open("rb") as stream:
        values = RAW.decode_arrays(
            stream, ("parameters", "m", "v", "step", "tokens", "targets"), max_bytes=256 * 1024**2
        )
    from transformer import spec

    for name in ("parameters", "m", "v"):
        v = values[name]
        C.require(
            v.dtype == np.float32 and v.shape == (spec.PARAM_COUNT,) and np.isfinite(v).all(),
            "Wrong repeat state",
        )
    C.require(
        values["step"].dtype == np.int64
        and values["step"].shape == ()
        and int(values["step"]) == 0
        and (values["v"] >= 0).all(),
        "Wrong repeat counter/moment",
    )
    for name in ("tokens", "targets"):
        v = values[name]
        C.require(
            v.dtype == np.int32 and v.shape == (3, 32) and ((0 <= v) & (v < 128)).all(),
            "Wrong repeat batch",
        )
    C.bound(row)
    return values


def training_graph(backend, output, runtime):
    graph = backend._training_graph
    C.require(
        isinstance(graph, str) and "Compiled" in graph and backend._training_python_traces == 1,
        "Actual compiled training graph required",
    )
    path = Path(runtime.roles["math"]["root"]) / "integration/trained-precision/graph_evidence.py"
    spec = importlib.util.spec_from_file_location("library_worker_graph_transport", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = output / "compiled-training"
    destination.mkdir()
    transport = module.write(destination, "gradient", graph)
    return dict(
        graph=C.binding(destination / transport["path"]),
        transport=transport,
        python_trace_count=backend._training_python_traces,
    )


def repeat(job, runtime, output):
    import numpy as np

    from transformer.gpu_study_cpu_admission import _compute

    values = supplied_arrays(job["operands"])
    engine = job["engine"]
    state = {k: values[k].copy() for k in ("parameters", "m", "v", "step")}
    backend = None
    proof = {}
    try:
        if engine.startswith("cpu_"):
            native = (
                N.load(runtime, job["native_assets"], job["width"])
                if engine == "cpu_arm64"
                else None
            )
            for _ in range(6):
                result = _compute(
                    engine,
                    {
                        **state,
                        "tokens": values["tokens"],
                        "targets": values["targets"],
                        "mode": np.asarray("advance"),
                        "step": np.asarray(int(state["step"]) + 1, np.int64),
                    },
                    native=native,
                    library_api=engine == "cpu_torch",
                )
                state = {k: result[k] for k in state}
            probe = _compute(
                engine,
                {
                    **state,
                    "tokens": values["tokens"],
                    "targets": values["targets"],
                    "mode": np.asarray("forward_gradient"),
                },
                native=native,
                library_api=engine == "cpu_torch",
            )
            state["logits"] = probe["logits"]
        else:
            backend = E.create_backend(runtime, engine, state["parameters"])
            backend.load_state({**state, "step": int(state["step"])})
            batch = backend.upload_inputs(values["tokens"], values["targets"])
            for _ in range(6):
                backend.to_numpy(backend.train_step(batch))
                C.require(backend.check_finite(), "Nonfinite repeat training state")
            state = backend.export_state()
            state["logits"] = backend.to_numpy(backend.forward(batch)).copy()
            proof["capabilities"] = E.validate_capabilities(engine, backend.capabilities())
            if engine == "metal":
                proof["metal_library_origin"] = backend._selected_native_origin
            if engine == "mlx":
                proof["compiled_training"] = training_graph(backend, output, runtime)
        C.require(int(state["step"]) == 6, "Wrong fresh-process training step count")
        with (output / "state.npz").open("xb") as stream:
            np.savez_compressed(stream, **state)
        proof["state"] = C.binding(output / "state.npz")
        return proof
    finally:
        if backend is not None:
            backend.close()


def tie(job, runtime, output):
    import numpy as np

    from transformer.gpu_study_audit import parameter_count, read_array
    from transformer.gpu_study_cpu_admission import write_development_tie_fixture

    engine, width = job["engine"], job["width"]
    C.require(engine in ("cpu_arm64", "cpu_torch"), "CPU tie engine required")
    fixture = write_development_tie_fixture(output / "fixture", width)
    run = output / "run"
    if engine == "cpu_torch":
        from transformer.python_train import run as cpu_fit

        cpu_fit(fixture, run, library_api=True)
    else:
        assets = N.verify(job["native_assets"], runtime.roles["math"])
        row = next(r for r in assets["widths"] if r["width"] == width)
        executable = row["selected"]["transformer-arm64"]
        C.bound(executable)
        with (output / "native.log").open("x") as log:
            subprocess.run(
                [executable["path"], str(fixture), str(run)],
                check=True,
                cwd=runtime.roles["math"]["root"],
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        C.bound(executable)
        from transformer.backend import read_backend, validate_runtime_backend

        metadata = read_backend(run / "backend.u64", policy="auto")
        validate_runtime_backend(
            metadata,
            dict(
                epochs=2,
                train_sequences=16,
                validation_sequences=16,
                test_sequences=16,
                width=width,
            ),
            policy="auto",
            precision_policy=row["policy"],
        )
    history = read_array(run / "history.f64", "<f8", (2, 3))
    performance = read_array(run / "performance.f64", "<f8", (8,))
    zeros = [
        bool((read_array(run / name, "<f4", (parameter_count(width),)) == 0).all())
        for name in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32")
    ]
    actual = np.array(
        [performance[5], performance[6], int((history[:, 1] == history[0, 1]).all()), *zeros],
        np.float64,
    )
    expected = np.array([1, 2, 1, 1, 1, 1, 1], np.float64)
    with (output / "observations.npz").open("xb") as stream:
        np.savez_compressed(stream, actual=actual, expected=expected)
    return dict(
        observations=C.binding(output / "observations.npz"),
        files=[C.binding(p) for p in sorted(output.rglob("*")) if p.is_file()],
    )


def child(context, engine, kind, job):
    output = context.path(kind + "-child").with_suffix("")
    child_job = {
        k: job[k] for k in ("schema", "roles", "capture", "width", "native_assets", "raw_store")
    }
    child_job.update(kind=kind, engine=engine, output=str(output), parent_pid=os.getpid())
    if kind == "repeat_training":
        from transformer.gpu_admission_checks import _npz

        operands = context.path("fresh-repeat-input")
        _npz(
            operands,
            {**context.initial, "tokens": context.tokens[:3], "targets": context.targets[:3]},
        )
        child_job["operands"] = C.binding(operands)
    path = output.with_suffix(".job.json")
    C.write(path, child_job)
    log_path = output.with_suffix(".log")
    argv = [
        context.runtime.inventory["layout"]["interpreter"],
        "-P",
        "-S",
        "-B",
        str(HERE / "worker.py"),
        "--job",
        str(path),
        "--job-sha256",
        C.binding(path)["sha256"],
    ]
    with log_path.open("x") as log:
        process = subprocess.Popen(
            argv,
            cwd=context.runtime.roles["math"]["root"],
            env=dict(os.environ),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                C.require(
                    time.monotonic() - started < 3600, "Development child time budget reached"
                )
                C.require(
                    shutil.disk_usage(output.parent).free > C.MIN_FREE_BYTES,
                    "Development child disk reserve reached",
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
    C.require(process.returncode == 0, "Development child failed; retain its log and receipt")
    receipt = C.read(output / "result.json")
    C.require(
        receipt["complete"] is True
        and receipt["integrity_after"] is True
        and receipt["pid"] == process.pid
        and receipt["parent_pid"] == os.getpid()
        and receipt["pid"] != os.getpid()
        and receipt["job"] == C.binding(path),
        "Development child identity or integrity differs",
    )
    context.child_proofs.append(
        dict(
            receipt=C.binding(output / "result.json"),
            job=C.binding(path),
            pid=process.pid,
            returncode=process.returncode,
            argv=argv,
            log=C.binding(log_path),
        )
    )
    return receipt["outputs"]


def gates(job, runtime, output):
    import numpy as np

    def fresh(context, engine):
        record = child(context, engine, "repeat_training", job)
        C.bound(record["state"])
        with np.load(record["state"]["path"], allow_pickle=False) as arrays:
            return {k: arrays[k].copy() for k in arrays.files}

    def checkpoint(context, engine):
        record = child(context, engine, "checkpoint_tie", job)
        C.bound(record["observations"])
        with np.load(record["observations"]["path"], allow_pickle=False) as arrays:
            return arrays["actual"].copy(), arrays["expected"].copy()

    context = CONTEXT.create(
        runtime,
        output,
        job["width"],
        job["native_assets"],
        job["raw_store"],
        fresh_training=fresh,
        checkpoint_tie=checkpoint,
    )
    context.child_proofs = []
    inventory = [
        g
        for g in C.comparison()["gate_inventory"]
        if g["id"].split("/")[:2]
        in [[e, str(job["width"])] for e in ("cpu_arm64", "metal", "oracle")]
    ]
    rows = []
    try:
        for specification in inventory:
            C.require(
                shutil.disk_usage(output).free > C.MIN_FREE_BYTES, "Gate disk reserve reached"
            )
            engine, width, family, *suffix = specification["id"].split("/")
            first_cpu = len(context.cpu_operations)
            pairs = context.produce(engine, family, "/".join(suffix))
            C.require(set(pairs) == set(specification["members"]), "Gate member inventory differs")
            row = dict(id=specification["id"], members={}, decisions={})
            for name, (actual, expected) in pairs.items():
                row["decisions"][name] = RAW.decision(
                    actual, expected, specification["members"][name]
                )
                budget = np.asarray(actual).nbytes + np.asarray(expected).nbytes + 8 * 1024**2
                C.require(
                    shutil.disk_usage(output).free > C.MIN_FREE_BYTES + budget,
                    "Preserve complete next raw pair and disk reserve",
                )
                row["members"][name] = RAW.save_pair(
                    actual, expected, job["raw_store"], runtime.roles["math"]["root"]
                )
            row["passed"] = all(d["passed"] for d in row["decisions"].values())
            row["cpu_operations"] = context.cpu_operations[first_cpu:]
            C.write(output / "groups" / (row["id"].replace("/", "--") + ".json"), row)
            rows.append(row)
            C.require(
                row["passed"] or not specification["blocking"],
                "Blocking development qualification failed; preserve raw arrays",
            )
        return dict(
            groups=rows,
            expected_inventory=inventory,
            child_proofs=context.child_proofs,
            capabilities={
                e: E.validate_capabilities(e, b.capabilities()) for e, b in context.backends.items()
            },
            metal_library_origin=context.backends["metal"]._selected_native_origin,
            raw_store=job["raw_store"],
        )
    except BaseException:
        C.write(
            output / "qualification-stopped.json",
            dict(
                completed_groups=rows,
                child_proofs=context.child_proofs,
                cpu_operations=context.cpu_operations,
                error=traceback.format_exc(),
            ),
        )
        raise
    finally:
        context.close()


def execute(path, expected):
    job = validate_job(C.read(path, expected))
    C.require(job["parent_pid"] == os.getppid(), "Development parent process differs")
    C.require(
        type(job["width"]) is int
        and job["width"] in N.WIDTHS
        and os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(job["width"]),
        "Select width first",
    )
    C.require(
        Path(job["roles"]["control"]["root"]) == HERE.parents[1], "Worker control origin differs"
    )
    sys.pycache_prefix = str(Path(job["output"]) / "unused-bytecode-cache")
    runtime = A.activate(job["roles"], job["capture"])
    output = protected_output(job, runtime)
    N.verify(job["native_assets"], runtime.roles["math"])
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir()
    receipt = dict(
        schema="frozen-library-development-worker-v1",
        job=C.binding(path),
        pid=os.getpid(),
        parent_pid=os.getppid(),
        started_unix=time.time(),
        complete=False,
        integrity_after=False,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
    )
    try:
        receipt["origins_before"] = runtime.origins(job.get("engine", "metal"))
        function = dict(native_metal_gates=gates, repeat_training=repeat, checkpoint_tie=tie)[
            job["kind"]
        ]
        receipt["outputs"] = function(job, runtime, output)
        receipt["origins_after"] = runtime.origins(job.get("engine", "metal"))
        runtime.recheck()
        C.require(C.binding(path)["sha256"] == expected, "Worker job changed")
        receipt.update(complete=True, integrity_after=True)
    except BaseException:
        receipt["error"] = traceback.format_exc()
    finally:
        receipt.update(ended_unix=time.time(), free_bytes=shutil.disk_usage(output).free)
        C.write(output / "result.json", receipt)
    print(
        json.dumps(dict(complete=receipt["complete"], integrity_after=receipt["integrity_after"]))
    )
    return 0 if receipt["complete"] and receipt["integrity_after"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.job, args.job_sha256))
