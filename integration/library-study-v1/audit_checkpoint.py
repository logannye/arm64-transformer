"""Independent raw audit of every engine at one saved best or last checkpoint."""

import argparse
import importlib.util
import io
import itertools
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("library_postrun_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R, RAW, ORIGINS, E, CAM = (
    load(n) for n in ("checkpoint_replay", "evidence", "audit", "execution", "campaign")
)
C, F, AD = R.C, R.F, R.AD


def measured_arrays(cell, directory):
    """Inspect final state, selection and timing without a legacy manifest route."""
    import numpy as np

    from transformer.gpu_study import required_artifacts
    from transformer.gpu_study_audit import parameter_count, read_array

    directory = Path(directory)
    C.require(
        all((directory / n).is_file() for n in required_artifacts(cell["engine"])),
        "Missing measured artifact",
    )
    for name in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32"):
        state = read_array(directory / name, "<f4", (parameter_count(cell["width"]),))
        C.require(name != "adam_v.f32" or (state >= 0).all(), "Negative second moment")
    history = read_array(directory / "history.f64", "<f8", (8, 3))
    timings = read_array(directory / "timings.f64", "<f8", (100,))
    logits = read_array(directory / "test_logits.f32", "<f4", (256, 32, 128))
    C.require((history >= 0).all() and (timings > 0).all(), "Invalid loss/timing")
    best_epoch = int(history[:, 1].argmin()) + 1
    if cell["engine"].startswith("cpu_"):
        performance = read_array(directory / "performance.f64", "<f8", (8,))
        seconds, epoch, steps = performance[0], performance[5], performance[6]
        C.require(
            0 < performance[2] <= performance[3] <= performance[4] and performance[7] >= seconds,
            "Invalid CPU memory or wall clock counters",
        )
        memory = dict(
            peak_rss_before=float(performance[2]),
            peak_rss_after=float(performance[3]),
            peak_rss_final=float(performance[4]),
        )
    else:
        runtime = C.read(directory / "runtime.json")
        C.require(
            runtime["scope"] == "formal_measurement"
            and runtime["status"] == "completed"
            and runtime["engine"] == cell["engine"]
            and runtime["warmup_restored_exactly"] is True
            and runtime["selected_validation_replay_exact"] is True
            and runtime["inference_replay_exact"] is True,
            "Incomplete measured GPU fit",
        )
        E.validate_capabilities(cell["engine"], runtime["capabilities"])
        seconds, epoch, steps = runtime["training_seconds"], runtime["best_epoch"], runtime["steps"]
        memory = {
            k: runtime[k]
            for k in (
                "memory_before",
                "memory_after_training",
                "memory_final",
                "process_peak_rss_bytes",
            )
        }
    C.require(epoch == best_epoch and steps == 1024, "Incorrect selection or update count")
    C.require(seconds >= float(history[:, 2].sum()) > 0, "Training timer excludes epochs")
    return dict(
        history=history,
        logits=logits,
        best_epoch=best_epoch,
        training_seconds=float(seconds),
        latency_p50=float(np.median(timings)),
        latency_p95=float(np.quantile(timings, 0.95)),
        memory=memory,
    )


def load_replay(binding, job, manifest, capture, inventory):
    record = F.PLAN.load(binding)
    source = F.PLAN.load(record["job"])
    expected = {
        k: job[k] for k in ("manifest", "admission", "cell_receipt", "checkpoint", "raw_store")
    }
    C.require(
        record["schema"] == "frozen-library-checkpoint-replay-v1"
        and record["complete"] is True
        and all(source[k] == value for k, value in expected.items())
        and source["engine"] == record["engine"]
        and record["manifest"] == job["manifest"]
        and record["cell_receipt"] == job["cell_receipt"]
        and record["checkpoint"] == job["checkpoint"]
        and record["raw_store"] == job["raw_store"]
        and Path(record["job"]["path"]).parent == Path(job["replay_root"])
        and Path(source["output"]) / "receipt.json" == Path(binding["path"])
        and type(record["pid"]) is int
        and record["pid"] > 0
        and record["pid"] != record["parent_pid"] == source["parent_pid"],
        "Foreign checkpoint replay",
    )
    roles = manifest["bindings"]["roles"]
    engine = record["engine"]
    if engine == "oracle":
        C.require(
            record["auditor_project_origins"] and record["oracle_versions"],
            "Missing independent oracle provenance",
        )
        for name, row in record["auditor_project_origins"].items():
            C.require(
                name == "transformer" or name.startswith("transformer."), "Foreign oracle module"
            )
            path = C.bound(row)
            C.require(
                path.is_relative_to(roles["math"]["root"])
                and roles["math"]["files"].get(path.relative_to(roles["math"]["root"]).as_posix())
                == row["sha256"],
                "Independent oracle source changed",
            )
        names = ("validation", "test", "validation_torch_fp64", "test_torch_fp64", "b1")
    else:
        for phase in ("before", "after"):
            ORIGINS.validate_origins(record["origins_" + phase], roles, capture, inventory, engine)
        if engine in ("metal", "mps", "mlx"):
            E.validate_capabilities(engine, record["capabilities"])
        if engine == "metal":
            C.require(
                record["metal_library_origin"]
                == C.binding(Path(roles["math"]["root"]) / R.A.METAL_LIBRARY),
                "Replay Metal host library differs",
            )
        if engine == "cpu_arm64":
            C.require(
                record["native_assets"] == manifest["bindings"]["native_assets"],
                "Native replay selection differs",
            )
        names = R.NAMES
    with io.BytesIO() as stream:
        raw = record["raw"]
        C.require(
            type(raw["bytes"]) is int and 0 < raw["bytes"] <= RAW.MAX_BYTES,
            "Oversized checkpoint arrays",
        )
        recipe = RAW.storage(roles["math"]["root"]).reconstruct(
            job["raw_store"], raw["sha256"], stream, max_bytes=RAW.MAX_BYTES
        )
        C.require(recipe["bytes"] == raw["bytes"], "Checkpoint raw byte count differs")
        stream.seek(0)
        return RAW.decode_arrays(stream, names)


def numerical(arrays, info, rows, checkpoint, own_engine, directory):
    import numpy as np

    from transformer.gpu_study_audit import compare, prediction_delta, quality, read_array

    oracle, comparisons, compatibility, differences = {}, {}, {}, {}
    expected_dtype = {e: np.float32 for e in R.ENGINES} | dict(oracle=np.float64)
    for split in ("validation", "test"):
        for engine in (*R.ENGINES, "oracle"):
            value = arrays[engine][split]
            C.require(
                value.dtype == expected_dtype[engine]
                and value.shape == (256, 32, 128)
                and np.isfinite(value).all(),
                "Wrong replay dtype/shape/finiteness",
            )
        gold = arrays["oracle"][split]
        double = arrays["oracle"][split + "_torch_fp64"]
        C.require(
            double.dtype == np.float64 and double.shape == (2, 32, 128), "Wrong double cross-check"
        )
        oracle[split] = compare(double, gold[[0, -1]], 1e-10, 1e-10)
        for engine in R.ENGINES:
            comparisons[f"{engine}/{split}"] = compare(arrays[engine][split], gold)
            differences[f"{engine}/{split}"] = prediction_delta(
                arrays[engine][split], gold, rows[split][:, 1:]
            )
            loss = arrays[engine][split + "_nll"]
            C.require(
                loss.dtype == np.float64 and loss.shape == () and np.isfinite(loss) and loss >= 0,
                "Invalid own-engine replay NLL",
            )
        for left, right in itertools.combinations(R.ENGINES, 2):
            compatibility[f"{left}/{right}/{split}"] = compare(
                arrays[left][split], arrays[right][split]
            )
    for engine in (*R.ENGINES, "oracle"):
        b1 = arrays[engine]["b1"]
        C.require(
            b1.dtype == expected_dtype[engine] and b1.shape == (1, 32, 128), "Wrong B1 replay"
        )
        comparisons[engine + "/b1"] = compare(b1, arrays["oracle"]["test"][:1])
    own = arrays[own_engine]
    epoch = info["best_epoch"] - 1 if checkpoint == "best.f32" else 7
    integrity = dict(
        own_validation_history_exact=float(own["validation_nll"])
        == float(info["history"][epoch, 1])
    )
    if checkpoint == "best.f32":
        integrity["own_test_replay_exact"] = own["test"].tobytes() == info["logits"].tobytes()
        if own_engine in ("metal", "mps", "mlx"):
            saved = read_array(Path(directory) / "inference_logits.f32", "<f4", (1, 32, 128))
            integrity["own_b1_replay_exact"] = own["b1"].tobytes() == saved.tobytes()
    return dict(
        oracle=oracle,
        fp64=comparisons,
        compatibility=compatibility,
        prediction_differences=differences,
        integrity=integrity,
        quality=quality(own["test"], rows["test"][:, 1:]),
        passed=all(
            c["passed"] for c in [*oracle.values(), *comparisons.values(), *compatibility.values()]
        )
        and all(integrity.values()),
    )


def execute(path, digest):
    C.require_execution_support()
    C.require(
        sys.flags.isolated and sys.flags.dont_write_bytecode,
        "Use independent checkpoint auditor -I -B",
    )
    C.require(
        not any(
            n.split(".", 1)[0] in {"torch", "mlx", "numpy", "transformer"} for n in sys.modules
        ),
        "Fresh independent audit required",
    )
    job = C.read(path, digest)
    C.require(
        set(job)
        == {
            "schema",
            "manifest",
            "admission",
            "cell_receipt",
            "checkpoint",
            "replays",
            "replay_root",
            "raw_store",
        }
        and job["schema"] == "frozen-library-checkpoint-audit-job-v1"
        and job["checkpoint"] in ("best.f32", "last.f32")
        and list(job["replays"]) == sorted((*R.ENGINES, "oracle")),
        "Unknown checkpoint audit job",
    )
    manifest = F.read(job["manifest"]["path"], job["manifest"]["sha256"])
    AD.verify(manifest, job["manifest"], job["admission"])
    receipt, proof = R.cell_receipt(
        job["cell_receipt"], manifest, job["manifest"], job["admission"]
    )
    cell, roles = receipt["cell"], manifest["bindings"]["roles"]
    C.require(
        os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(cell["width"]), "Select audit width first"
    )
    capture = F.PLAN.load(manifest["bindings"]["capture"])
    inventory = F.PLAN.load(capture["runtime_inventory"])
    sys.path.insert(0, roles["math"]["root"])
    directory = Path(job["cell_receipt"]["path"]).parent
    for phase in ("before", "after"):
        ORIGINS.validate_origins(
            proof["origins_" + phase], roles, capture, inventory, cell["engine"]
        )
    if cell["engine"] == "cpu_arm64":
        from transformer.backend import read_backend, validate_runtime_backend

        assets = R.W.N.verify(manifest["bindings"]["native_assets"], roles["math"])
        selected = next(r for r in assets["widths"] if r["width"] == cell["width"])
        C.require(
            proof["native_executable"] == selected["selected"]["transformer-arm64"],
            "Measured native executable differs",
        )
        validate_runtime_backend(
            read_backend(directory / "backend.u64", policy="auto"),
            dict(
                width=cell["width"],
                epochs=8,
                train_sequences=2048,
                validation_sequences=256,
                test_sequences=256,
            ),
            policy="auto",
            precision_policy=selected["policy"],
        )
    info = measured_arrays(cell, directory)
    if cell["engine"] == "mlx":
        runtime = C.read(directory / "runtime.json")
        graph = runtime["compiled_training"]
        C.bound(graph["graph"])
        spec = importlib.util.spec_from_file_location(
            "postrun_training_graph",
            Path(roles["math"]["root"]) / "integration/trained-precision/graph_evidence.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        text = module.read(Path(graph["graph"]["path"]), graph["transport"])
        C.require(
            graph["python_trace_count"] == 1 and "Compiled" in text,
            "Missing actual measured MLX training graph",
        )
    fixture = next(
        r
        for r in F.PLAN.load(manifest["bindings"]["fixtures"])["fixtures"]
        if (r["width"], r["seed"]) == (cell["width"], cell["seed"])
    )
    from transformer.prepare import load_fixture

    arrays = {
        engine: load_replay(binding, job, manifest, capture, inventory)
        for engine, binding in job["replays"].items()
    }
    result = numerical(
        arrays,
        info,
        load_fixture(fixture["directory"]),
        job["checkpoint"],
        cell["engine"],
        directory,
    )
    F.verify(manifest)
    R.cell_receipt(job["cell_receipt"], manifest, job["manifest"], job["admission"])
    C.require(C.binding(path)["sha256"] == digest, "Postrun audit job changed")
    result.update(
        schema="frozen-library-checkpoint-audit-v1",
        job=C.binding(path),
        manifest=job["manifest"],
        cell=cell,
        checkpoint=job["checkpoint"],
        evidence_valid=True,
        auditor_project_origins=ORIGINS.project_origins(roles["math"], sys.modules),
        **{k: info[k] for k in ("training_seconds", "latency_p50", "latency_p95", "memory")},
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    args = parser.parse_args()
    execute(args.job, args.job_sha256)
