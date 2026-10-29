"""Exercise the shared GPU fit/output contract on training-only development data.

Formal measurement is deliberately disabled until the GPU protocol, admission,
repeatability classification and provenance machinery are frozen.
"""

import argparse
import hashlib
import importlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

from transformer.gpu_environment import ENVIRONMENT
from transformer.gpu_environment import configure_environment as _configure_environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    """Authenticate implementation bytes in an uncommitted development tree."""
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "transformer").glob("*.py"))
    paths += [
        path
        for path in (root / "native/metal").glob("*")
        if path.is_file() and path.suffix in (".mm", ".metal")
    ]
    paths.append(root / "native/metal/Makefile")
    return {str(path.relative_to(root)): digest(path) for path in sorted(paths) if path.is_file()}


def configure_environment():
    _configure_environment(ENVIRONMENT)


def state_equal(left, right):
    return left["step"] == right["step"] and all(
        left[name].dtype == right[name].dtype
        and left[name].shape == right[name].shape
        and left[name].tobytes() == right[name].tobytes()
        for name in ("parameters", "m", "v")
    )


def checked_state(backend, parameter_count):
    import numpy as np

    state = backend.export_state()
    if set(state) != {"parameters", "m", "v", "step"}:
        raise ValueError("Invalid backend state keys")
    for name in ("parameters", "m", "v"):
        array = state[name]
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.float32
            or array.shape != (parameter_count,)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"Invalid backend state: {name}")
    if type(state["step"]) is not int or state["step"] < 0 or np.any(state["v"] < 0):
        raise ValueError("Invalid optimizer step or second moment")
    return state


def require_finite(backend, value=None):
    if not backend.check_finite(value):
        raise FloatingPointError("Nonfinite GPU numerical state")


def scalar(backend, value):
    array = backend.to_numpy(value)
    if array.size != 1 or not math.isfinite(float(array.reshape(-1)[0])):
        raise FloatingPointError("Expected a finite GPU scalar")
    return float(array.reshape(-1)[0])


def evaluate(backend, dataset, order, count, batch, *, logits_file=None):
    import numpy as np

    total = 0.0
    for offset in range(0, count, batch):
        size = min(batch, count - offset)
        selected = backend.gather_batch(dataset, order, offset, size)
        loss, logits = backend.evaluate(selected)
        require_finite(backend, logits)
        require_finite(backend, loss)
        total += scalar(backend, loss) * size
        if logits_file is not None:
            values = backend.to_numpy(logits)
            if values.shape != (size, 32, 128) or values.dtype != np.float32:
                raise ValueError("Unexpected inference result shape or dtype")
            values.astype("<f4").tofile(logits_file)
    return total / count


def run_development(input_dir, output_dir, engine):
    numpy_preloaded = "numpy" in sys.modules
    configure_environment()
    import numpy as np

    from transformer.gpu_contract import GPU_CONTRACT, GPU_ENGINES

    if engine not in GPU_ENGINES:
        raise ValueError("Unknown GPU engine")
    module = importlib.import_module(f"transformer.gpu_{engine}")
    from transformer.gpu_inputs import load_fixture
    from transformer.spec import CONTEXT, PARAM_COUNT, model_metadata

    if Path(output_dir).is_symlink():
        raise ValueError("Output must be fresh and separate from input")
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be fresh and separate from input")
    inputs_before = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    fixture = load_fixture(source)
    if fixture["manifest"].get("gpu_development_fixture") is not True:
        raise ValueError(
            "Formal GPU measurement is not admitted; use training-only development inputs"
        )
    config = fixture["config"]
    if config["batch"] > 16:
        raise ValueError("GPU batch capacity is 16")
    output.mkdir(parents=True)
    backend = None
    failure = None
    receipt = {
        "status": "started",
        "scope": "development_only",
        "engine": engine,
        "contract": GPU_CONTRACT,
        "model": model_metadata(),
        "config": config,
        "environment": dict(ENVIRONMENT),
        "host_thread_environment_set_before_numpy_import": not numpy_preloaded,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_sha256": inputs_before,
        "source_sha256": source_manifest(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1], text=True
            )
        ),
    }
    try:
        setup_start = time.perf_counter()
        backend = module.Backend(fixture["initial"], max_batch=config["batch"])
        datasets = {
            name: backend.upload_dataset(fixture[name]) for name in ("train", "validation", "test")
        }
        schedule = backend.upload_indices(fixture["schedule"].reshape(-1))
        orders = {
            name: backend.upload_indices(np.arange(len(fixture[name]), dtype=np.int32))
            for name in ("validation", "test")
        }
        original = checked_state(backend, PARAM_COUNT)
        warm_start = time.perf_counter()
        batch_sizes = {
            min(config["batch"], config[f"{name}_sequences"])
            for name in ("train", "validation", "test")
        }
        batch_sizes.update(
            config[f"{name}_sequences"] % config["batch"]
            for name in ("train", "validation", "test")
        )
        # Evaluation may use a larger batch than a tiny development training
        # partition. Repeated valid training indices warm every shape safely.
        warm_order = backend.upload_indices(
            np.arange(max(batch_sizes), dtype=np.int32) % config["train_sequences"]
        )
        for size in sorted(batch_sizes - {0}):
            example = backend.gather_batch(datasets["train"], warm_order, 0, size)
            backend.train_step(example)
            require_finite(backend)
            loss, logits = backend.evaluate(example)
            require_finite(backend, loss)
            require_finite(backend, logits)
            scalar(backend, loss)
        backend.export_parameters()
        backend.synchronize()
        backend.load_state(original)
        if not state_equal(original, checked_state(backend, PARAM_COUNT)):
            raise RuntimeError("Warmup changed initial parameter/optimizer bytes")
        receipt["warmup"] = {
            "seconds": time.perf_counter() - warm_start,
            "state_restored_exactly": True,
        }
        receipt["setup_seconds"] = time.perf_counter() - setup_start
        receipt["setup_scope"] = (
            "Backend construction, device uploads and restored warmup; excludes "
            "framework import and fixture loading"
        )
        receipt["capabilities"] = backend.capabilities()
        receipt["memory_before"] = backend.memory_stats()
        history = np.empty((config["epochs"], 3), dtype="<f8")
        best, best_epoch, best_parameters = float("inf"), 0, None
        backend.synchronize()
        train_start = time.perf_counter()
        for epoch in range(config["epochs"]):
            epoch_start, total = time.perf_counter(), 0.0
            for start in range(0, config["train_sequences"], config["batch"]):
                size = min(config["batch"], config["train_sequences"] - start)
                selected = backend.gather_batch(
                    datasets["train"], schedule, epoch * config["train_sequences"] + start, size
                )
                loss = backend.train_step(selected)
                require_finite(backend)
                total += scalar(backend, loss) * size
            validation = evaluate(
                backend,
                datasets["validation"],
                orders["validation"],
                config["validation_sequences"],
                config["batch"],
            )
            if validation < best:
                best, best_epoch = validation, epoch + 1
                best_parameters = backend.export_parameters()
                if (
                    best_parameters.dtype != np.float32
                    or best_parameters.shape != (PARAM_COUNT,)
                    or not np.isfinite(best_parameters).all()
                ):
                    raise ValueError("Invalid selected checkpoint parameters")
                best_parameters.astype("<f4").tofile(output / "best.f32")
            backend.synchronize()
            history[epoch] = (
                total / config["train_sequences"],
                validation,
                time.perf_counter() - epoch_start,
            )
        backend.synchronize()
        receipt["training_seconds"] = time.perf_counter() - train_start
        receipt["memory_after_training"] = backend.memory_stats()
        history.tofile(output / "history.f64")
        last = checked_state(backend, PARAM_COUNT)
        expected_steps = config["epochs"] * math.ceil(config["train_sequences"] / config["batch"])
        if last["step"] != expected_steps:
            raise RuntimeError("Wrong Adam update count")
        for key, name in (("parameters", "last.f32"), ("m", "adam_m.f32"), ("v", "adam_v.f32")):
            last[key].astype("<f4").tofile(output / name)
        backend.load_state({**last, "parameters": best_parameters})
        with (output / "test_logits.f32").open("wb") as handle:
            receipt["test_nll"] = evaluate(
                backend,
                datasets["test"],
                orders["test"],
                config["test_sequences"],
                config["batch"],
                logits_file=handle,
            )
        repeated_validation = evaluate(
            backend,
            datasets["validation"],
            orders["validation"],
            config["validation_sequences"],
            config["batch"],
        )
        receipt["selected_validation_replay_exact"] = repeated_validation == best
        sample = backend.upload_inputs(fixture["test"][:1, :CONTEXT])
        for _ in range(20):
            inference_logits = backend.forward(sample)
        backend.synchronize()
        require_finite(backend, inference_logits)
        inference_before = backend.to_numpy(inference_logits).copy()
        if inference_before.shape != (1, CONTEXT, 128) or inference_before.dtype != np.float32:
            raise ValueError("Unexpected batch-one inference shape or dtype")
        timings = np.empty(100, dtype="<f8")
        for index in range(100):
            started = time.perf_counter()
            inference_logits = backend.forward(sample)
            backend.synchronize()
            timings[index] = time.perf_counter() - started
        require_finite(backend, inference_logits)
        inference_after = backend.to_numpy(inference_logits)
        if inference_after.shape != (1, CONTEXT, 128) or inference_after.dtype != np.float32:
            raise ValueError("Unexpected batch-one inference shape or dtype")
        receipt["inference_replay_exact"] = inference_before.tobytes() == inference_after.tobytes()
        inference_after.astype("<f4").tofile(output / "inference_logits.f32")
        timings.tofile(output / "timings.f64")
        receipt.update(
            status="completed",
            best_epoch=best_epoch,
            steps=last["step"],
            memory_final=backend.memory_stats(),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            process_peak_rss_scope=(
                "macOS process lifetime including imports, host fixtures/state snapshots "
                "and GPU runtime; distinct from device allocations"
            ),
        )
        receipt["inputs_unchanged"] = inputs_before == {
            p.name: digest(p) for p in source.iterdir() if p.is_file()
        }
        receipt["sources_unchanged"] = receipt["source_sha256"] == source_manifest()
        if not all(
            receipt[key]
            for key in (
                "inputs_unchanged",
                "sources_unchanged",
                "selected_validation_replay_exact",
                "inference_replay_exact",
            )
        ):
            receipt["status"] = "development_replay_failed"
    except Exception as exc:
        failure = exc
        receipt.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        cleanup_failure = None
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                cleanup_failure = exc
                receipt.update(
                    status="failed",
                    cleanup_error=repr(exc),
                    cleanup_traceback=traceback.format_exc(),
                )
        receipt["artifact_sha256"] = {
            p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "runtime.json"
        }
        (output / "runtime.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        if cleanup_failure is not None and failure is None:
            raise RuntimeError("GPU backend cleanup failed; see runtime.json") from cleanup_failure
    return receipt


def main():
    from transformer.gpu_contract import GPU_ENGINES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=GPU_ENGINES, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_development(args.input, args.output, args.engine)
    print(json.dumps({k: result[k] for k in ("status", "engine", "steps", "best_epoch")}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
