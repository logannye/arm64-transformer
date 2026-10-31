"""Measurement orchestration for the separately authenticated library successor.

The numerical model and optimizers stay in the frozen math role. The training
and inference timing boundaries mirror that role's gpu_study_worker.gpu_fit.
This function grants no admission, generates no operands and has no CLI route.
"""

import importlib.util
import resource
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "library_measurement_execution", HERE / "execution.py"
)
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)
C = E.C


def gpu_fit(fixture, output, engine, *, runtime, scope):
    """Mirror the declared timing boundary, preserving all device state controls."""
    import numpy as np

    from transformer.gpu_run import checked_state, evaluate, require_finite, scalar, state_equal
    from transformer.spec import PARAM_COUNT

    C.require(scope in ("formal_measurement", "development_only"), "Invalid execution scope")
    if scope == "formal_measurement":
        C.require_execution_support()
    output = Path(output)
    C.require(
        output.is_dir()
        and output.resolve(strict=True) == output.absolute()
        and not any(output.iterdir()),
        "Measurement output must be a fresh empty canonical directory",
    )
    receipt = {
        "scope": scope,
        "engine": engine,
        "status": "started",
        "framework_import_seconds": None,
    }
    receipt["import_scope"] = (
        "Authenticated runtime import and construction included in setup_seconds"
    )
    setup, backend = time.perf_counter(), None
    try:
        backend = E.create_backend(runtime, engine, fixture["initial"], max_batch=16)
        data = {
            name: backend.upload_dataset(fixture[name]) for name in ("train", "validation", "test")
        }
        schedule = backend.upload_indices(fixture["schedule"].reshape(-1))
        order = backend.upload_indices(np.arange(256, dtype=np.int32))
        original = checked_state(backend, PARAM_COUNT)
        warm = backend.gather_batch(data["train"], schedule, 0, 16)
        backend.train_step(warm)
        require_finite(backend)
        loss, logits = backend.evaluate(warm)
        require_finite(backend, logits)
        require_finite(backend, loss)
        scalar(backend, loss)
        backend.export_parameters()
        backend.synchronize()
        backend.load_state(original)
        C.require(
            state_equal(original, checked_state(backend, PARAM_COUNT)),
            "Warmup changed initial state",
        )
        receipt.update(
            setup_seconds=time.perf_counter() - setup,
            warmup_restored_exactly=True,
            setup_scope=(
                "Backend construction/shader compilation, uploads and restored warmup; "
                "framework import separate"
            ),
            capabilities=E.validate_capabilities(engine, backend.capabilities()),
            memory_before=backend.memory_stats(),
        )
        history = np.empty((8, 3), dtype="<f8")
        best, best_epoch, best_parameters = float("inf"), 0, None
        backend.synchronize()
        started = time.perf_counter()
        for epoch in range(8):
            epoch_start, total = time.perf_counter(), 0.0
            for offset in range(0, 2048, 16):
                batch = backend.gather_batch(data["train"], schedule, epoch * 2048 + offset, 16)
                loss = backend.train_step(batch)
                require_finite(backend)
                total += scalar(backend, loss) * 16
            validation = evaluate(backend, data["validation"], order, 256, 16)
            if validation < best:
                best, best_epoch = validation, epoch + 1
                best_parameters = backend.export_parameters()
                C.require(
                    best_parameters.dtype == np.float32
                    and best_parameters.shape == (PARAM_COUNT,)
                    and np.isfinite(best_parameters).all(),
                    "Invalid selected checkpoint",
                )
                best_parameters.astype("<f4").tofile(output / "best.f32")
            backend.synchronize()
            history[epoch] = total / 2048, validation, time.perf_counter() - epoch_start
        backend.synchronize()
        receipt["training_seconds"] = time.perf_counter() - started
        receipt["memory_after_training"] = backend.memory_stats()
        history.tofile(output / "history.f64")
        last = checked_state(backend, PARAM_COUNT)
        C.require(last["step"] == 1024, "Incorrect optimizer update count")
        for key, filename in (("parameters", "last.f32"), ("m", "adam_m.f32"), ("v", "adam_v.f32")):
            last[key].astype("<f4").tofile(output / filename)
        backend.load_state({**last, "parameters": best_parameters})
        with (output / "test_logits.f32").open("wb") as handle:
            receipt["test_nll"] = evaluate(
                backend, data["test"], order, 256, 16, logits_file=handle
            )
        replay_validation = evaluate(backend, data["validation"], order, 256, 16)
        C.require(replay_validation == best, "Own-engine validation replay changed")
        sample = backend.upload_inputs(fixture["test"][:1, :32])
        for _ in range(20):
            logits = backend.forward(sample)
        backend.synchronize()
        before = backend.to_numpy(logits).copy()
        timings = np.empty(100, dtype="<f8")
        for i in range(100):
            tick = time.perf_counter()
            logits = backend.forward(sample)
            backend.synchronize()
            timings[i] = time.perf_counter() - tick
        require_finite(backend, logits)
        after = backend.to_numpy(logits)
        C.require(
            before.dtype == after.dtype == np.float32
            and before.shape == after.shape == (1, 32, 128)
            and before.tobytes() == after.tobytes(),
            "B1 replay changed",
        )
        after.astype("<f4").tofile(output / "inference_logits.f32")
        timings.tofile(output / "timings.f64")
        receipt["origins_after"] = runtime.origins(engine)
        if engine == "metal":
            receipt["metal_library_after"] = E.loaded_metal_library(runtime)
        runtime.recheck()
        receipt.update(
            status="completed",
            best_epoch=best_epoch,
            steps=1024,
            selected_validation_replay_exact=True,
            inference_replay_exact=True,
            memory_final=backend.memory_stats(),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
        except Exception as exc:
            receipt.update(status="failed", cleanup_error=repr(exc))
            raise
        finally:
            C.write(output / "runtime.json", receipt)
