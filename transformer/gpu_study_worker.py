"""One authenticated formal fit. Development runner semantics are unchanged."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import resource
import subprocess
import time
from pathlib import Path

from transformer.gpu_environment import child_environment, require_environment
from transformer.gpu_study import (
    ENVIRONMENT,
    digest,
    policy_path,
    python_path,
    read_json,
    require,
    required_artifacts,
    study_order,
    verify_admission,
    verify_bindings,
    write_new,
)


def checkpoint_improves(candidate_loss, previous_loss):
    """Production selection rule: finite candidates replace only a strict minimum."""
    require(
        math.isfinite(candidate_loss) and not math.isnan(previous_loss),
        "Invalid checkpoint selection loss",
    )
    return candidate_loss < previous_loss


def gpu_fit(fixture, output, engine, *, manifest=None, roots=None):
    """Mirror the declared timing boundary, preserving all device state controls."""
    import numpy as np

    from transformer.gpu_run import checked_state, evaluate, require_finite, scalar, state_equal
    from transformer.gpu_study_context import create_backend, resolve_context
    from transformer.spec import PARAM_COUNT

    owned = resolve_context(manifest).owned
    import_start = time.perf_counter()
    if not owned:
        module = importlib.import_module(f"transformer.gpu_{engine}")
    receipt = {
        "scope": "formal_measurement",
        "engine": engine,
        "status": "started",
        "framework_import_seconds": None if owned else time.perf_counter() - import_start,
    }
    if owned:
        receipt["import_scope"] = (
            "Authenticated runtime import and construction included in setup_seconds"
        )
    setup, backend = time.perf_counter(), None
    try:
        backend = (
            create_backend(engine, fixture["initial"], max_batch=16, manifest=manifest, roots=roots)
            if owned
            else module.Backend(fixture["initial"], max_batch=16)
        )
        if engine == "metal":
            from transformer.gpu_policy import bundle, identity
            from transformer.spec import WIDTH

            require(
                backend.capabilities().get("kernel_policy_sha256") == identity(bundle(width=WIDTH)),
                "Executed Metal policy does not match selected bundle",
            )
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
        require(
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
            capabilities=backend.capabilities(),
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
            if checkpoint_improves(validation, best):
                best, best_epoch = validation, epoch + 1
                best_parameters = backend.export_parameters()
                require(
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
        require(last["step"] == 1024, "Incorrect optimizer update count")
        for key, filename in (("parameters", "last.f32"), ("m", "adam_m.f32"), ("v", "adam_v.f32")):
            last[key].astype("<f4").tofile(output / filename)
        backend.load_state({**last, "parameters": best_parameters})
        with (output / "test_logits.f32").open("wb") as handle:
            receipt["test_nll"] = evaluate(
                backend, data["test"], order, 256, 16, logits_file=handle
            )
        replay_validation = evaluate(backend, data["validation"], order, 256, 16)
        require(replay_validation == best, "Own-engine validation replay changed")
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
        require(
            before.dtype == after.dtype == np.float32
            and before.shape == after.shape == (1, 32, 128)
            and before.tobytes() == after.tobytes(),
            "B1 replay changed",
        )
        after.astype("<f4").tofile(output / "inference_logits.f32")
        timings.tofile(output / "timings.f64")
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
            write_new(output / "runtime.json", receipt)


def run_cell(manifest_path, expected_hash, admission, roots, cell_id, output):
    require(digest(manifest_path) == expected_hash, "Manifest hash changed")
    manifest = read_json(manifest_path)
    verify_admission(manifest, roots, admission)
    matches = [cell for cell in study_order() if cell["id"] == cell_id]
    require(len(matches) == 1, "Unknown planned cell")
    cell = matches[0]
    require(
        os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(cell["width"]),
        "Width must be selected before process imports",
    )
    require_environment(ENVIRONMENT)
    require(
        os.environ.get("ARM64_TRANSFORMER_METAL_POLICY") == str(policy_path(manifest, roots)),
        "Selected policy path changed",
    )
    fixture_path = Path(roots["inputs"]) / f"width-{cell['width']}/seed-{cell['seed']}"
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Output collision")
    output.parent.mkdir(parents=True, exist_ok=True)
    from transformer.gpu_study_context import child_controls, python_command, resolve_context

    if cell["engine"] in ("cpu_arm64", "cpu_torch"):
        cpu = Path(roots["cpu"]).resolve()
        command = (
            [str(cpu / f"native/build/width-{cell['width']}/transformer-arm64")]
            if cell["engine"] == "cpu_arm64"
            else python_command(manifest, output, roots=roots, cpu=True)
            + ["-m", "transformer.python_train"]
        )
        env = child_environment(ENVIRONMENT)
        env["PYTHONPATH"] = python_path(manifest, roots, cpu=True)
        arguments = [str(fixture_path.resolve()), str(output.resolve())]
        if resolve_context(manifest).owned and cell["engine"] == "cpu_torch":
            from transformer.gpu_owned_cpu import prepare_command

            command = prepare_command(
                manifest, roots, "cpu_torch", "fit", arguments, output, admission_receipt=admission
            )
            arguments = []
        subprocess.run(
            command + arguments,
            cwd=cpu,
            env=child_controls(manifest, env),
            check=True,
        )
    else:
        from transformer.gpu_inputs import load_fixture

        fixture = load_fixture(fixture_path)
        output.mkdir()
        gpu_fit(fixture, output, cell["engine"], manifest=manifest, roots=roots)
    verify_bindings(manifest, roots)
    if resolve_context(manifest).owned and cell["engine"] == "cpu_torch":
        from transformer.gpu_owned_cpu import verify_proof

        files = {
            path.name: digest(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "owned-cpu-proof.json"
        }
        verify_proof(
            output / "owned-cpu-proof.json",
            manifest,
            engine="cpu_torch",
            kind="fit",
            output_hashes=files,
            input_hashes={
                path.name: digest(path) for path in fixture_path.iterdir() if path.is_file()
            },
        )
    require(
        all((output / name).is_file() for name in required_artifacts(cell["engine"])),
        "Missing measured artifacts",
    )
    write_new(
        output / "cell.json",
        {
            "schema": 1,
            "cell": cell,
            "manifest_sha256": expected_hash,
            "artifacts": {
                path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "manifest-sha256", "admission", "roots", "cell", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run_cell(
        args.manifest,
        args.manifest_sha256,
        args.admission,
        json.loads(args.roots),
        args.cell,
        args.output,
    )


if __name__ == "__main__":
    main()
