"""Independent checks of saved run artifacts and language-model quality.

Metrics use NumPy float64 arithmetic. Selected-checkpoint predictions are checked
against a separate functional PyTorch implementation, never by retraining.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

from transformer import spec
from transformer.validate import Checks, reference_forward, sha256, tensor


def read_exact(path, dtype, shape):
    path = Path(path)
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path}: expected exactly {expected_bytes} bytes")
    values = np.fromfile(path, dtype=dtype).reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: nonfinite values")
    return values


def prediction_metrics(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    if logits.shape[:-1] != targets.shape or logits.shape[-1] != spec.VOCAB:
        raise ValueError("Logits and targets do not have the expected matching shapes")
    if not np.isfinite(logits).all() or np.any((targets < 0) | (targets >= spec.VOCAB)):
        raise ValueError("Nonfinite logits or invalid targets")
    rows = logits.reshape(-1, spec.VOCAB)
    labels = targets.reshape(-1).astype(np.int64)
    maximum = rows.max(axis=-1)
    log_normalizer = maximum + np.log(np.exp(rows - maximum[:, None]).sum(axis=-1))
    nll = log_normalizer - rows[np.arange(len(rows)), labels]
    mean_nll = float(nll.mean())
    return {
        "tokens": len(rows),
        "nll": mean_nll,
        "bits_per_character": mean_nll / math.log(2),
        "perplexity": math.exp(mean_nll),
        "accuracy": float((rows.argmax(axis=-1) == labels).mean()),
        "total_nll": float(nll.sum()),
    }


def audit_run(fixture_dir, run_dir, checks=None, *, recompute=True):
    """Return independently measured metrics and add evidence to ``checks``."""
    if checks is None:
        checks = Checks()
    fixture, run = Path(fixture_dir), Path(run_dir)
    name = run.name
    raw_config = (fixture / "config.bin").read_bytes()
    if len(raw_config) == 36:
        magic, version, epochs, batch, train_count, val_count, test_count, seed, width = (
            struct.unpack("<9I", raw_config)
        )
        if version != 2:
            raise ValueError("Fixture config version mismatch")
    elif len(raw_config) == 32:
        magic, version, epochs, batch, train_count, val_count, test_count, seed = struct.unpack(
            "<8I", raw_config
        )
        width = 32
        if version != 1:
            raise ValueError("Legacy config version mismatch")
    else:
        raise ValueError("Malformed fixture config")
    if magic != 0x54465231 or width != spec.WIDTH:
        raise ValueError("Fixture magic/model width mismatch")
    if not (
        1 <= epochs <= 100
        and 1 <= batch <= 128
        and 1 <= train_count <= 8192
        and 1 <= val_count <= 2048
        and 1 <= test_count <= 2048
    ):
        raise ValueError("Fixture bounds invalid")
    manifest = json.loads((fixture / "manifest.json").read_text())
    expected_files = {
        "config.bin",
        "initial.f32",
        "train.i32",
        "validation.i32",
        "test.i32",
        "schedule.i32",
    }
    checks.require(name + "/fixture_manifest_files", set(manifest["sha256"]) == expected_files)
    for filename in expected_files:
        checks.require(
            name + "/fixture_hash/" + filename,
            sha256(fixture / filename) == manifest["sha256"].get(filename),
        )
    test = read_exact(fixture / "test.i32", "<i4", (test_count, spec.CONTEXT + 1))
    if np.any((test < 0) | (test >= spec.VOCAB)):
        raise ValueError("Invalid fixture test tokens")
    parameters = {}
    for filename in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32"):
        parameters[filename] = read_exact(run / filename, "<f4", (spec.PARAM_COUNT,))
    checks.require(name + "/second_moment_nonnegative", np.all(parameters["adam_v.f32"] >= 0))
    history = read_exact(run / "history.f64", "<f8", (epochs, 3))
    performance = read_exact(run / "performance.f64", "<f8", (8,))
    timings = read_exact(run / "timings.f64", "<f8", (100,))
    logits = read_exact(run / "test_logits.f32", "<f4", (test_count, spec.CONTEXT, spec.VOCAB))
    checks.require(name + "/history_nonnegative", np.all(history >= 0))
    checks.require(name + "/timers_positive", np.all(performance[[0, 1, 7]] > 0))
    checks.require(name + "/latencies_positive", np.all(timings > 0))
    checks.require(name + "/rss_monotone", 0 < performance[2] <= performance[3] <= performance[4])
    checks.require(name + "/native_wall_includes_training", performance[7] >= performance[0])
    checks.require(name + "/best_validation_epoch", performance[5] == 1 + history[:, 1].argmin())
    checks.require(
        name + "/optimizer_step_count", performance[6] == epochs * math.ceil(train_count / batch)
    )
    checks.require(name + "/training_includes_epoch_time", performance[0] >= history[:, 2].sum())
    selected_validation_nll = None
    if recompute:
        torch.set_num_threads(1)
        flat = tensor(parameters["best.f32"])
        with torch.no_grad():
            for index, sequence in enumerate(test):
                expected = reference_forward(flat, tensor(sequence[:-1], dtype=torch.long))
                checks.close(
                    f"{name}/selected_checkpoint/sequence_{index}", logits[index], expected
                )
            validation = read_exact(
                fixture / "validation.i32", "<i4", (val_count, spec.CONTEXT + 1)
            )
            if np.any((validation < 0) | (validation >= spec.VOCAB)):
                raise ValueError("Invalid fixture validation tokens")
            validation_logits = np.empty((val_count, spec.CONTEXT, spec.VOCAB), np.float32)
            for index, sequence in enumerate(validation):
                expected = reference_forward(flat, tensor(sequence[:-1], dtype=torch.long))
                validation_logits[index] = expected.numpy()
            selected_validation_nll = prediction_metrics(validation_logits, validation[:, 1:])[
                "nll"
            ]
            checks.close(
                name + "/selected_checkpoint_validation_nll",
                selected_validation_nll,
                history[:, 1].min(),
            )
    quality = prediction_metrics(logits, test[:, 1:])
    return {
        "seed": seed,
        "fixture": str(fixture.resolve()),
        "run": str(run.resolve()),
        "fixture_sha256": manifest["sha256"],
        "artifact_sha256": {
            path.name: sha256(path) for path in sorted(run.iterdir()) if path.is_file()
        },
        "quality": quality,
        "training_seconds": float(performance[0]),
        "training_cpu_seconds": float(performance[1]),
        "peak_rss_before_bytes": int(performance[2]),
        "peak_rss_after_bytes": int(performance[3]),
        "peak_rss_final_bytes": int(performance[4]),
        "best_epoch": int(performance[5]),
        "optimizer_steps": int(performance[6]),
        "native_total_seconds": float(performance[7]),
        "forward_latency_seconds": {
            "p50": float(np.quantile(timings, 0.5)),
            "p95": float(np.quantile(timings, 0.95)),
            "p99": float(np.quantile(timings, 0.99)),
            "mean": float(timings.mean()),
            "count": len(timings),
        },
        "learning_history": history[:, :2].tolist(),
        "checkpoint_predictions_recomputed": bool(recompute),
        "selected_checkpoint_validation_nll": selected_validation_nll,
    }


def compare_repeats(first, second, checks):
    """Within-implementation repeats must reproduce non-timing outputs exactly."""
    first, second = Path(first), Path(second)
    prefix = f"repeat/{first.name}/{second.name}"
    for filename in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32", "test_logits.f32"):
        checks.require(
            prefix + "/" + filename,
            (first / filename).read_bytes() == (second / filename).read_bytes(),
        )
    a = np.fromfile(first / "history.f64", "<f8").reshape(-1, 3)
    b = np.fromfile(second / "history.f64", "<f8").reshape(-1, 3)
    checks.require(prefix + "/learning_history", np.array_equal(a[:, :2], b[:, :2]))
    a = np.fromfile(first / "performance.f64", "<f8")
    b = np.fromfile(second / "performance.f64", "<f8")
    checks.require(prefix + "/selection_and_steps", np.array_equal(a[5:7], b[5:7]))


def paired_predictions(first, second):
    """Descriptive final-model differences; no equivalence assertion."""
    a = np.fromfile(Path(first) / "test_logits.f32", "<f4").reshape(-1, spec.VOCAB)
    b = np.fromfile(Path(second) / "test_logits.f32", "<f4").reshape(-1, spec.VOCAB)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired logits have mismatched shape or nonfinite values")
    return {
        "token_count": len(a),
        "prediction_disagreement": float((a.argmax(-1) != b.argmax(-1)).mean()),
        "logit_rmse": float(np.sqrt(np.mean((a.astype(np.float64) - b) ** 2))),
        "logit_max_absolute_difference": float(np.max(np.abs(a.astype(np.float64) - b))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Audit output exists; choose a new path")
    checks = Checks()
    error = None
    try:
        result = audit_run(args.fixture, args.run, checks)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        checks.require("audit/exception", False, error)
        result = {}
    result.update(
        passed=checks.passed, check_count=len(checks.records), checks=checks.records, error=error
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": checks.passed, "checks": len(checks.records), "error": error}))
    return 0 if checks.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
