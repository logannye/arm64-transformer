"""All-origin, all-engine checkpoint audit, separate from measured execution.

Replay uses isolated subprocesses so CPU arithmetic comes from the frozen CPU
tree. Numerical qualification, exact replay, repeat equality and evidence
integrity are separate results. Failure records and arrays are retained.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

from transformer.gpu_study import (
    ENGINES,
    ENVIRONMENT,
    PROTOCOL,
    digest,
    disk_preflight,
    policy_path,
    python_path,
    read_json,
    require,
    required_artifacts,
    study_order,
    verify_admission,
    verify_bindings,
    verify_files,
    write_new,
)


def parameter_count(width):
    return 8 * width * width + 301 * width + 128


def read_array(path, dtype, shape):
    import numpy as np

    require(
        Path(path).stat().st_size == int(np.prod(shape)) * np.dtype(dtype).itemsize,
        f"Wrong artifact size: {path}",
    )
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    require(np.isfinite(value).all(), f"Nonfinite artifact: {path}")
    return value


def compare(actual, expected, atol=2e-5, rtol=2e-4):
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and actual.size, "Comparison shape mismatch")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite comparison")
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bound = atol + rtol * np.abs(expected.astype(np.float64))
    failed = error > bound
    indices = np.argwhere(failed)[:16]
    return {
        "passed": not bool(failed.any()),
        "atol": atol,
        "rtol": rtol,
        "coordinates": int(expected.size),
        "failed_coordinates": int(failed.sum()),
        "max_absolute_error": float(error.max()),
        "max_tolerance_ratio": float((error / bound).max()),
        "failure_examples": [
            {
                "index": i.tolist(),
                "actual": float(actual[tuple(i)]),
                "expected": float(expected[tuple(i)]),
                "allowed": float(bound[tuple(i)]),
            }
            for i in indices
        ],
    }


def quality(logits, targets):
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    require(logits.shape == (*targets.shape, 128), "Prediction/target shape mismatch")
    rows, labels = logits.reshape(-1, 128), targets.reshape(-1)
    centered = rows - rows.max(-1, keepdims=True)
    logsum = np.log(np.exp(centered).sum(-1))
    nll = float((logsum - centered[np.arange(len(labels)), labels]).mean())
    return {
        "tokens": len(labels),
        "nll": nll,
        "bits_per_character": nll / np.log(2),
        "accuracy": float((rows.argmax(-1) == labels).mean()),
    }


def prediction_delta(actual, expected, targets):
    import numpy as np

    def probabilities(value):
        value = value.astype(np.float64)
        exponent = np.exp(value - value.max(-1, keepdims=True))
        return exponent / exponent.sum(-1, keepdims=True)

    return {
        "max_probability_error": float(
            np.abs(probabilities(actual) - probabilities(expected)).max()
        ),
        "nll_delta": quality(actual, targets)["nll"] - quality(expected, targets)["nll"],
        "argmax_changes": int(np.count_nonzero(actual.argmax(-1) != expected.argmax(-1))),
    }


def inspect_cell(cell, directory, manifest_sha256):
    import numpy as np

    directory = Path(directory)
    receipt = read_json(directory / "cell.json")
    require(
        receipt.get("cell") == cell and receipt.get("manifest_sha256") == manifest_sha256,
        "Cell identity mismatch",
    )
    verify_files(directory, receipt.get("artifacts"))
    required = set(required_artifacts(cell["engine"]))
    require(required <= set(receipt["artifacts"]), "Cell omits required artifacts")
    count = parameter_count(cell["width"])
    for name in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32"):
        state = read_array(directory / name, "<f4", (count,))
        if name == "adam_v.f32":
            require(np.all(state >= 0), "Negative Adam second moment")
    history = read_array(directory / "history.f64", "<f8", (8, 3))
    timings = read_array(directory / "timings.f64", "<f8", (100,))
    logits = read_array(directory / "test_logits.f32", "<f4", (256, 32, 128))
    runtime = read_json(directory / "runtime.json") if cell["engine"] != "cpu_arm64" else {}
    require(np.all(history >= 0) and np.all(timings > 0), "Invalid loss/timing values")
    best_epoch = int(history[:, 1].argmin()) + 1
    if cell["engine"].startswith("cpu_"):
        require("performance.f64" in receipt["artifacts"], "Missing CPU performance artifact")
        performance = read_array(directory / "performance.f64", "<f8", (8,))
        training_seconds, recorded_epoch, steps = (
            float(performance[0]),
            performance[5],
            performance[6],
        )
        memory = {
            "process_rss_before": float(performance[2]),
            "process_rss_after": float(performance[3]),
            "process_rss_final": float(performance[4]),
        }
        require(
            0 < performance[2] <= performance[3] <= performance[4],
            "Invalid CPU peak memory counters",
        )
        require(performance[7] >= performance[0], "CPU total wall time excludes training")
        if cell["engine"] == "cpu_arm64":
            from transformer.backend import read_backend, validate_runtime_backend

            backend = read_backend(directory / "backend.u64", policy="auto")
            validate_runtime_backend(
                backend,
                {
                    "epochs": 8,
                    "train_sequences": 2048,
                    "validation_sequences": 256,
                    "test_sequences": 256,
                    "width": cell["width"],
                },
                policy="auto",
            )
            memory["native_execution_backend"] = backend
    else:
        training_seconds, recorded_epoch, steps = (
            runtime.get("training_seconds"),
            runtime.get("best_epoch"),
            runtime.get("steps"),
        )
        require(
            runtime.get("status") == "completed" and runtime.get("scope") == "formal_measurement",
            "Incomplete GPU fit",
        )
        memory = {
            key: runtime.get(key)
            for key in (
                "memory_before",
                "memory_after_training",
                "memory_final",
                "process_peak_rss_bytes",
            )
        }
    require(
        recorded_epoch == best_epoch and steps == 1024,
        "Checkpoint selection or update count mismatch",
    )
    require(
        isinstance(training_seconds, (int, float))
        and training_seconds >= float(history[:, 2].sum()) > 0,
        "Training timer excludes epochs or is invalid",
    )
    return {
        "cell": cell,
        "history": history,
        "logits": logits,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "latency_p50": float(np.median(timings)),
        "latency_p95": float(np.quantile(timings, 0.95)),
        "memory": memory,
        "receipt_sha256": digest(directory / "cell.json"),
    }


def replay_engine(engine, roots, width, fixture, checkpoint, output, *, manifest):
    env = os.environ.copy()
    env.update(ENVIRONMENT, ARM64_TRANSFORMER_WIDTH=str(width))
    tree = roots["cpu"] if engine.startswith("cpu_") else roots["source"]
    env["PYTHONPATH"] = python_path(manifest, roots, cpu=engine.startswith("cpu_"))
    env["ARM64_TRANSFORMER_METAL_POLICY"] = str(policy_path(manifest, roots))
    script = Path(roots["source"]) / "transformer/gpu_study_replay.py"
    args = [
        sys.executable,
        str(script.resolve()),
        "--engine",
        engine,
        "--fixture",
        str(Path(fixture).resolve()),
        "--checkpoint",
        str(Path(checkpoint).resolve()),
        "--output",
        str(Path(output).resolve()),
    ]
    subprocess.run(
        args, cwd=tree, env=env, check=True, capture_output=True, text=True, timeout=7200
    )


def audit_checkpoint(
    cell, manifest_sha256, roots, campaign, output, *, replay=replay_engine, manifest=None
):
    import numpy as np

    directory = Path(campaign) / cell["id"]
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Audit output must be fresh")
    info = inspect_cell(cell, directory, manifest_sha256)
    fixture = Path(roots["inputs"]) / f"width-{cell['width']}/seed-{cell['seed']}"
    output.mkdir(parents=True)
    arrays = {}
    for engine in (*ENGINES, "oracle"):
        path = output / f"{engine}.npz"
        replay(
            engine, roots, cell["width"], fixture, directory / "best.f32", path, manifest=manifest
        )
        with np.load(path, allow_pickle=False) as loaded:
            arrays[engine] = {key: loaded[key] for key in loaded.files}
    numerical, compatibility, oracle, differences = {}, {}, {}, {}
    integrity = {"own_test_replay_exact": False, "own_validation_history_exact": False}
    for split in ("validation", "test"):
        rows = read_array(fixture / f"{split}.i32", "<i4", (256, 33))
        targets = rows[:, 1:]
        expected = arrays["oracle"][split]
        require(
            expected.dtype == np.float64 and expected.shape == (256, 32, 128),
            "Wrong FP64 replay shape/type",
        )
        oracle[split] = compare(
            expected[[0, -1]], arrays["oracle"][split + "_torch_fp64"], 1e-10, 1e-10
        )
        for engine in ENGINES:
            value = arrays[engine][split]
            require(
                value.dtype == np.float32 and value.shape == expected.shape,
                "Wrong engine replay shape/type",
            )
            numerical[f"{engine}/{split}"] = compare(value, expected)
            differences[f"{engine}/{split}"] = prediction_delta(value, expected, targets)
        for left, right in itertools.combinations(ENGINES, 2):
            compatibility[f"{left}/{right}/{split}"] = compare(
                arrays[left][split], arrays[right][split]
            )
    own = arrays[cell["engine"]]
    integrity["own_test_replay_exact"] = own["test"].tobytes() == info["logits"].tobytes()
    integrity["own_validation_history_exact"] = float(own["validation_nll"]) == float(
        info["history"][info["best_epoch"] - 1, 1]
    )
    if (directory / "inference_logits.f32").exists():
        saved_b1 = read_array(directory / "inference_logits.f32", "<f4", (1, 32, 128))
        integrity["own_b1_replay_exact"] = own["b1"].tobytes() == saved_b1.tobytes()
    for engine in ENGINES:
        numerical[f"{engine}/b1"] = compare(arrays[engine]["b1"], arrays["oracle"]["test"][:1])
    metrics = quality(info["logits"], read_array(fixture / "test.i32", "<i4", (256, 33))[:, 1:])
    report = {
        "schema": 1,
        "protocol": PROTOCOL,
        "cell": cell,
        "manifest_sha256": manifest_sha256,
        "cell_receipt_sha256": info["receipt_sha256"],
        "integrity": integrity,
        "oracle": oracle,
        "fp64": numerical,
        "compatibility": compatibility,
        "prediction_differences": differences,
        "quality": metrics,
        "training_seconds": info["training_seconds"],
        "latency_p50": info["latency_p50"],
        "latency_p95": info["latency_p95"],
        "memory": info["memory"],
        "replay_sha256": {p.name: digest(p) for p in sorted(output.glob("*.npz"))},
    }
    write_new(output / "audit.json", report)
    return report


def compare_repeat_pair(left, right, engine):
    left, right = Path(left), Path(right)
    names = ["best.f32", "last.f32", "adam_m.f32", "adam_v.f32", "test_logits.f32"]
    if not engine.startswith("cpu_"):
        names.append("inference_logits.f32")
    checks = {name: digest(left / name) == digest(right / name) for name in names}
    histories = [read_array(p / "history.f64", "<f8", (8, 3))[:, :2].copy() for p in (left, right)]
    checks["history_without_timings"] = histories[0].tobytes() == histories[1].tobytes()
    return {"checks": checks, "passed": all(checks.values())}


def repeat_integrity(campaign):
    results = []
    cells = study_order()
    index = {(c["width"], c["seed"], c["repeat"], c["engine"]): c for c in cells}
    for cell in (c for c in cells if c["repeat"] == 1):
        other = index[cell["width"], cell["seed"], 2, cell["engine"]]
        left, right = (Path(campaign) / c["id"] for c in (cell, other))
        results.append(
            {
                "first": cell["id"],
                "second": other["id"],
                **compare_repeat_pair(left, right, cell["engine"]),
            }
        )
    return results


def audit_campaign(manifest_path, expected_hash, roots, admission, campaign, output):
    require(digest(manifest_path) == expected_hash, "Manifest trust anchor mismatch")
    manifest = read_json(manifest_path)
    verify_admission(manifest, roots, admission)
    campaign = Path(campaign)
    ledger = read_json(campaign / "campaign.json")
    require(
        ledger.get("manifest_sha256") == expected_hash
        and ledger.get("status") == "measurement_complete_unaudited",
        "Incomplete or foreign campaign",
    )
    require(
        [
            {k: entry.get(k) for k in cell}
            for entry, cell in zip(ledger.get("cells", []), study_order())
        ]
        == study_order()
        and len(ledger.get("cells", [])) == 180,
        "Incomplete/reordered campaign",
    )
    require(
        all(entry.get("status") == "completed" for entry in ledger["cells"]), "Incomplete cells"
    )
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Audit destination must be fresh")
    disk = disk_preflight(output, "audit")
    output.mkdir(parents=True)
    reports = []
    for cell, entry in zip(study_order(), ledger["cells"], strict=True):
        require(
            digest(campaign / cell["id"] / "cell.json") == entry.get("receipt_sha256"),
            "Campaign receipt changed",
        )
        audit_checkpoint(
            cell, expected_hash, roots, campaign, output / cell["id"], manifest=manifest
        )
        reports.append(
            {
                "cell": cell,
                "path": f"{cell['id']}/audit.json",
                "sha256": digest(output / cell["id"] / "audit.json"),
            }
        )
        verify_bindings(manifest, roots)
    repeats = repeat_integrity(campaign)
    result = {
        "schema": 1,
        "protocol": PROTOCOL,
        "manifest_sha256": expected_hash,
        "campaign_sha256": digest(campaign / "campaign.json"),
        "reports": reports,
        "repeat_checks": repeats,
        "planned_checkpoints": 180,
        "audited_checkpoints": len(reports),
        "checkpoint_split_sequences": 92160,
        "five_engine_sequence_evaluations": 460800,
        "disk_preflight": disk,
    }
    write_new(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "manifest-sha256", "roots", "admission", "campaign", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    audit_campaign(
        args.manifest,
        args.manifest_sha256,
        json.loads(args.roots),
        args.admission,
        args.campaign,
        args.output,
    )


if __name__ == "__main__":
    main()
