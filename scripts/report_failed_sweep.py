"""Post-study descriptive reporting for the failed V5 checkpoint audit.

This reporting-only script never trains, runs model inference, changes tolerances,
or changes a failed result into a passed result. Original reports remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformer.audit import (  # noqa: E402
    compare_repeats,
    paired_predictions,
    prediction_metrics,
    read_exact,
)
from transformer.backend import read_backend, validate_runtime_backend  # noqa: E402
from transformer.report import aggregate_runs, count_baselines, seed_statistics  # noqa: E402
from transformer.sweep_report import validate_manifest  # noqa: E402
from transformer.validate import SOURCE_FILES, Checks  # noqa: E402

WIDTHS = (512, 1024)
SEEDS = (42, 43, 44)
IMPLEMENTATIONS = ("python", "arm64")
PARAMETER_FILES = ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32")
FIXTURE_FILES = (
    "config.bin",
    "initial.f32",
    "train.i32",
    "validation.i32",
    "test.i32",
    "schedule.i32",
)
RUN_FILES = {*PARAMETER_FILES, "history.f64", "performance.f64", "timings.f64", "test_logits.f32"}
REPORT_FILES = ("README.md", "audit.json", "summary.json")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_file(path, expected, verified):
    path = Path(path).resolve()
    require(path.is_file() and digest(path) == expected, f"Evidence hash mismatch: {path}")
    require(
        str(path) not in verified or verified[str(path)] == expected, f"Conflicting hash: {path}"
    )
    verified[str(path)] = expected


def authenticate_git_sources(root, revision, expected):
    """Authenticate the frozen source tree, allowing later documentation-only edits."""
    require(expected and len(revision) == 40, "Missing measured source identity")
    names = list(expected)
    for name in names:
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe measured source path")
    request = "".join(f"{revision}:{name}\n" for name in names).encode()
    data = subprocess.check_output(["git", "cat-file", "--batch"], input=request, cwd=root)
    offset = 0
    for name in names:
        end = data.index(b"\n", offset)
        header = data[offset:end].decode().split()
        require(len(header) == 3 and header[1] == "blob", f"Missing source blob: {name}")
        size = int(header[2])
        blob = data[end + 1 : end + 1 + size]
        require(hashlib.sha256(blob).hexdigest() == expected[name], f"Source mismatch: {name}")
        offset = end + size + 2
    require(offset == len(data), "Unexpected Git source stream content")
    # Imported validators must be precisely those used for the measured campaign.
    imported = {
        *SOURCE_FILES,
        "transformer/sweep_report.py",
        "transformer/sweep.py",
        "transformer/report.py",
    }
    for name in imported:
        require(
            digest(Path(root) / name) == expected.get(name), f"Changed reused validator: {name}"
        )


def authenticate_admitted_files(manifest, inputs, verified):
    """Reconstruct coverage; a shortened hash map must not omit an admitted binary."""
    expected = {}
    for width, modes in manifest["regression_preflights"].items():
        for mode, gate in modes.items():
            for prefix in ("library", "executable", "generated_layout"):
                path = str(Path(gate[prefix + "_path"]).resolve())
                value = gate[prefix + "_sha256"]
                require(path not in expected or expected[path] == value, "Gate binary disagreement")
                expected[path] = value
            filename = f"width-{width}{'' if mode == 'auto' else '-' + mode}.json"
            path = Path(manifest["preflight_directory"]) / filename
            require(read_json(path) == gate, "Original gate differs from admitted record")
            expected[str(path.resolve())] = digest(path)
    recorded = {str(Path(p).resolve()): h for p, h in manifest["authenticated_files"].items()}
    require(recorded == expected, "Authenticated binary/layout/gate coverage changed")
    fixture_paths = {
        str((Path(inputs) / f"width-{w}" / f"seed-{s}" / name).resolve())
        for w in WIDTHS
        for s in SEEDS
        for name in (*FIXTURE_FILES, "manifest.json")
    }
    require(
        {str(Path(p).resolve()) for p in manifest["fixture_files"]} == fixture_paths,
        "Authenticated fixture coverage changed",
    )
    for group in ("authenticated_files", "fixture_files"):
        for path, value in manifest[group].items():
            verify_file(path, value, verified)


def expected_audit_names(entries):
    names = {
        "manifest/preflight_passed",
        "manifest/forced_backend_gates",
        "baselines/training_target_count",
    }
    names.update("source/" + name for name in SOURCE_FILES)
    run_checks = (
        "conditions_valid",
        "source_unchanged",
        "binary_unchanged",
        "fixture_manifest_files",
        "second_moment_nonnegative",
        "history_nonnegative",
        "timers_positive",
        "latencies_positive",
        "rss_monotone",
        "native_wall_includes_training",
        "best_validation_epoch",
        "optimizer_step_count",
        "training_includes_epoch_time",
        "selected_checkpoint_validation_nll",
        "runner_fixture_identity",
        "parent_wall_includes_training",
    )
    for entry in entries:
        prefix = entry["name"] + "/"
        names.update(prefix + suffix for suffix in run_checks)
        names.update(prefix + "fixture_hash/" + filename for filename in FIXTURE_FILES)
        names.update(prefix + f"selected_checkpoint/sequence_{i}" for i in range(256))
        if entry["implementation"] == "arm64":
            names.add(prefix + "backend_authenticated")
    for seed in SEEDS:
        for impl in IMPLEMENTATIONS:
            prefix = f"repeat/seed-{seed}-repeat-1-{impl}/seed-{seed}-repeat-2-{impl}/"
            names.update(
                prefix + f
                for f in (
                    *PARAMETER_FILES,
                    "test_logits.f32",
                    "learning_history",
                    "selection_and_steps",
                )
            )
    names.update(
        f"fixtures/seed-{s}/same_{split}"
        for s in SEEDS[1:]
        for split in ("train", "validation", "test")
    )
    return names


def validate_failed_audit(audit, summary, entries):
    """Keep every original failed record; reject missing checks or altered status."""
    require(summary.get("status") == "audit_failed", "Original summary must retain audit_failed")
    records = audit.get("checks", [])
    names = [record.get("name") for record in records]
    require(
        audit.get("passed") is False
        and audit.get("error") is None
        and audit.get("check_count") == len(records)
        and len(set(names)) == len(names)
        and set(names) == expected_audit_names(entries),
        "Original audit is incomplete, duplicated, or did not finish",
    )
    require(all(type(r.get("passed")) is bool for r in records), "Invalid original check status")
    failures = [r for r in records if not r["passed"]]
    require(
        failures and audit.get("failed_checks") == [r["name"] for r in failures],
        "Failure list was altered",
    )
    for record in records:
        if "/selected_checkpoint/sequence_" in record["name"]:
            require(record.get("tolerance") == "forward", "Checkpoint tolerance label changed")
            require(
                record.get("actual_shape") == [32, 128]
                and record.get("expected_shape") == [32, 128],
                "Checkpoint shape changed",
            )
            require(
                all(
                    math.isfinite(record[k]) and record[k] >= 0
                    for k in ("max_absolute_error", "rmse")
                ),
                "Invalid checkpoint error",
            )
    require(
        all("-arm64/selected_checkpoint/sequence_" in r["name"] for r in failures),
        "Unexpected failed audit class; preserve and investigate it separately",
    )
    run_names = [r.get("name") for r in summary.get("runs", [])]
    require(
        run_names == [e["name"] for e in entries], "Failed report run order or coverage changed"
    )
    return failures


def raw_run_metrics(run_dir, fixture, entry, saved, model, verified):
    """Recompute descriptive metrics from saved bytes, with no model execution."""
    expected_files = RUN_FILES | (
        {"backend.u64"} if entry["implementation"] == "arm64" else {"runtime.json"}
    )
    require(set(saved["artifact_sha256"]) == expected_files, "Original run hash coverage changed")
    require(
        {p.name for p in run_dir.iterdir() if p.is_file()} == expected_files,
        "Unexpected raw run files",
    )
    for filename, value in saved["artifact_sha256"].items():
        verify_file(run_dir / filename, value, verified)
    for filename in PARAMETER_FILES:
        require(
            (run_dir / filename).stat().st_size == model["parameters"] * 4, "Wrong checkpoint size"
        )
    performance = read_exact(run_dir / "performance.f64", "<f8", (8,))
    history = read_exact(run_dir / "history.f64", "<f8", (8, 3))
    timings = read_exact(run_dir / "timings.f64", "<f8", (100,))
    logits = read_exact(run_dir / "test_logits.f32", "<f4", (256, 32, 128))
    tokens = read_exact(fixture / "test.i32", "<i4", (256, 33))
    require(
        performance[6] == 1024 and performance[5] == history[:, 1].argmin() + 1,
        "Wrong step budget or checkpoint selection",
    )
    require(np.all(performance[[0, 1, 7]] > 0) and np.all(timings > 0), "Invalid timer")
    require(0 < performance[2] <= performance[3] <= performance[4], "Invalid RSS boundaries")
    require(
        performance[0] >= history[:, 2].sum() and performance[7] >= performance[0],
        "Invalid timing scope",
    )
    metrics = {
        "quality": prediction_metrics(logits, tokens[:, 1:]),
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
    }
    require(
        all(saved[key] == value for key, value in metrics.items()),
        "Descriptive saved metrics do not match raw bytes",
    )
    for key in ("seed", "repeat", "implementation", "name", "process_wall_seconds"):
        require(saved[key] == entry[key], f"Run identity/timing mismatch: {key}")
    require(entry["process_wall_seconds"] >= performance[0], "Parent timer excludes training")
    require(
        saved.get("checkpoint_predictions_recomputed") is True, "Original checkpoint audit missing"
    )
    require(
        math.isfinite(saved["selected_checkpoint_validation_nll"]), "Invalid audited validation NLL"
    )
    return dict(saved, **metrics)


def authenticate_completed_failed_campaign(
    results, inputs, parent_failure, width_reports, *, root=ROOT
):
    """Authenticate all retained evidence; return data for reporting or packaging.

    Success here means the failed study's bytes and descriptive arithmetic are
    consistent. It never means that the numerical checkpoint audit passed.
    """
    results, inputs, parent_failure, root = map(Path, (results, inputs, parent_failure, root))
    width_reports = {int(w): Path(p) for w, p in width_reports.items()}
    require(set(width_reports) == set(WIDTHS), "Both original width reports are required")
    manifest = read_json(results / "manifest.json")
    require(manifest.get("protocol_version") == "v5", "This outcome reporter only supports V5")
    children = {w: read_json(results / f"width-{w}" / "manifest.json") for w in WIDTHS}
    checks = Checks()
    validate_manifest(manifest, children, checks)
    authenticate_git_sources(root, manifest["revision"], manifest["source_sha256"])
    checks.require("outcome/frozen_git_source", True)
    verified = {}
    authenticate_admitted_files(manifest, inputs, verified)
    for path in (
        results / "manifest.json",
        *(results / f"width-{w}" / "manifest.json" for w in WIDTHS),
    ):
        verify_file(path, digest(path), verified)
    require(manifest["pause_seconds"] == 15, "Changed inter-run pause")
    for entry in manifest["runs"]:
        for phase in ("source_before", "source_after"):
            require(
                entry[phase]["head"]["stdout"].strip() == manifest["revision"],
                "Run source revision changed",
            )
    parent_audit = read_json(parent_failure / "audit.json")
    require(
        parent_audit.get("passed") is False
        and read_json(parent_failure / "summary.json").get("status") == "audit_failed",
        "Parent failure must remain explicit",
    )
    for filename in (*REPORT_FILES, "report-processes.json"):
        verify_file(parent_failure / filename, digest(parent_failure / filename), verified)
    runs, audits, failed, pairs = {}, {}, [], {}
    for width in WIDTHS:
        child, report = children[width], width_reports[width]
        saved_summary, audit = read_json(report / "summary.json"), read_json(report / "audit.json")
        failures = validate_failed_audit(audit, saved_summary, child["runs"])
        audits[width] = audit
        failed.extend(dict(record, name=f"width-{width}/" + record["name"]) for record in failures)
        for filename in REPORT_FILES:
            verify_file(report / filename, digest(report / filename), verified)
        for seed in SEEDS:
            fixture = inputs / f"width-{width}" / f"seed-{seed}"
            fixture_manifest = read_json(fixture / "manifest.json")
            require(
                fixture_manifest == child["fixtures"][str(seed)],
                "Fixture manifest identity differs",
            )
            verify_file(fixture / "manifest.json", digest(fixture / "manifest.json"), verified)
            config = struct.unpack("<9I", (fixture / "config.bin").read_bytes())
            require(
                config == (0x54465231, 2, 8, 16, 2048, 256, 256, seed, width),
                "Raw training budget changed",
            )
            for filename, value in fixture_manifest["sha256"].items():
                verify_file(fixture / filename, value, verified)
        runs[width] = []
        for entry, saved in zip(child["runs"], saved_summary["runs"], strict=True):
            fixture = inputs / f"width-{width}" / f"seed-{entry['seed']}"
            require(
                saved["fixture_sha256"] == child["fixtures"][str(entry["seed"])]["sha256"],
                "Run fixture hash mismatch",
            )
            run_dir = results / entry["relative_run"]
            run = raw_run_metrics(run_dir, fixture, entry, saved, child["model"], verified)
            if entry["implementation"] == "arm64":
                backend = read_backend(run_dir / "backend.u64", policy="auto")
                validate_runtime_backend(
                    backend,
                    child["fixtures"][str(entry["seed"])]["config"],
                    admitted=child["preflight"]["backend"],
                )
                require(
                    backend == entry["backend"] == saved["backend"]
                    and digest(run_dir / "backend.u64") == entry["backend_sha256"],
                    "Actual SME identity differs",
                )
            runs[width].append(run)
        for seed in SEEDS:
            for impl in IMPLEMENTATIONS:
                compare_repeats(
                    results / f"width-{width}/seed-{seed}-repeat-1-{impl}",
                    results / f"width-{width}/seed-{seed}-repeat-2-{impl}",
                    checks,
                )
        pairs[width] = []
        for seed in SEEDS:
            for repeat in (1, 2):
                pair = paired_predictions(
                    results / f"width-{width}/seed-{seed}-repeat-{repeat}-python",
                    results / f"width-{width}/seed-{seed}-repeat-{repeat}-arm64",
                )
                pairs[width].append(dict(pair, seed=seed, repeat=repeat))
        checks.require(f"outcome/width-{width}/original_audit_and_raw_metrics", True)
    baselines = count_baselines(inputs / "width-512" / "seed-42", checks)
    require(
        checks.passed,
        "Evidence integrity failed: " + str([r["name"] for r in checks.records if not r["passed"]]),
    )
    gates = [g for modes in manifest["regression_preflights"].values() for g in modes.values()]
    return {
        "manifest": manifest,
        "raw_manifest_sha256": digest(results / "manifest.json"),
        "width_manifests": children,
        "runs": runs,
        "paired_predictions": pairs,
        "baselines": baselines,
        "failed_checks": failed,
        "original_audits": audits,
        "integrity_checks": checks.records,
        "verified_files": verified,
        "measured_source_sha256": manifest["source_sha256"],
        "original_reports": {
            "parent": str(parent_failure.resolve()),
            **{str(w): str(p.resolve()) for w, p in width_reports.items()},
        },
        "gate_counts": manifest["regression_gate_counts"],
        "legacy_diagnostic_count": sum(g["legacy_diagnostic_count"] for g in gates),
        "legacy_failed_diagnostic_count": sum(len(g["legacy_failed_checks"]) for g in gates),
    }


def descriptive_summary(evidence):
    manifest = evidence["manifest"]
    by_width = {}
    for width, runs in evidence["runs"].items():
        aggregate = aggregate_runs(runs)
        means = aggregate["by_implementation"]
        by_width[str(width)] = {
            "model": manifest["models"][str(width)],
            "aggregate": aggregate,
            "forward_mean_p50_ratio_python_over_arm64": means["python"]["forward_p50_ms"]["mean"]
            / means["arm64"]["forward_p50_ms"]["mean"],
            "post_run_failed_check_count": len(evidence["original_audits"][width]["failed_checks"]),
            "baselines": evidence["baselines"],
            "paired_predictions": evidence["paired_predictions"][width],
            "prediction_disagreement": seed_statistics(
                [
                    float(
                        np.mean(
                            [
                                p["prediction_disagreement"]
                                for p in evidence["paired_predictions"][width]
                                if p["seed"] == seed
                            ]
                        )
                    )
                    for seed in SEEDS
                ]
            ),
        }
    config = manifest["fixtures"]["512"]["42"]["config"]
    return {
        "schema_version": 1,
        "status": "audit_failed",
        "protocol_version": "v5",
        "interpretation": (
            "Descriptive measurements from a failed numerical audit; "
            "no validated performance claim."
        ),
        "evidence_integrity_passed": True,
        "revision": manifest["revision"],
        "raw_manifest_sha256": evidence["raw_manifest_sha256"],
        "validation_contract": manifest["validation_contract"],
        "widths": list(WIDTHS),
        "fit_count": len(manifest["runs"]),
        "training_target_tokens_per_fit": (
            config["epochs"] * config["train_sequences"] * manifest["models"]["512"]["context"]
        ),
        "baselines": evidence["baselines"],
        "retained_test_prediction_rows": sum(
            r["quality"]["tokens"] for rs in evidence["runs"].values() for r in rs
        ),
        "post_run_check_count": sum(a["check_count"] for a in evidence["original_audits"].values()),
        "post_run_failed_check_count": len(evidence["failed_checks"]),
        "regression_gate_counts": evidence["gate_counts"],
        "legacy_diagnostic_count": evidence["legacy_diagnostic_count"],
        "legacy_failed_diagnostic_count": evidence["legacy_failed_diagnostic_count"],
        "descriptive_by_width": by_width,
        "runs": [
            dict(run, width=width) for width, runs in evidence["runs"].items() for run in runs
        ],
        "source_sha256": manifest["source_sha256"],
        "verified_files": evidence["verified_files"],
        "reporting_provenance": {
            "phase": "post-study",
            "purpose": "failed-study disclosure only",
            "script": "scripts/report_failed_sweep.py",
            "script_sha256": digest(__file__),
        },
    }


def charts(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"python": "#3568ad", "arm64": "#cc6b32"}
    labels = {"python": "PyTorch", "arm64": "ARM64 / SME"}
    metric_groups = [
        (
            "timing.png",
            [
                ("training_seconds", "Training + validation (s)"),
                ("forward_p50_ms", "Full-context forward p50 (ms)"),
            ],
        ),
        (
            "memory-quality.png",
            [
                ("peak_rss_training_mib", "Process peak RSS through training (MiB)"),
                ("test_nll", "Saved-prediction test NLL (nats)"),
            ],
        ),
    ]
    for filename, metrics in metric_groups:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
        for ax, (metric, title) in zip(axes, metrics, strict=True):
            for index, impl in enumerate(IMPLEMENTATIONS):
                stats = [
                    summary["descriptive_by_width"][str(w)]["aggregate"]["by_implementation"][impl][
                        metric
                    ]
                    for w in WIDTHS
                ]
                x = np.arange(len(WIDTHS)) + (index - 0.5) * 0.34
                bars = ax.bar(
                    x,
                    [s["mean"] for s in stats],
                    0.31,
                    yerr=[s["sample_sd"] for s in stats],
                    capsize=4,
                    color=colors[impl],
                    label=labels[impl],
                )
                for bar, stat in zip(bars, stats, strict=True):
                    ax.annotate(
                        f"{stat['mean']:.3f}"
                        if metric in ("test_nll", "forward_p50_ms")
                        else f"{stat['mean']:.2f}",
                        (bar.get_x() + bar.get_width() / 2, stat["mean"] + stat["sample_sd"]),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                    )
            ax.set_xticks(range(len(WIDTHS)), [str(w) for w in WIDTHS])
            ax.set_xlabel("Model width")
            ax.set_title(title, fontsize=11)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
            ax.grid(axis="y", alpha=0.18)
            ax.set_axisbelow(True)
        axes[0].legend(frameon=False)
        fig.suptitle(
            "AUDIT FAILED · descriptive observations only",
            color="#9a291f",
            fontsize=15,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.02,
            "Two repeats averaged within each seed; bars = mean ± sample SD across three seeds.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.93))
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def report_markdown(summary):
    timing, quality = [], []
    for width in WIDTHS:
        row = summary["descriptive_by_width"][str(width)]
        agg = row["aggregate"]
        for impl in IMPLEMENTATIONS:
            stats = agg["by_implementation"][impl]

            def fmt(name, n=3):
                return f"{stats[name]['mean']:.{n}f} ± {stats[name]['sample_sd']:.{n}f}"

            label = "PyTorch" if impl == "python" else "ARM64 / SME"
            timing.append(
                f"| {width} | {label} | {fmt('training_seconds')} | "
                f"{fmt('forward_p50_ms', 4)} | {fmt('peak_rss_training_mib', 2)} |"
            )
            quality.append(
                f"| {width} | {label} | {fmt('test_nll', 6)} | {fmt('test_accuracy_percent')} |"
            )
    ratio_lines = []
    for w in WIDTHS:
        row = summary["descriptive_by_width"][str(w)]
        ratio = row["aggregate"]["geometric_mean_training_ratio_python_over_arm64"]
        ratio_lines.append(
            f"- Width {w}: geometric mean of six paired training ratios **{ratio:.4f}**; "
            "ratio of mean forward p50 values "
            f"**{row['forward_mean_p50_ratio_python_over_arm64']:.4f}**."
        )
    ratios = "\n".join(ratio_lines)
    failed_512 = summary["descriptive_by_width"]["512"]["post_run_failed_check_count"]
    failed_1024 = summary["descriptive_by_width"]["1024"]["post_run_failed_check_count"]
    return f"""# Wider transformer study: audit failed

**The completed 24-fit campaign failed its frozen post-run checkpoint audit. No validated speed
or numerical-equivalence claim is made.** This reporting-only document was created after the
failed audits to disclose all outcomes. It does not replace either original report, relax a
threshold, reclassify a failed check, or retrain a model after inspecting test results.

Measured source: `{summary["revision"]}`. Widths 512 and 1024 contain 2,251,392 and 8,696,960
parameters. The known engineering benchmark reused the same data partitions, 32-character
contexts, seeds 42/43/44, eight epochs, batch size 16, 1,024 updates, and 524,288 training
target characters per fit. There were two repeats of each implementation per seed and width, in
the prespecified balanced order with 15-second pauses. Each process used one CPU compute thread.

## Numerical result and retained failures

The original per-width post-run audits contain **{summary["post_run_check_count"]:,} checks**,
of which **{summary["post_run_failed_check_count"]} failed**: {failed_512} at width 512 and
{failed_1024} at width 1024. Each failed check compares a saved native checkpoint's prediction
for one 32-character sequence with an independent functional PyTorch forward pass at the **same
weights**. The original forward tolerance remains `atol=2e-5, rtol=2e-4`. These are post-run
numerical failures, distinct from divergence between independently trained models. Repeated runs
reproduce the same failures; the count includes both repeats.

Before measurement, all **{summary["regression_gate_counts"]["all"]:,} blocking regression
checks** passed across six widths and auto, forced NEON, and forced SME policies. The
prospective wide-model contract also retained **{summary["legacy_diagnostic_count"]} original
free-running diagnostic comparisons**, including **{summary["legacy_failed_diagnostic_count"]}
failures**, explicitly excluded from passing gate counts. Passing those synthetic admission
checks did not guarantee the later trained-checkpoint audit would pass.

Original records are copied byte-for-byte under [original-reports/](original-reports/); all
six-width regression gates remain under [regression-preflights/](regression-preflights/).
[audit.json](audit.json) retains all post-run failure records and separates evidence-integrity
verification from numerical success. [summary.json](summary.json) retains all 24 run metrics and
artifact hashes. Neither file claims that the numerical audit passed.

## Descriptive timing and memory observations

| Width | Implementation | Train + val, s | Forward p50, ms | Training RSS, MiB |
|---|---|---:|---:|---:|
{chr(10).join(timing)}

Values are means ± sample SD of three seed means, after averaging two repeats within each seed.
Training time includes training and periodic validation. Forward timing is batch-one inference
over the complete 32-character context: 20 warmups and 100 timed samples per run, including
attention softmax but excluding output-probability softmax, loss, sampling, and I/O. It is not
cached single-token generation. RSS is the whole process lifetime peak through training,
including framework/runtime and data, not parameter-only memory.

Ratios below are **PyTorch seconds / ARM64 seconds**, retained as descriptive arithmetic only.
Values below one mean the recorded ARM64 times were longer.

{ratios}

![Descriptive timing; audit failed](timing.png)

## Descriptive saved-prediction quality

| Width | Implementation | Test NLL, nats | Next-character accuracy, % |
|---|---|---:|---:|
{chr(10).join(quality)}

These metrics are float64 calculations directly from the
{summary["retained_test_prediction_rows"]:,} retained prediction rows and fixed labels. They
describe the emitted predictions; independent FP32 reference checkpoint replay exceeded tolerance.
This is reused engineering evaluation, not a fresh scientific assessment or a
capacity-optimization experiment. Differences between final models are descriptive, not evidence
of equivalent learning trajectories.

The JSON summary also retains perplexity, bits per character, paired prediction disagreement,
the uniform baseline, and a Laplace-smoothed unigram baseline fitted only to training targets.

![Descriptive memory and quality; audit failed](memory-quality.png)

## Evidence scope

The reporting script authenticates the measured Git source, all input and binary/layout/gate
hashes, all raw output hashes, exact run identities/order/budgets, matching
within-implementation repeats, and actual backend counters. Every native run records 94,040 SME
forward and 81,920 SME backward dense calls, with zero NEON dense calls. Counts identify
dispatched kernels, not elapsed-time attribution.

The original failed study and all raw artifacts remain preserved. This document introduces no
new model execution or performance measurement. Evidence integrity passing means the failed
outcome is faithfully reproduced; it does not certify numerical correctness. The outcome applies
to this machine, short-context CPU workload, fixed token budget, and eager PyTorch baseline. It
makes no claim about general transformer efficiency, GPU performance, energy, cached decoding,
or production serving.
"""


def failed_outcome_audit(evidence):
    """Separate the original numerical failures from current byte-integrity checks."""
    return {
        "schema_version": 1,
        "status": "audit_failed",
        "passed": False,
        "check_count": sum(a["check_count"] for a in evidence["original_audits"].values()),
        "failed_check_count": len(evidence["failed_checks"]),
        "failed_checks": evidence["failed_checks"],
        "evidence_integrity_passed": True,
        "integrity_check_count": len(evidence["integrity_checks"]),
        "integrity_checks": evidence["integrity_checks"],
        "legacy_diagnostic_count": evidence["legacy_diagnostic_count"],
        "legacy_failed_diagnostic_count": evidence["legacy_failed_diagnostic_count"],
    }


def validate_failed_report(report_dir, authenticated):
    """Require the new outcome's data to match independently authenticated evidence."""
    report_dir = Path(report_dir)
    summary = descriptive_summary(authenticated)
    require(read_json(report_dir / "summary.json") == summary, "Outcome summary was altered")
    require(
        read_json(report_dir / "audit.json") == failed_outcome_audit(authenticated),
        "Outcome audit was altered",
    )
    require(
        (report_dir / "README.md").read_text() == report_markdown(summary),
        "Outcome narrative was altered",
    )
    for label, source in authenticated["original_reports"].items():
        directory = "parent" if label == "parent" else f"width-{label}"
        filenames = (*REPORT_FILES, *(("report-processes.json",) if label == "parent" else ()))
        for filename in filenames:
            require(
                digest(report_dir / "original-reports" / directory / filename)
                == digest(Path(source) / filename),
                "Original failure copy was altered",
            )
    for width, modes in authenticated["manifest"]["regression_preflights"].items():
        for mode, gate in modes.items():
            filename = f"width-{width}{'' if mode == 'auto' else '-' + mode}.json"
            require(
                read_json(report_dir / "regression-preflights" / filename) == gate,
                "Original gate copy was altered",
            )
    for filename in ("timing.png", "memory-quality.png"):
        require((report_dir / filename).is_file(), "Outcome figure missing")
    return summary


def create_report(results, inputs, parent_failure, width_reports, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evidence = authenticate_completed_failed_campaign(
        results, inputs, parent_failure, width_reports
    )
    summary = descriptive_summary(evidence)
    for label, source in evidence["original_reports"].items():
        destination = (
            output / "original-reports" / ("parent" if label == "parent" else f"width-{label}")
        )
        destination.mkdir(parents=True)
        for filename in (*REPORT_FILES, *(("report-processes.json",) if label == "parent" else ())):
            shutil.copyfile(Path(source) / filename, destination / filename)
    gates_dir = output / "regression-preflights"
    gates_dir.mkdir()
    for width, modes in evidence["manifest"]["regression_preflights"].items():
        for mode, gate in modes.items():
            filename = f"width-{width}{'' if mode == 'auto' else '-' + mode}.json"
            source = Path(evidence["manifest"]["preflight_directory"]) / filename
            require(
                read_json(source) == gate, "Regression gate copy differs from admitted manifest"
            )
            shutil.copyfile(source, gates_dir / filename)
    charts(summary, output)
    write_json(output / "summary.json", summary)
    write_json(output / "audit.json", failed_outcome_audit(evidence))
    (output / "README.md").write_text(report_markdown(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "results",
        "inputs",
        "parent-failure",
        "width-512-report",
        "width-1024-report",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = create_report(
            args.results,
            args.inputs,
            args.parent_failure,
            {512: args.width_512_report, 1024: args.width_1024_report},
            args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "report_created": False,
                    "status": "audit_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "report_created": True,
                "status": "audit_failed",
                "evidence_integrity_passed": True,
                "post_run_failed_check_count": summary["post_run_failed_check_count"],
            }
        )
    )
    return 0  # Reporting succeeded; the explicit study status remains audit_failed.


if __name__ == "__main__":
    raise SystemExit(main())
