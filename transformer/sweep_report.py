"""Audit every protocol-declared width-sweep fit and retain per-width and aggregate reports.

Each width's mathematical audit runs in a separate process selected before imports.
This module never trains a model or performs performance measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from transformer import spec
from transformer.backend import (
    WIDE_VALIDATION_CONTRACT,
    validate_backend_preflights,
    validate_precision_preflight,
    validate_regression_preflights,
    validate_runtime_backend,
    validation_contract,
)
from transformer.prepare import SOURCE_SHA256
from transformer.report import COLORS, IMPLEMENTATIONS, LABELS, aggregate_runs, seed_statistics
from transformer.sweep import (
    REGRESSION_WIDTHS,
    child_environment,
    on_ac_power,
    protocol_widths,
    study_order,
)
from transformer.validate import ROOT, Checks, sha256

REPORT_SOURCES = (
    "transformer/sweep_report.py",
    "transformer/report.py",
    "transformer/audit.py",
    "transformer/sweep.py",
)
TRAINING_TOKENS_PER_FIT = spec.EPOCHS * spec.TRAIN_SEQUENCES * spec.CONTEXT
TEST_ROWS_PER_FIT = spec.TEST_SEQUENCES * spec.CONTEXT
OPTIMIZER_STEPS = spec.EPOCHS * math.ceil(spec.TRAIN_SEQUENCES / spec.BATCH)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(checks, name, condition):
    checks.require(name, condition)
    if not condition:
        raise ValueError(name)


def identity(entry):
    return (entry["width"], entry["seed"], entry["repeat"], entry["implementation"])


def v6_qualification(summaries):
    """Numerical limitations never masquerade as qualified matched results."""
    summaries = list(summaries)
    fields = (
        "evidence_integrity_passed",
        "native_fp64_accuracy_passed",
        "python_fp64_accuracy_passed",
        "cross_implementation_fp32_compatibility_passed",
    )
    if not summaries or any(
        type(summary.get(key)) is not bool for summary in summaries for key in fields
    ):
        raise ValueError("Incomplete v6 numerical qualification fields")
    result = {key: all(summary[key] for summary in summaries) for key in fields}
    result["numerically_qualified"] = all(result[key] for key in fields[:3])
    result["passed"] = result["numerically_qualified"]
    result["status"] = (
        "audit_failed"
        if not result["evidence_integrity_passed"]
        else "native_precision_limit"
        if not result["native_fp64_accuracy_passed"]
        else "baseline_precision_limit"
        if not result["python_fp64_accuracy_passed"]
        else "completed"
    )
    return result


def validate_v6_numerical_report(summary, audit):
    """Derive qualification from retained comparisons, never from a passed flag alone."""
    from scripts.package_sweep_evidence import check_run_numerical_evidence
    from transformer.precision_audit import numerical_status, summarize_numerical_runs

    for run in summary["runs"]:
        check_run_numerical_evidence(run)
    categories = summarize_numerical_runs(summary["runs"])
    expected = numerical_status(True, categories)
    if any(summary.get(key) != value or audit.get(key) != value for key, value in expected.items()):
        raise ValueError("V6 numerical qualification disagrees with retained records")
    flat = [
        record | {"category": category}
        for category, group in categories.items()
        for record in group["checks"]
    ]
    compact = {
        category: {key: value for key, value in group.items() if key != "checks"}
        for category, group in categories.items()
    }
    if (
        audit.get("numerical_checks") != flat
        or audit.get("numerical_categories") != compact
        or summary.get("numerical_categories") != compact
        or audit.get("numerical_check_count") != len(flat)
        or summary.get("numerical_check_count") != len(flat)
        or audit.get("passed") is not expected["numerically_qualified"]
    ):
        raise ValueError("V6 numerical records or counts are incomplete")
    return flat


def validate_manifest(manifest, width_manifests, checks):
    """Reject incomplete, duplicated or internally inconsistent campaign records."""
    widths = protocol_widths(manifest.get("protocol_version"))
    require(
        checks,
        "manifest/completed_width_protocol",
        manifest.get("status") == "completed"
        and manifest.get("protocol_version") in ("v3", "v4", "v5", "v6"),
    )
    require(checks, "manifest/protocol_widths", manifest.get("widths") == list(widths))
    if manifest["protocol_version"] == "v6":
        for field in ("precision_contract", "postrun_numerical_contract"):
            require(
                checks,
                "manifest/" + field,
                manifest.get(field) == "v6-fp64-centered-attention-v1"
                and all(child.get(field) == manifest[field] for child in width_manifests.values()),
            )
        for width in widths:
            for gate in manifest.get("regression_preflights", {}).get(str(width), {}).values():
                validate_precision_preflight(gate, width)
    require(
        checks,
        "manifest/source_identity",
        bool(manifest.get("revision")) and bool(manifest.get("source_sha256")),
    )
    expected_order = study_order(manifest["protocol_version"])
    observed = [identity(entry) for entry in manifest["runs"]]
    require(
        checks,
        "manifest/exact_run_identities",
        len(observed) == len(expected_order)
        and len(set(observed)) == len(observed)
        and set(observed) == set(expected_order),
    )
    require(
        checks,
        "manifest/frozen_order",
        observed == expected_order and [tuple(row) for row in manifest["order"]] == expected_order,
    )
    require(checks, "manifest/width_reports", set(width_manifests) == set(widths))
    require(
        checks,
        "manifest/models_and_fixtures",
        all(
            set(manifest[key]) == {str(width) for width in widths}
            for key in ("models", "fixtures", "preflights")
        ),
    )
    for entry in manifest["runs"]:
        width, seed, repeat, impl = identity(entry)
        name = f"seed-{seed}-repeat-{repeat}-{impl}"
        tag = f"width-{width}/{name}"
        require(
            checks,
            tag + "/identity",
            entry["name"] == name
            and entry["relative_run"] == tag
            and entry["width_environment"] == str(width),
        )
        require(
            checks,
            tag + "/completed",
            entry.get("status") == "completed" and entry.get("returncode") == 0,
        )
        require(
            checks,
            tag + "/conditions",
            all(
                entry.get(key) is True
                for key in (
                    "conditions_valid",
                    "source_unchanged",
                    "binary_unchanged",
                    "fixtures_unchanged",
                )
            ),
        )
        require(
            checks,
            tag + "/before_after_evidence",
            on_ac_power(entry["before"])
            and on_ac_power(entry["after"])
            and entry["source_before"]["unchanged"] is True
            and entry["source_after"]["unchanged"] is True
            and all(
                entry[key] == []
                for key in (
                    "changed_files_before",
                    "changed_authenticated_files",
                    "changed_fixture_files",
                )
            ),
        )
        require(
            checks,
            tag + "/process_timer",
            math.isfinite(entry["process_wall_seconds"]) and entry["process_wall_seconds"] > 0,
        )
    for width in widths:
        key = str(width)
        child = width_manifests[width]
        model = manifest["models"][key]
        required_model = {
            "width": width,
            "heads": 4,
            "head_width": width // 4,
            "hidden": 2 * width,
            "context": 32,
            "vocab": 128,
            "parameters": 8 * width**2 + 301 * width + 128,
        }
        require(
            checks,
            f"width-{width}/model",
            all(model.get(name) == value for name, value in required_model.items()),
        )
        require(
            checks,
            f"width-{width}/manifest_status",
            child.get("status") == "completed"
            and child.get("protocol_version") == manifest["protocol_version"]
            and child.get("width") == width
            and child.get("model_width") == width
            and child.get("parameter_count") == model["parameters"],
        )
        for field, expected in (
            ("model", model),
            ("revision", manifest["revision"]),
            ("source_sha256", manifest["source_sha256"]),
            ("preflight", manifest["preflights"][key]),
            ("fixtures", manifest["fixtures"][key]),
            ("runs", [entry for entry in manifest["runs"] if entry["width"] == width]),
            ("order", [list(row[1:]) for row in expected_order if row[0] == width]),
        ):
            require(
                checks, f"width-{width}/manifest_matches_global/{field}", child[field] == expected
            )
        if manifest["protocol_version"] in ("v4", "v5", "v6"):
            supplements = manifest["backend_preflights"][key]
            validate_backend_preflights(manifest["preflights"][key], supplements)
            require(
                checks, f"width-{width}/backend_gates", child["backend_preflights"] == supplements
            )
            for entry in child["runs"]:
                if entry["implementation"] == "arm64":
                    validate_runtime_backend(
                        entry["backend"],
                        child["fixtures"][str(entry["seed"])]["config"],
                        admitted=child["preflight"]["backend"],
                    )
                    require(
                        checks,
                        f"width-{width}/{entry['name']}/backend_admitted",
                        entry.get("backend_valid") is True
                        and bool(entry.get("backend_sha256"))
                        and entry.get("backend_environment") == "auto",
                    )
        preflight = manifest["preflights"][key]
        records = preflight.get("checks", [])
        require(
            checks,
            f"width-{width}/preflight",
            preflight.get("passed") is True
            and preflight.get("runtime_included") is True
            and preflight.get("model_width") == width
            and preflight.get("parameter_count") == model["parameters"]
            and preflight.get("model_identity")
            == {"width": width, "parameter_count": model["parameters"], "verified": True}
            and bool(records)
            and preflight.get("check_count") == len(records)
            and all(item.get("passed") is True for item in records)
            and preflight.get("failed_checks") == []
            and preflight.get("error") is None,
        )
    if manifest["protocol_version"] in ("v5", "v6"):
        require(
            checks,
            "manifest/validation_contract",
            (
                manifest.get("validation_contract") == WIDE_VALIDATION_CONTRACT
                and all(
                    child.get("validation_contract") == WIDE_VALIDATION_CONTRACT
                    for child in width_manifests.values()
                )
            )
            if manifest["protocol_version"] == "v5"
            else (
                manifest.get("validation_contract") == "v6-width-specific-v1"
                and manifest.get("validation_contracts")
                == {str(width): validation_contract(width) for width in widths}
                and all(
                    child.get("validation_contract") == validation_contract(width)
                    for width, child in width_manifests.items()
                )
            ),
        )
        counts = validate_regression_preflights(
            manifest["regression_preflights"], REGRESSION_WIDTHS, manifest["source_sha256"]
        )
        require(
            checks,
            "manifest/regression_widths",
            manifest.get("regression_widths") == list(REGRESSION_WIDTHS)
            and manifest.get("regression_gate_counts") == counts,
        )
        for width in widths:
            modes = manifest["regression_preflights"][str(width)]
            require(
                checks,
                f"width-{width}/regression_gate_alignment",
                modes["auto"] == manifest["preflights"][str(width)]
                and {mode: gate for mode, gate in modes.items() if mode != "auto"}
                == manifest["backend_preflights"][str(width)],
            )
    validate_shared_fixtures(manifest["fixtures"], checks, widths)


def validate_shared_fixtures(fixtures, checks, widths=None):
    widths = tuple(sorted(map(int, fixtures))) if widths is None else widths
    base = fixtures[str(widths[0])]
    first = base[str(spec.SEEDS[0])]
    for width in widths:
        require(
            checks,
            f"fixtures/{width}/seeds",
            set(fixtures[str(width)]) == {str(seed) for seed in spec.SEEDS},
        )
        for seed in spec.SEEDS:
            fixture = fixtures[str(width)][str(seed)]
            tag = f"fixtures/{width}/{seed}"
            expected_config = {
                "version": 2,
                "width": width,
                "seed": seed,
                "epochs": spec.EPOCHS,
                "batch": spec.BATCH,
                "train_sequences": spec.TRAIN_SEQUENCES,
                "validation_sequences": spec.VALID_SEQUENCES,
                "test_sequences": spec.TEST_SEQUENCES,
            }
            require(
                checks,
                tag + "/configuration",
                all(
                    fixture["config"].get(name) == value for name, value in expected_config.items()
                ),
            )
            require(
                checks,
                tag + "/source",
                fixture["source"]["sha256"] == SOURCE_SHA256
                and fixture["source"]["sha256"] == first["source"]["sha256"],
            )
            require(
                checks,
                tag + "/partitions",
                fixture["splits"] == first["splits"]
                and fixture["vocabulary"] == first["vocabulary"]
                and fixture["sequence_rule"] == first["sequence_rule"],
            )
            for name in ("train.i32", "validation.i32", "test.i32"):
                require(
                    checks, tag + "/same_" + name, fixture["sha256"][name] == first["sha256"][name]
                )
            require(
                checks,
                tag + "/same_seed_schedule",
                fixture["sha256"]["schedule.i32"] == base[str(seed)]["sha256"]["schedule.i32"],
            )
    for seed in spec.SEEDS:
        require(
            checks,
            f"fixtures/{seed}/distinct_width_initializations",
            len({fixtures[str(width)][str(seed)]["sha256"]["initial.f32"] for width in widths})
            == len(widths),
        )


def aggregate_summaries(manifest, summaries, checks):
    """Recompute aggregations from all runs, with no pooling across model widths."""
    widths = protocol_widths(manifest["protocol_version"])
    require(checks, "reports/all_widths", set(summaries) == set(widths))
    by_width, all_runs = {}, []
    expected_runs = {identity(entry): entry for entry in manifest["runs"]}
    first_baseline = summaries[widths[0]]["baselines"]
    for width in widths:
        summary = summaries[width]
        tag = f"report/width-{width}"
        require(
            checks,
            tag + "/completed",
            (
                summary.get("status") == "completed"
                or (
                    manifest["protocol_version"] == "v6"
                    and summary.get("status")
                    in ("baseline_precision_limit", "native_precision_limit")
                    and summary.get("evidence_integrity_passed") is True
                )
            )
            and summary.get("protocol_version") == manifest["protocol_version"],
        )
        require(
            checks,
            tag + "/provenance",
            summary["revision"] == manifest["revision"]
            and summary["source_sha256"] == manifest["source_sha256"]
            and summary["model"] == manifest["models"][str(width)],
        )
        require(checks, tag + "/baseline_identity", summary["baselines"] == first_baseline)
        aggregate = aggregate_runs(summary["runs"])
        require(checks, tag + "/aggregation_recomputed", aggregate == summary["aggregate"])
        for run in summary["runs"]:
            key = (width, run["seed"], run["repeat"], run["implementation"])
            entry = expected_runs[key]
            require(
                checks,
                f"{tag}/{run['name']}/runner_alignment",
                run["name"] == entry["name"]
                and run["process_wall_seconds"] == entry["process_wall_seconds"]
                and run["fixture_sha256"]
                == manifest["fixtures"][str(width)][str(run["seed"])]["sha256"],
            )
            require(
                checks,
                f"{tag}/{run['name']}/fixed_budget",
                run["optimizer_steps"] == OPTIMIZER_STEPS
                and run["quality"]["tokens"] == TEST_ROWS_PER_FIT
                and len(run["learning_history"]) == spec.EPOCHS
                and run["checkpoint_predictions_recomputed"] is True,
            )
            all_runs.append(run | {"width": width, "relative_run": entry["relative_run"]})
        retained_rows = sum(run["quality"]["tokens"] for run in summary["runs"])
        require(
            checks, tag + "/row_count", summary["retained_test_prediction_rows"] == retained_rows
        )
        require(
            checks,
            tag + "/preflight_count",
            summary["preflight_check_count"] == manifest["preflights"][str(width)]["check_count"],
        )
        if manifest["protocol_version"] in ("v5", "v6"):
            require(
                checks,
                tag + "/validation_contract",
                summary.get("validation_contract")
                == (
                    validation_contract(width)
                    if manifest["protocol_version"] == "v6"
                    else WIDE_VALIDATION_CONTRACT
                ),
            )
        if manifest["protocol_version"] in ("v4", "v5", "v6"):
            forced = validate_backend_preflights(
                manifest["preflights"][str(width)], manifest["backend_preflights"][str(width)]
            )
            require(
                checks,
                tag + "/forced_backend_counts",
                summary.get("forced_backend_preflight_check_count") == forced
                and summary.get("all_backend_preflight_check_count")
                == forced + summary["preflight_check_count"]
                and summary.get("backend") == manifest["preflights"][str(width)]["backend"],
            )
        throughput = {}
        for impl in IMPLEMENTATIONS:
            seed_values = [
                float(
                    np.mean(
                        [
                            TRAINING_TOKENS_PER_FIT / run["training_seconds"]
                            for run in summary["runs"]
                            if run["implementation"] == impl and run["seed"] == seed
                        ]
                    )
                )
                for seed in spec.SEEDS
            ]
            throughput[impl] = seed_statistics(seed_values)
        timings = aggregate["by_implementation"]
        by_width[str(width)] = {
            "model": summary["model"],
            "aggregate": aggregate,
            "training_target_tokens_per_second": throughput,
            "forward_p50_mean_ratio_python_over_arm64": timings["python"]["forward_p50_ms"]["mean"]
            / timings["arm64"]["forward_p50_ms"]["mean"],
            "paired_predictions": summary["paired_predictions"],
            "audit_check_count": summary["audit_check_count"],
            "preflight_check_count": summary["preflight_check_count"],
            "forced_backend_preflight_check_count": summary.get(
                "forced_backend_preflight_check_count", 0
            ),
            "backend": summary.get("backend"),
            "legacy_diagnostic_counts": summary.get("legacy_diagnostic_counts", {}),
            "retained_test_prediction_rows": retained_rows,
            "raw_manifest_sha256": summary["raw_manifest_sha256"],
        }
        if manifest["protocol_version"] == "v6":
            by_width[str(width)].update(v6_qualification([summary]))
            by_width[str(width)]["numerical_categories"] = summary["numerical_categories"]
            by_width[str(width)]["numerical_check_count"] = summary["numerical_check_count"]
    require(
        checks,
        "reports/all_test_rows_recomputed",
        len(all_runs) == len(expected_runs)
        and sum(item["quality"]["tokens"] for item in all_runs)
        == len(expected_runs) * TEST_ROWS_PER_FIT,
    )
    return {
        "by_width": by_width,
        "runs": all_runs,
        "baselines": first_baseline,
        "retained_test_prediction_rows": sum(item["quality"]["tokens"] for item in all_runs),
    }


def charts(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False}
    )
    widths = np.array(summary["widths"])

    def decorate(axis):
        axis.set_xscale("log", base=2)
        axis.set_xticks(widths, [str(width) for width in widths])
        axis.set_xlabel("Model width (four attention heads)")
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)

    def metric(axis, field, title, ylabel, *, zero=False, log=False):
        for index, impl in enumerate(IMPLEMENTATIONS):
            stats = [
                summary["by_width"][str(width)]["aggregate"]["by_implementation"][impl][field]
                for width in widths
            ]
            axis.errorbar(
                widths,
                [item["mean"] for item in stats],
                yerr=[item["sample_sd"] for item in stats],
                marker="o",
                capsize=4,
                color=COLORS[impl],
                label=LABELS[impl],
                linewidth=2,
            )
            for width, item in zip(widths, stats, strict=True):
                axis.scatter(
                    width * (1 + np.linspace(-0.025, 0.025, 3) + (index - 0.5) * 0.02),
                    item["seed_values"],
                    s=13,
                    alpha=0.65,
                    color=COLORS[impl],
                    zorder=4,
                )
        decorate(axis)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        if zero:
            axis.set_ylim(bottom=0)
        if log:
            axis.set_yscale("log")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    metric(axes[0], "training_seconds", "Training + validation per fit", "Seconds", zero=True)
    metric(axes[1], "forward_p50_ms", "Full 32-character forward, p50", "Milliseconds", zero=True)
    axes[0].legend(frameon=False)
    title = "Matched transformer cost across model widths"
    if summary["protocol_version"] == "v6":
        title += f"\nNumerical status: {summary['status']}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Means ± sample SD of three seed means; dots: seeds. Two repeats averaged within seed.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    fig.savefig(output / "scaling.png")
    plt.close(fig)

    if summary["protocol_version"] in ("v4", "v5", "v6"):
        training = [
            summary["by_width"][str(width)]["aggregate"][
                "geometric_mean_training_ratio_python_over_arm64"
            ]
            for width in widths
        ]
        forward = [
            summary["by_width"][str(width)]["forward_p50_mean_ratio_python_over_arm64"]
            for width in widths
        ]
        fig, axis = plt.subplots(figsize=(10, 5.8))
        positions = np.arange(len(widths))
        for values, offset, color, label in (
            (training, -0.18, "#3568ad", "Training: geometric mean of six paired ratios"),
            (forward, 0.18, "#d66b34", "Forward p50: ratio of implementation means"),
        ):
            bars = axis.barh(positions + offset, values, height=0.30, color=color, label=label)
            labels = axis.bar_label(
                bars, labels=[f"{value:.3f}×" for value in values], padding=6, fontsize=10
            )
            for label in labels:
                label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 1})
        axis.axvline(1, color="#333333", linestyle="--", linewidth=1)
        axis.set_yticks(positions, [f"Width {width}" for width in widths])
        axis.invert_yaxis()
        axis.set_xlim(0, max(1, *training, *forward) * 1.20)
        axis.set_xlabel("PyTorch time / ARM64 time", labelpad=10)
        title = "Implementation time ratios at every width"
        if summary["protocol_version"] == "v6":
            title += f"\nNumerical status: {summary['status']}; see per-width qualification"
        axis.set_title(title, pad=16, weight="bold", fontsize=12)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)
        fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.065), frameon=False, fontsize=9)
        fig.text(
            0.5,
            0.025,
            "Above 1 favors assembly; below 1 favors PyTorch. "
            f"{len(summary['runs'])} fits · 3 seeds · 2 repeats · 1 CPU thread",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        fig.savefig(output / "speedup.png")
        plt.close(fig)
    else:
        fig, axis = plt.subplots(figsize=(9, 5))
        training = [
            summary["by_width"][str(w)]["aggregate"][
                "geometric_mean_training_ratio_python_over_arm64"
            ]
            for w in widths
        ]
        forward = [
            summary["by_width"][str(w)]["forward_p50_mean_ratio_python_over_arm64"] for w in widths
        ]
        for values, label, color, marker in (
            (training, "Training: geometric mean of six paired ratios", "#3568ad", "o"),
            (forward, "Forward p50: ratio of implementation means", "#d66b34", "s"),
        ):
            axis.plot(widths, values, marker=marker, color=color, label=label, linewidth=2)
            for width, value in zip(widths, values, strict=True):
                axis.annotate(
                    f"{value:.2f}×",
                    (width, value),
                    textcoords="offset points",
                    xytext=(0, 8 if marker == "o" else -16),
                    ha="center",
                    color=color,
                )
        axis.axhline(1, color="#333333", linestyle="--", linewidth=1, label="Equal time")
        decorate(axis)
        axis.set_ylim(bottom=0, top=max(1, *training, *forward) * 1.18)
        axis.set_ylabel("PyTorch time / ARM64 time")
        axis.set_title("Implementation time ratios at each width", fontweight="bold")
        axis.legend(frameon=False, fontsize=9)
        fig.text(
            0.5,
            0.015,
            "Above 1 favors assembly; below 1 favors PyTorch. "
            "Descriptive ratios, not confidence intervals.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(output / "speedup.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metric(
        axes[0],
        "peak_rss_training_mib",
        "Process peak RSS through training",
        "MiB (log scale)",
        log=True,
    )
    metric(axes[1], "test_nll", "Test negative log-likelihood", "NLL; lower is better")
    metric(axes[2], "test_accuracy_percent", "Next-character accuracy", "Percent")
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Memory and quality on the reused engineering benchmark", fontsize=15, fontweight="bold"
    )
    fig.text(
        0.5,
        0.015,
        "Means ± sample SD; dots: seeds. RSS includes framework and runtime. "
        "Width changes model capacity.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    fig.savefig(output / "memory-quality.png")
    plt.close(fig)

    if summary["protocol_version"] == "v6":
        fig, axis = plt.subplots(figsize=(10, 5.5))
        positions = np.arange(len(widths))
        for category, offset, color, label in (
            ("native_fp64", -0.18, COLORS["arm64"], "Native vs FP64"),
            ("python_fp64", 0.18, COLORS["python"], "PyTorch vs FP64"),
        ):
            values = [
                summary["by_width"][str(width)]["numerical_categories"][category][
                    "failed_logit_coordinates"
                ]
                for width in widths
            ]
            bars = axis.bar(positions + offset, values, width=0.32, color=color, label=label)
            axis.bar_label(bars, padding=3, fontsize=9)
        axis.set_xticks(positions, [str(width) for width in widths])
        axis.set_xlabel("Model width")
        axis.set_ylabel("Logit coordinates outside the fixed FP64 error limit")
        axis.set_yscale("symlog", linthresh=1)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
        axis.legend(frameon=False)
        axis.set_title("Accuracy is evaluated separately for both implementations", weight="bold")
        fig.text(
            0.5,
            0.025,
            "Same weights at every selected checkpoint, "
            "including both training engines and repeats.\n"
            "Forward limit: 2e-5 + 2e-4 × |FP64 reference|. Zero means no coordinates exceeded it.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(output / "numerical-accuracy.png")
        plt.close(fig)


def report_markdown(summary):
    def stat(record, field, digits=3):
        data = record[field]
        return f"{data['mean']:.{digits}f} ± {data['sample_sd']:.{digits}f}"

    train, forward, quality = [], [], []
    for width in summary["widths"]:
        item = summary["by_width"][str(width)]
        aggregate = item["aggregate"]["by_implementation"]
        ratio = item["aggregate"]["geometric_mean_training_ratio_python_over_arm64"]
        train.append(
            f"| {width} | {item['model']['parameters']:,} | "
            f"{stat(aggregate['python'], 'training_seconds')} | "
            f"{stat(aggregate['arm64'], 'training_seconds')} | {ratio:.3f}× |"
        )
        forward.append(
            f"| {width} | {stat(aggregate['python'], 'forward_p50_ms', 4)} | "
            f"{stat(aggregate['arm64'], 'forward_p50_ms', 4)} | "
            f"{item['forward_p50_mean_ratio_python_over_arm64']:.3f}× |"
        )
        for impl in IMPLEMENTATIONS:
            throughput = item["training_target_tokens_per_second"][impl]
            quality.append(
                f"| {width} | {LABELS[impl]} | {throughput['mean'] / 1000:.1f} "
                f"± {throughput['sample_sd'] / 1000:.1f} | "
                f"{stat(aggregate[impl], 'peak_rss_training_mib', 2)} | "
                f"{stat(aggregate[impl], 'test_nll', 4)} | "
                f"{stat(aggregate[impl], 'test_accuracy_percent')} |"
            )
    evidence = "\n".join(
        f"- [Width {width}: full report](width-{width}/README.md), "
        f"[audit](width-{width}/audit.json) "
        f"({summary['by_width'][str(width)]['audit_check_count']:,} checks), "
        f"[preflight](width-{width}/preflight.json) "
        f"({summary['by_width'][str(width)]['preflight_check_count']:,} checks)."
        for width in summary["widths"]
    )
    backend_note = ""
    prior = {"v3": "v1/v2", "v4": "v1/v2/v3", "v5": "v1/v2/v3/v4", "v6": "v1–v5"}[
        summary["protocol_version"]
    ]
    if summary["protocol_version"] in ("v4", "v5", "v6"):
        rows = []
        for width in summary["widths"]:
            native = next(
                run["backend"]
                for run in summary["runs"]
                if run["width"] == width and run["implementation"] == "arm64"
            )
            rows.append(
                f"| {width} | {native['sme_forward_calls']:,} / "
                f"{native['sme_backward_calls']:,} | {native['neon_forward_calls']:,} / "
                f"{native['neon_backward_calls']:,} |"
            )
        backend_note = (
            "All measured native processes requested automatic dispatch. The table "
            "records actual dense calls per fit, including validation, final test and "
            "inference samples; it does not imply every transformer operation used SME. "
            "Hardware capability, vector length and exact counters are authenticated "
            "against each width's numerical gate.\n\n"
            "| Width | SME forward / backward | NEON forward / backward |\n"
            "|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
        )
    regression_note = ""
    if summary["protocol_version"] == "v5":
        regression_note = (
            f"Validation contract: `{summary['validation_contract']}`. "
            "The wide-model gates compare both engines at identical evolving states and replay "
            "each engine's gradients through both optimizers using unchanged tolerances. "
            f"The original free-running comparisons are retained as "
            f"{summary['legacy_diagnostic_count']:,} diagnostic records, including "
            f"{summary['legacy_failed_diagnostic_count']:,} failures. These are excluded from the "
            "passing gate counts; the original wide trajectory contract is not claimed to pass. "
            "Independent four-update control replays require exact own saved states/history "
            "and verify actual partial-batch normalization plus identical-input Adam updates. "
            "The failed initial admission remains preserved. "
            "The measured-width admission counts below cover widths 512 and 1024 only. "
            f"The complete six-width correctness regression contains "
            f"{summary['regression_all_backend_preflight_check_count']:,} checks: "
            f"{summary['regression_preflight_check_count']:,} automatic and "
            f"{summary['regression_forced_backend_preflight_check_count']:,} forced-policy checks. "
            "All six widths' gates are retained under `regression-preflights/`. "
            "The four smaller widths have no new timed fits in this campaign.\n\n"
        )
    qualification_note = ""
    if summary["protocol_version"] == "v6":
        rows = []
        for width in summary["widths"]:
            item = summary["by_width"][str(width)]
            rows.append(
                f"| {width} | {item['native_fp64_accuracy_passed']} | "
                f"{item['python_fp64_accuracy_passed']} | "
                f"{item['cross_implementation_fp32_compatibility_passed']} | `{item['status']}` |"
            )
        qualification_note = (
            f"## Numerical qualification\n\nOverall status: **`{summary['status']}`**. "
            "Evidence integrity, each implementation's accuracy against independent FP64, "
            "and mutual FP32 compatibility are separate results. Full numerical qualification "
            "requires integrity and both FP64 accuracy checks. Timings remain descriptive "
            "where either implementation misses that requirement; no failed comparison is "
            "silently promoted to a pass.\n\n"
            "| Width | Native FP64 accuracy | PyTorch FP64 accuracy | "
            "FP32 compatibility | Status |\n"
            "|---:|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
            "![Numerical accuracy against FP64](numerical-accuracy.png)\n\n"
            "The unchanged forward limit is `2e-5 + 2e-4*abs(reference)`. Both engines "
            "are evaluated at every selected checkpoint from both training paths. Originating "
            "predictions must replay exactly with their original batch semantics. All "
            "numerical failures are retained separately in each audit's `numerical_checks`.\n\n"
        )
        regression_note = (
            "Widths 32–256 retain the strict trajectory contract; widths 512/1024 retain "
            "the explicit common-state/optimizer-replay contract and its original trajectory "
            "diagnostics. The additional prospective FP64 attention stress contract covers "
            "large competing scores, gradients, causal interventions, and independent oracles. "
            "All six widths were measured anew. All "
            f"{summary['regression_all_backend_preflight_check_count']:,} "
            "automatic/forced regression checks and their diagnostics are retained under "
            "`regression-preflights/`. Stored model state remains FP32; wide native attention "
            "uses selective FP64 accumulation in forward and backward. "
            "The separate numerical audit retains "
            f"{summary['numerical_check_count']:,} comparisons, "
            "including every failed FP64 and FP32-compatibility check.\n\n"
        )
    return f"""# Transformer width sweep, {summary["protocol_version"]}

The complete sweep contains **{len(summary["runs"])} fits**
at widths {", ".join(str(width) for width in summary["widths"])}:
three seeds, two timing repeats and both implementations at each width.
Each comparison uses the same initial weights, token rows and schedules within
its width/seed pair. Source revision: `{summary["revision"]}`.

This is the reused engineering benchmark from {prior}, not a fresh untouched test
set. Widths and training rules were fixed before timing. Wider models change
capacity and computational cost; this report does not select a width from test quality.

{qualification_note}{backend_note}## Training and forward timing

| Width | Parameters | PyTorch training (s) | ARM64 training (s) | Paired training ratio |
|---:|---:|---:|---:|---:|
{chr(10).join(train)}

The primary ratio at each width is the geometric mean of six paired
PyTorch/ARM64 training-time ratios. Above one favors assembly; below one favors
PyTorch. There is no pooled-width primary ratio or excluded unfavorable width.
Training includes validation, finite checks, checkpoint selection/writes and
history bookkeeping. Each fit performs {OPTIMIZER_STEPS:,} updates over eight epochs.

![Training and forward cost across widths](scaling.png)

| Width | PyTorch forward p50 (ms) | ARM64 forward p50 (ms) | Ratio of mean p50s |
|---:|---:|---:|---:|
{chr(10).join(forward)}

Forward latency is one full 32-character context, batch one, after 20 warmups,
with 100 samples per fit. It excludes token loading, loss, output-probability softmax, sampling and
I/O. It is not cached next-token generation. The forward ratio uses the two
implementation means, a different statistic from the primary training ratio.

![Per-width implementation time ratios](speedup.png)

Entries and error bars are mean ± sample SD of three seed means, with the two
repeats averaged within each seed. Repeats are not independent quality seeds.
OS scheduling, temperatures and clock frequencies were not experimentally fixed.

## Throughput, memory and prediction quality

| Width | Implementation | Target ktoken/s | Training peak RSS (MiB) | Test NLL | Accuracy (%) |
|---:|---|---:|---:|---:|---:|
{chr(10).join(quality)}

Throughput divides **{TRAINING_TOKENS_PER_FIT:,} training target characters per fit**
by the training-plus-validation timer, then averages repetitions within seed.
Validation tokens are excluded from the numerator while validation time remains
in the denominator. RSS includes the framework, runtime and data; it does not
isolate model memory. NLL and accuracy are descriptive observations, not evidence
of statistical equivalence between implementations.

![Memory and quality across the fixed width family](memory-quality.png)

Uniform and training-only unigram baselines, all per-seed quality differences,
prediction disagreements, checkpoints and individual runs are retained in the
per-width reports. The shared data audit verifies identical corpus identity,
partitions and train/validation/test rows across widths, identical schedules for
each seed, and distinct width-specific initial parameter files. No model-quality
tuning used this test set during the sweep.

## Evidence and limits

{regression_note}The evidence-integrity audit contains **{summary["audit_check_count"]:,} checks**:
{summary["sweep_audit_check_count"]:,} sweep consistency checks plus all per-width
post-run records. The primary automatic-policy numerical/runtime preflights contain
**{summary["preflight_check_count"]:,} checks** in total. Forced-policy gates add
**{summary.get("forced_backend_preflight_check_count", 0):,} checks**. All
**{summary["retained_test_prediction_rows"]:,} retained test prediction rows** were
independently recomputed from selected checkpoints; selected validation losses
were also checked against checkpoint-selection histories. Within-implementation
repeat equality is required before quality aggregation.

{evidence}

See [summary.json](summary.json) for all observations, statistics and hashes,
[audit.json](audit.json) for every post-run check, and
[report-processes.json](report-processes.json) for the separate width-selected
audit subprocesses. Full raw evidence remains in the original result/input
directories. Reporting never overwrites an existing destination.

The first repeat visited ascending widths and the second descending widths;
implementation order reversed within each seed pair. Fits ran sequentially on
AC power with one CPU compute thread and 15-second pauses. Every measured entry
matches its per-width manifest exactly, with unchanged source, binary and inputs.

The scope is this short-context float32 model family on one CPU machine against
eager PyTorch. Native training still accumulates per-sequence gradients. These
results establish neither fresh test-set generalization nor GPU, long-context,
graph-compiled PyTorch, energy, cached decoding or production-serving performance.
Historical {prior} comparisons remain separate campaigns with different conditions.
"""


def create_report(results, inputs, output):
    results, inputs, output = (Path(path).resolve() for path in (results, inputs, output))
    output.mkdir(parents=True, exist_ok=False)
    checks, child_checks, processes = Checks(), [], []
    numerical_checks = []
    summary = {"schema_version": 1, "status": "audit_failed", "protocol_version": "v3"}
    error = None
    try:
        manifest_path = results / "manifest.json"
        manifest_hash = sha256(manifest_path)
        manifest = read_json(manifest_path)
        summary["protocol_version"] = manifest.get("protocol_version")
        widths = protocol_widths(manifest.get("protocol_version"))
        width_manifests = {
            width: read_json(results / f"width-{width}/manifest.json") for width in widths
        }
        manifest_hashes = {
            width: sha256(results / f"width-{width}/manifest.json") for width in widths
        }
        validate_manifest(manifest, width_manifests, checks)
        report_sources = REPORT_SOURCES
        if manifest["protocol_version"] == "v6":
            report_sources += ("scripts/package_sweep_evidence.py",)
        for name in report_sources:
            require(
                checks, "source/" + name, sha256(ROOT / name) == manifest["source_sha256"].get(name)
            )
        summaries, audit_counts = {}, {}
        for width in widths:
            destination = output / f"width-{width}"
            command = [
                sys.executable,
                "-m",
                "transformer.report",
                "--results",
                str(results / f"width-{width}"),
                "--inputs",
                str(inputs / f"width-{width}"),
                "--output",
                str(destination),
            ]
            print(json.dumps({"auditing_width": width}), flush=True)
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=child_environment(width),
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                record = {
                    "width": width,
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except subprocess.TimeoutExpired as exc:
                record = {
                    "width": width,
                    "command": command,
                    "returncode": -1,
                    "error": str(exc),
                    "stdout": str(exc.stdout),
                    "stderr": str(exc.stderr),
                }
            processes.append(record)
            require(checks, f"width-{width}/report_process", record["returncode"] == 0)
            child_audit = read_json(destination / "audit.json")
            child_summary = read_json(destination / "summary.json")
            records = child_audit.get("checks", [])
            require(
                checks,
                f"width-{width}/audit_records",
                (
                    child_audit.get("evidence_integrity_passed") is True
                    if manifest["protocol_version"] == "v6"
                    else child_audit.get("passed") is True
                )
                and bool(records)
                and child_audit.get("check_count") == len(records)
                and child_summary.get("audit_check_count") == len(records)
                and all(item.get("passed") is True for item in records)
                and child_audit.get("failed_checks") == []
                and child_audit.get("error") is None,
            )
            child_checks.extend(
                item | {"name": f"width-{width}/{item['name']}"} for item in records
            )
            if manifest["protocol_version"] == "v6":
                numerical_checks.extend(
                    record | {"name": f"width-{width}/{record['name']}"}
                    for record in validate_v6_numerical_report(child_summary, child_audit)
                )
            audit_counts[str(width)] = len(records)
            require(
                checks,
                f"width-{width}/raw_manifest_hash",
                child_summary["raw_manifest_sha256"] == manifest_hashes[width],
            )
            summaries[width] = child_summary
            write_json(destination / "preflight.json", manifest["preflights"][str(width)])
            for mode, gate in manifest.get("backend_preflights", {}).get(str(width), {}).items():
                write_json(destination / f"preflight-{mode}.json", gate)
        if manifest["protocol_version"] in ("v5", "v6"):
            regression_dir = output / "regression-preflights"
            regression_dir.mkdir()
            for width, modes in manifest["regression_preflights"].items():
                for mode, gate in modes.items():
                    suffix = "" if mode == "auto" else "-" + mode
                    write_json(regression_dir / f"width-{width}{suffix}.json", gate)
        aggregate = aggregate_summaries(manifest, summaries, checks)
        require(
            checks,
            "manifest/unchanged_during_reporting",
            sha256(manifest_path) == manifest_hash
            and all(
                sha256(results / f"width-{width}/manifest.json") == manifest_hashes[width]
                for width in widths
            ),
        )
        for width in widths:
            item = aggregate["by_width"][str(width)]
            item["report_summary_sha256"] = sha256(output / f"width-{width}/summary.json")
            item["report_audit_sha256"] = sha256(output / f"width-{width}/audit.json")
        summary.update(
            aggregate,
            status="completed",
            revision=manifest["revision"],
            source_sha256=manifest["source_sha256"],
            raw_manifest_sha256=manifest_hash,
            widths=list(widths),
            models=manifest["models"],
            platform=manifest["platform"],
            hardware=manifest["hardware"],
            torch=manifest["torch"],
            numpy=manifest["numpy"],
            audit_check_count=len(checks.records) + len(child_checks),
            sweep_audit_check_count=len(checks.records),
            per_width_audit_check_counts=audit_counts,
            preflight_check_count=sum(
                item["preflight_check_count"] for item in aggregate["by_width"].values()
            ),
            forced_backend_preflight_check_count=sum(
                item["forced_backend_preflight_check_count"]
                for item in aggregate["by_width"].values()
            ),
            training_target_tokens_per_fit=TRAINING_TOKENS_PER_FIT,
            throughput_scope=(
                "Training target tokens / training-plus-validation seconds; "
                "repeats averaged within seed"
            ),
            evaluation_scope=(
                "Reused fixed engineering benchmark; no fresh untouched test-set claim"
            ),
        )
        summary["all_backend_preflight_check_count"] = (
            summary["preflight_check_count"] + summary["forced_backend_preflight_check_count"]
        )
        if manifest["protocol_version"] in ("v5", "v6"):
            counts = validate_regression_preflights(
                manifest["regression_preflights"], REGRESSION_WIDTHS, manifest["source_sha256"]
            )
            summary.update(
                validation_contract=WIDE_VALIDATION_CONTRACT,
                legacy_diagnostic_count=sum(
                    gate["legacy_diagnostic_count"]
                    for modes in manifest["regression_preflights"].values()
                    for gate in modes.values()
                ),
                legacy_failed_diagnostic_count=sum(
                    len(gate["legacy_failed_checks"])
                    for modes in manifest["regression_preflights"].values()
                    for gate in modes.values()
                ),
                regression_widths=list(REGRESSION_WIDTHS),
                regression_preflight_check_count=counts["auto"],
                regression_forced_backend_preflight_check_count=counts["forced"],
                regression_all_backend_preflight_check_count=counts["all"],
            )
        if manifest["protocol_version"] == "v6":
            from transformer.precision_audit import CATEGORIES, category_summary

            summary.update(v6_qualification(summaries.values()))
            summary["precision_contract"] = manifest["precision_contract"]
            summary["postrun_numerical_contract"] = manifest["postrun_numerical_contract"]
            summary["validation_contract"] = manifest["validation_contract"]
            summary["validation_contracts"] = manifest["validation_contracts"]
            summary["numerical_check_count"] = len(numerical_checks)
            summary["numerical_categories"] = {
                category: {
                    key: value
                    for key, value in category_summary(
                        [record for record in numerical_checks if record["category"] == category]
                    ).items()
                    if key != "checks"
                }
                for category in CATEGORIES
            }
        charts(summary, output)
        (output / "README.md").write_text(report_markdown(summary))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        checks.require("sweep_report/exception", False, error)
        summary["status"] = "audit_failed"
        (output / "README.md").write_text(
            "# Width-sweep report: audit failed\n\nNo validated aggregate result is claimed. "
            "See [audit.json](audit.json) and [report-processes.json](report-processes.json). "
            "Completed child reports and all raw evidence remain preserved.\n"
        )
    records = checks.records + child_checks
    audit = {
        "schema_version": 1,
        "passed": checks.passed and error is None,
        "check_count": len(records),
        "sweep_check_count": len(checks.records),
        "per_width_post_run_check_count": len(child_checks),
        "checks": records,
        "error": error,
        "failed_checks": [item["name"] for item in records if not item["passed"]],
    }
    if summary["protocol_version"] == "v6":
        integrity = audit["passed"]
        flags = v6_qualification(summaries.values()) if integrity else {}
        audit.update(
            flags,
            evidence_integrity_passed=integrity,
            numerical_checks=numerical_checks,
            numerical_check_count=len(numerical_checks),
        )
        audit["passed"] = integrity and flags.get("numerically_qualified", False)
        audit["schema_version"] = 2
        audit["postrun_numerical_contract"] = summary.get("postrun_numerical_contract")
        audit["numerical_categories"] = summary.get("numerical_categories", {})
        audit["integrity_check_count"] = len(records)
        audit["integrity_failed_checks"] = audit["failed_checks"]
        audit["numerical_failed_checks"] = [
            record["name"]
            for record in numerical_checks
            if record["category"] != "fp32_compatibility" and not record["passed"]
        ]
    write_json(output / "report-processes.json", processes)
    write_json(output / "audit.json", audit)
    write_json(output / "summary.json", summary)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = create_report(args.results, args.inputs, args.output)
    print(
        json.dumps({key: audit[key] for key in ("passed", "check_count", "failed_checks", "error")})
    )
    return 0 if audit.get("evidence_integrity_passed", audit["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
