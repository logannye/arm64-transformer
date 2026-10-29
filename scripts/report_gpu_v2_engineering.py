"""Report retained Metal engineering diagnostics, never execute model arithmetic.

Only direct final selected-bundle/reference pairs support latency ratios. Greedy
candidate/incumbent ratios are retained as selection evidence, never multiplied.
Framework repeatability, runtime support, and FP64 accuracy remain separate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

WIDTHS = (32, 64, 128, 256, 512, 1024)
ENGINES = ("mps", "mlx")
MODES = ("training", "inference")
SCOPE = "training_partition_development_only_no_formal_admission_or_speed_claim"
MAX_JSON = 64 * 1024 * 1024
PROFILE_SCOPE = (
    "intrusive diagnostic: separate synchronized command buffers per stage; not production timing"
)
STAGE_CATEGORIES = (
    ("dense_forward", "Dense forward", "#4477AA"),
    ("dense_dx", "Dense dX", "#66AACC"),
    ("dense_dw", "Dense dW", "#99CCEE"),
    ("qkv", "QKV forward / dX / dW", "#AA3377"),
    ("reductions", "Norm + bias reductions", "#EE6677"),
    ("attention", "Attention", "#228833"),
    ("embedding", "Embedding", "#66CCAA"),
    ("adam", "Adam", "#CCBB44"),
    ("other", "Other / elementwise / loss", "#BBBBBB"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def safe_file(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe evidence path")
    candidate = root / relative
    require(candidate.resolve().is_relative_to(root), "Evidence path escaped its root")
    require(not any(p.is_symlink() for p in (candidate, *candidate.parents)), "Linked evidence")
    require(candidate.is_file(), f"Missing evidence: {candidate}")
    return candidate


class Receipts:
    def __init__(self):
        self.sources = {}

    def read(self, root, relative, role):
        path = safe_file(root, relative)
        require(path.stat().st_size <= MAX_JSON, f"Oversized JSON: {path}")
        before = digest(path)
        result = json.loads(path.read_text())
        require(isinstance(result, dict), f"Expected JSON object: {path}")
        require(digest(path) == before, f"Receipt changed while reading: {path}")
        self.sources[f"{role}/{Path(relative).as_posix()}"] = before
        return result


def exact_pass(check):
    return (
        check.get("passed") is True
        and check.get("different_coordinates") == 0
        and isinstance(check.get("actual"), dict)
        and set(check["actual"]) == {"dtype", "shape", "sha256"}
        and isinstance(check["actual"]["dtype"], str)
        and isinstance(check["actual"]["shape"], list)
        and check["actual"] == check.get("expected")
        and len(check["actual"].get("sha256", "")) == 64
    )


def baseline_rows(root, receipts):
    summary = receipts.read(root, "summary.json", "baseline")
    require(summary.get("scope") == "development_only", "Not a development baseline receipt")
    require(summary.get("formal_admission_opened") is False, "Baseline launcher has not finalized")
    require(summary.get("engines") == list(ENGINES), "Expected both standard baseline engines")
    require(summary.get("widths") == list(WIDTHS), "Expected complete six-width baseline matrix")
    require(
        summary.get("stages") == ["embedding", "full_model", "training"], "Partial baseline stages"
    )
    workers = summary.get("workers", [])
    require(len(workers) == 24, "Expected 24 fresh baseline workers")
    by_key = {(w["engine"], w["width"], w["process"]): w for w in workers}
    require(len(by_key) == 24, "Duplicate baseline worker")
    rows = []
    for engine in ENGINES:
        for width in WIDTHS:
            row = {
                "engine": engine,
                "width": width,
                "cases": 0,
                "completed_cases": 0,
                "exact_failed_cases": [],
                "accuracy_failed_cases": [],
                "runtime_errors": [],
                "cleanup_errors": [],
                "unexecuted_repetitions": 0,
                "executions": 0,
                "accuracy_checks": 0,
                "exact_checks": 0,
                "identity_passed": True,
                "unsupported_unique_controls": 0,
            }
            for process in range(2):
                launch = by_key[(engine, width, process)]
                worker = receipts.read(root, launch["result"], "baseline")
                cases = worker.get("cases", [])
                row["identity_passed"] &= worker.get("source_unchanged") is True
                row["identity_passed"] &= worker.get("source_sha256") == worker.get(
                    "source_sha256_after"
                )
                require(worker.get("repetitions") == 10, "Unexpected repetition count")
                expected_names = worker.get("expected_cases", [])
                names = [case["name"] for case in cases]
                inventory_ok = len(names) == len(set(names)) and set(names) == set(expected_names)
                if not inventory_ok:
                    row["runtime_errors"].append(
                        {"process": process, "error": "Incomplete case inventory"}
                    )
                row["unsupported_unique_controls"] += len(worker.get("unsupported_controls", []))
                row["cleanup_errors"].extend(worker.get("cleanup_errors", []))
                for case in cases:
                    name = f"process-{process}/{case['name']}"
                    executions = case.get("executions", [])
                    completed = case.get("complete") is True and len(executions) == 10
                    row["cases"] += 1
                    row["completed_cases"] += int(completed)
                    row["unexecuted_repetitions"] += len(case.get("unexecuted_repetitions", []))
                    exact_failed = accuracy_failed = False
                    for execution in executions:
                        row["executions"] += 1
                        if "error" in execution:
                            row["runtime_errors"].append(
                                {"case": name, "error": execution["error"]}
                            )
                            continue
                        exact = execution.get("exact", {})
                        accuracy = execution.get("accuracy", {})
                        row["exact_checks"] += len(exact)
                        row["accuracy_checks"] += len(accuracy)
                        exact_failed |= not exact or not all(
                            exact_pass(value) for value in exact.values()
                        )
                        accuracy_failed |= not accuracy or not all(
                            value.get("passed") is True
                            and value.get("failed_coordinates") == 0
                            and value.get("nonfinite_coordinates") == 0
                            for value in accuracy.values()
                        )
                        if execution.get("finite") is not True:
                            accuracy_failed = True
                    if exact_failed:
                        row["exact_failed_cases"].append(name)
                    if accuracy_failed:
                        row["accuracy_failed_cases"].append(name)
                if launch.get("returncode") not in (0, 1):
                    row["runtime_errors"].append(
                        {"process": process, "error": "Abnormal worker exit"}
                    )
            cross = receipts.read(root, f"{engine}-w{width}-cross-process.json", "baseline")
            cross_cases = cross.get("cases", [])
            cross_passed = (
                bool(cross_cases)
                and cross.get("source_equal") is True
                and cross.get("inventory_equal") is True
            )
            cross_passed &= all(
                case.get("inventory_equal") is True
                and bool(case.get("exact"))
                and all(exact_pass(value) for value in case["exact"].values())
                for case in cross_cases
            )
            row["cross_process_passed"] = cross_passed
            complete = row["completed_cases"] == row["cases"] and row["cases"] > 0
            row["runtime_status"] = (
                "pass"
                if complete and not row["runtime_errors"] and not row["cleanup_errors"]
                else "fail"
            )
            row["exact_status"] = (
                "fail"
                if row["exact_failed_cases"] or not cross_passed
                else "pass"
                if complete
                else "incomplete"
            )
            row["accuracy_status"] = (
                "fail" if row["accuracy_failed_cases"] else "pass" if complete else "incomplete"
            )
            row["standard_baseline_qualified"] = row["identity_passed"] and all(
                row[key] == "pass" for key in ("runtime_status", "exact_status", "accuracy_status")
            )
            rows.append(row)
    return rows


def percentile(values, quantile):
    """Linear interpolation at (n - 1) * q, explicitly defined for saved samples."""
    ordered = sorted(values)
    require(bool(ordered) and 0 <= quantile <= 1, "Invalid percentile request")
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def validate_fresh_pair(receipt):
    fields = (
        "owned_buffer_count",
        "workspace_buffer_count",
        "input_buffer_count",
        "batch_buffer_count",
        "owned_buffer_bytes_including_guards",
    )
    require(
        receipt.get("fresh_instances_without_qualification_allocations") is True
        and receipt.get("equal_initial_state") is True,
        "Direct pair lacks fresh instances or equal initial-state proof",
    )
    memories = receipt.get("memory_before_timing", {})
    for label in ("reference", "candidate"):
        memory = memories.get(label, {})
        require(
            all(type(memory.get(key)) is int and memory[key] >= 0 for key in fields),
            "Direct pair lacks valid owned-buffer measurements",
        )
        require(
            memory["workspace_buffer_count"] >= 1
            and memory["input_buffer_count"] == 0
            and memory["batch_buffer_count"] == 3,
            "Direct pair does not contain only the B16T32/B1T32 batch buffers",
        )
        require(
            memory["owned_buffer_count"]
            == sum(
                memory[key]
                for key in ("workspace_buffer_count", "input_buffer_count", "batch_buffer_count")
            ),
            "Direct pair owned-buffer counts are inconsistent",
        )
    require(
        all(memories["reference"][key] == memories["candidate"][key] for key in fields),
        "Direct pair owned counts or bytes are asymmetric",
    )
    saved = receipt.get("allocation_validation", {})
    require(
        saved.get("equal_owned_counts_and_bytes") is True
        and saved.get("compared_fields") == list(fields),
        "Fresh-pair allocation validation receipt is incomplete",
    )
    return {
        "compared_fields": list(fields),
        "equal_owned_counts_and_bytes": True,
        "batch_buffer_count_per_backend": 3,
        "owned_buffer_bytes_including_guards_per_backend": memories["reference"][fields[-1]],
        "device_global_allocation_or_process_rss_compared": False,
    }


def paired_summary(pair, *, mode=None):
    require(isinstance(pair, dict), "Missing direct paired timing")
    medians = {}
    for label in ("reference", "candidate"):
        row = pair[label]
        require(
            (row["warmups"], row["observations"], row["blocks"]) == (20, 100, 5),
            "Timing contract changed",
        )
        require(
            all(
                row.get(key) is True
                for key in ("exact_block_replay", "exact_state_restoration", "guards")
            ),
            "Timing control failed",
        )
        samples = row["seconds"]
        require(
            len(samples) == 100 and all(math.isfinite(x) and x > 0 for x in samples),
            "Invalid timing samples",
        )
        medians[label] = [statistics.median(samples[i : i + 20]) for i in range(0, 100, 20)]
        require(
            all(
                math.isclose(a, b, rel_tol=1e-12)
                for a, b in zip(medians[label], row["block_median_seconds"], strict=True)
            ),
            "Stored block medians disagree",
        )
    ratios = [c / r for c, r in zip(medians["candidate"], medians["reference"], strict=True)]
    require(
        all(
            math.isclose(a, b, rel_tol=1e-12)
            for a, b in zip(ratios, pair["paired_latency_ratios"], strict=True)
        ),
        "Stored paired ratios disagree",
    )
    ratio = statistics.median(ratios)
    require(
        math.isclose(ratio, pair["median_paired_latency_ratio"], rel_tol=1e-12),
        "Stored paired summary disagrees",
    )
    result = {
        "candidate_over_reference_latency": ratio,
        "latency_reduction_fraction": 1 - ratio,
        "paired_ratios": ratios,
        "minimum_paired_ratio": min(ratios),
        "maximum_paired_ratio": max(ratios),
        "candidate_median_seconds": statistics.median(pair["candidate"]["seconds"]),
        "reference_median_seconds": statistics.median(pair["reference"]["seconds"]),
        "boundary": pair.get("boundary"),
        "dispersion_label": "range of five paired block ratios; not a confidence interval",
        "percentile_method": "linear interpolation at (100 - 1) * quantile",
    }
    for label in ("reference", "candidate"):
        samples = pair[label]["seconds"]
        for name, quantile in (("p50", 0.5), ("p95", 0.95)):
            seconds = percentile(samples, quantile)
            result[f"{label}_{name}_us"] = seconds * 1e6
            result[f"{label}_{name}_ms"] = seconds * 1e3
        result[f"{label}_training_targets_per_second"] = (
            16 * 32 * len(samples) / sum(samples) if mode == "training" else None
        )
    result["training_throughput_definition"] = (
        (
            "51,200 next-token targets / summed durations of 100 B16T32 updates; "
            "same production timing boundary, excludes warmups"
        )
        if mode == "training"
        else None
    )
    return result


def stage_category(name):
    if name == "qkv_forward" or name.startswith("qkv_dx/") or name == "dense_dw/qkv":
        return "qkv"
    for prefix in ("dense_forward/", "dense_dx/", "dense_dw/"):
        if name.startswith(prefix):
            return prefix[:-1]
    if name.startswith(("norm_", "bias_gradient/")):
        return "reductions"
    for prefix in ("attention", "embedding", "adam"):
        if name.startswith(prefix):
            return prefix
    return "other"


def profile_summary(profile):
    require(profile.get("scope") == PROFILE_SCOPE, "Unexpected GPU profiling boundary")
    require(
        profile.get("repetitions") == 3 and profile.get("restored_warmups") == 20,
        "Unexpected GPU profile repetition/warmup contract",
    )
    records = profile.get("records", [])
    grouped = {}
    for record in records:
        name, seconds = record.get("stage"), record.get("gpu_seconds")
        require(isinstance(name, str) and bool(name), "Unnamed GPU profile stage")
        require(
            isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0,
            "Invalid GPU stage duration",
        )
        host = record.get("host_completion_seconds")
        require(
            isinstance(host, (int, float)) and math.isfinite(host) and host > 0,
            "Invalid host completion duration",
        )
        grouped.setdefault(name, []).append(record)
    require(
        bool(grouped) and all(len(values) == 3 for values in grouped.values()),
        "Incomplete repeated GPU stage inventory",
    )
    stages = list(grouped)
    require(
        [record["stage"] for record in records] == stages * 3,
        "GPU stage order changed between repetitions",
    )
    saved = profile.get("median_gpu_seconds", {})
    require(set(saved) == set(grouped), "Stored stage median inventory differs")
    medians = {
        name: statistics.median(value["gpu_seconds"] for value in values)
        for name, values in grouped.items()
    }
    require(
        all(math.isclose(saved[name], value, rel_tol=1e-12) for name, value in medians.items()),
        "Stored GPU stage median differs",
    )
    totals = {name: 0.0 for name, _, _ in STAGE_CATEGORIES}
    for name, value in medians.items():
        totals[stage_category(name)] += value
    dispatch_count = None
    if all("dispatch_count" in record for record in records):
        require(
            all(
                type(record["dispatch_count"]) is int and record["dispatch_count"] >= 0
                for record in records
            ),
            "Invalid retained dispatch count",
        )
        require(
            all(
                len({record["dispatch_count"] for record in values}) == 1
                for values in grouped.values()
            ),
            "Dispatch counts differ across repetitions",
        )
        dispatch_count = sum(values[0]["dispatch_count"] for values in grouped.values())
    return {
        "scope": PROFILE_SCOPE,
        "records": records,
        "stage_median_gpu_seconds": medians,
        "category_sum_of_stage_medians_seconds": totals,
        "total_sum_of_stage_medians_seconds": sum(medians.values()),
        "profiled_stage_command_buffers_per_repetition": len(stages),
        "kernel_dispatches_per_repetition": dispatch_count,
        "dispatch_count_scope": (
            "Per-stage dispatch counts retained in each raw record"
            if dispatch_count is not None
            else "Unavailable: cumulative backend counters do not isolate matched profile stages"
        ),
        "production_runtime_attribution_claimed": False,
    }


def compare_profiles(reference, search, mode):
    require(
        search.get("profile_identity_verified") is True,
        "Search did not authenticate its matching reference profile",
    )
    for key in (
        "source_sha256",
        "runtime_identity",
        "input_sha256",
        "input_directory",
        "rows_used",
        "environment",
        "warmups",
        "observations",
        "blocks",
        "profile_repeats",
    ):
        require(
            key in reference and reference[key] == search.get(key),
            f"Reference/selected profile identity differs: {key}",
        )
    require(reference.get("status") == "completed", "Reference profile was not completed")
    original = profile_summary(reference["profiles"][mode])
    selected = profile_summary(search["selected_profiles_after_production"][mode])
    original_dispatches = original["kernel_dispatches_per_repetition"]
    selected_dispatches = selected["kernel_dispatches_per_repetition"]
    return {
        "reference": original,
        "selected": selected,
        "selected_minus_reference_profiled_stage_command_buffers": (
            selected["profiled_stage_command_buffers_per_repetition"]
            - original["profiled_stage_command_buffers_per_repetition"]
        ),
        "selected_minus_reference_kernel_dispatches": (
            selected_dispatches - original_dispatches
            if original_dispatches is not None and selected_dispatches is not None
            else None
        ),
        "interpretation": (
            "Intrusive synchronized stage profiles of the reference and final selected bundle; "
            "sums of stage medians do not attribute production wall time or isolate causal effects"
        ),
    }


def metal_rows(profiles, searches, receipts, *, attempt_id="selection-v2"):
    require(
        not (Path(searches) / "invalidated.json").exists(),
        "This search attempt was invalidated; its timings are ineligible",
    )
    attempt = receipts.read(searches, "attempt.json", "searches")
    require(
        attempt.get("schema") == 1
        and attempt.get("scope") == "metal_development_search_attempt"
        and attempt.get("attempt_id") == attempt_id
        and attempt.get("eligible_for_reporting") is True,
        "Explicit eligible search attempt marker is required",
    )
    rows = []
    for width in WIDTHS:
        name = f"width-{width}.json"
        profile = receipts.read(profiles, name, "profiles")
        search = receipts.read(searches, name, "searches")
        require(profile.get("scope") == search.get("scope") == SCOPE, "Unexpected search scope")
        require(profile.get("width") == search.get("width") == width, "Wrong width receipt")
        for root, role, value in ((profiles, "profiles", profile), (searches, "searches", search)):
            receipts.read(root, name + ".ledger.json", role)
            require(
                value["ledger_sha256"] == receipts.sources[f"{role}/{name}.ledger.json"],
                "Ledger hash changed",
            )
        require(
            search["reference_profile_sha256"] == receipts.sources[f"profiles/{name}"],
            "Profile hash changed",
        )
        fresh_pair = None
        if search.get("status") == "completed" and search.get("selected_vs_reference"):
            fresh_pair = validate_fresh_pair(search.get("final_fresh_pair", {}))
        modes = {}
        for mode in MODES:
            saved = search.get("modes", {}).get(mode, {})
            candidates = saved.get("candidates", [])
            details = []
            for candidate in candidates:
                if "evidence" not in candidate:
                    details.append(candidate)
                    continue
                relative = f"{name}.evidence/{candidate['evidence']}"
                detail = receipts.read(searches, relative, "searches")
                require(
                    candidate["evidence_sha256"] == receipts.sources[f"searches/{relative}"],
                    "Candidate receipt changed",
                )
                require(
                    detail["id"] == candidate["id"] and detail["status"] == candidate["status"],
                    "Candidate status changed",
                )
                details.append(
                    {
                        key: detail[key]
                        for key in ("id", "status", "family", "error")
                        if key in detail
                    }
                )
            direct = search.get("selected_vs_reference", {}).get(mode)
            eligible = (
                search.get("status") == "completed"
                and search.get("sources_unchanged") is True
                and search.get("inputs_unchanged") is True
                and search.get("runtime_unchanged") is True
                and search.get("selected_bundle_full_checks", {}).get("passed") is True
                and fresh_pair is not None
            )
            modes[mode] = {
                "candidate_status_counts": dict(Counter(row["status"] for row in candidates)),
                "candidates": details,
                "selected_policy": saved.get("selected_policy"),
                "selected_policy_sha256": saved.get("selected_policy_sha256"),
                "direct_pair_available": direct is not None and eligible,
                "selected_vs_reference": paired_summary(direct, mode=mode)
                if direct is not None and eligible
                else None,
                "kernel_profile_comparison": compare_profiles(profile, search, mode)
                if eligible
                else None,
                "unavailable_reason": None
                if direct is not None and eligible
                else (
                    "No qualified direct final selected-bundle/reference pair; "
                    "greedy ratios are not combined"
                ),
            }
        rows.append(
            {
                "width": width,
                "search_status": search.get("status"),
                "error": search.get("error"),
                "tested_revision": search.get("tested_revision"),
                "input_sha256": search["input_sha256"],
                "fresh_pair_allocation_proof": fresh_pair,
                "modes": modes,
                "profile_stage_medians": {
                    mode: profile["profiles"][mode]["median_gpu_seconds"] for mode in MODES
                },
            }
        )
    return rows


def write_profile_tables(summary, output):
    with (
        (output / "kernel-stage-costs.csv").open("w", newline="") as raw_handle,
        (output / "kernel-category-costs.csv").open("w", newline="") as category_handle,
    ):
        raw, categories = csv.writer(raw_handle), csv.writer(category_handle)
        raw.writerow(
            (
                "width",
                "mode",
                "variant",
                "repetition",
                "stage",
                "category",
                "gpu_us",
                "host_completion_us",
                "stage_median_gpu_us",
                "kernel_dispatch_count_if_retained",
                "scope",
            )
        )
        categories.writerow(
            (
                "width",
                "mode",
                "variant",
                "category",
                "sum_of_stage_medians_gpu_us",
                "profiled_stage_command_buffers",
                "kernel_dispatches_if_retained",
                "scope",
            )
        )
        for row in summary["metal"]:
            for mode, data in row["modes"].items():
                comparison = data["kernel_profile_comparison"]
                if comparison is None:
                    continue
                for variant in ("reference", "selected"):
                    profile = comparison[variant]
                    seen = Counter()
                    for record in profile["records"]:
                        name = record["stage"]
                        repetition = seen[name]
                        seen[name] += 1
                        raw.writerow(
                            (
                                row["width"],
                                mode,
                                variant,
                                repetition,
                                name,
                                stage_category(name),
                                record["gpu_seconds"] * 1e6,
                                record["host_completion_seconds"] * 1e6,
                                profile["stage_median_gpu_seconds"][name] * 1e6,
                                record.get("dispatch_count", ""),
                                PROFILE_SCOPE,
                            )
                        )
                    for category, seconds in profile[
                        "category_sum_of_stage_medians_seconds"
                    ].items():
                        names = [
                            name
                            for name in profile["stage_median_gpu_seconds"]
                            if stage_category(name) == category
                        ]
                        dispatches = ""
                        if profile["kernel_dispatches_per_repetition"] is not None:
                            dispatches = sum(
                                next(
                                    record["dispatch_count"]
                                    for record in profile["records"]
                                    if record["stage"] == name
                                )
                                for name in names
                            )
                        categories.writerow(
                            (
                                row["width"],
                                mode,
                                variant,
                                category,
                                seconds * 1e6,
                                len(names),
                                dispatches,
                                PROFILE_SCOPE,
                            )
                        )


def render_profile_comparison(summary, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 6, figsize=(18, 8), squeeze=False)
    for mode_index, mode in enumerate(MODES):
        for index, row in enumerate(summary["metal"]):
            axis = axes[mode_index, index]
            comparison = row["modes"][mode]["kernel_profile_comparison"]
            axis.set_xticks((0, 1), ("Reference", "Selected"), fontsize=9)
            axis.set_title(f"Width {row['width']}", fontsize=11)
            if index == 0:
                label = "B16T32 training" if mode == "training" else "B1T32 inference"
                axis.set_ylabel(label + "\nSummed stage GPU medians (µs)", fontsize=10)
            if comparison is None:
                axis.text(0.5, 0.5, "Unavailable", transform=axis.transAxes, ha="center")
                continue
            bottoms = [0.0, 0.0]
            for category, _, color in STAGE_CATEGORIES:
                values = [
                    comparison[variant]["category_sum_of_stage_medians_seconds"][category] * 1e6
                    for variant in ("reference", "selected")
                ]
                axis.bar((0, 1), values, bottom=bottoms, width=0.6, color=color)
                bottoms = [a + b for a, b in zip(bottoms, values, strict=True)]
            for position, variant in enumerate(("reference", "selected")):
                count = comparison[variant]["profiled_stage_command_buffers_per_repetition"]
                axis.text(
                    position,
                    bottoms[position],
                    f"{bottoms[position]:.0f} µs\n{count} stages",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            axis.set_ylim(0, max(bottoms) * 1.25)
            axis.grid(axis="y", alpha=0.15)
            axis.set_axisbelow(True)
    fig.suptitle(
        "How the final Metal policy changes measured kernel-stage costs",
        fontsize=16,
        weight="bold",
        y=0.99,
    )
    fig.legend(
        handles=[Patch(color=color, label=label) for _, label, color in STAGE_CATEGORIES],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.025,
        "Intrusive synchronized command buffers · 3 repetitions per stage · "
        "each panel has its own µs scale · not production wall-time attribution",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.20, 1, 0.95))
    fig.savefig(output / "metal-kernel-ablation.png", dpi=180)
    fig.savefig(output / "metal-kernel-ablation.svg")
    plt.close(fig)


def render(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for mode, color in (("training", "#177E89"), ("inference", "#BD5D30")):
        points = [
            (i, row["modes"][mode]["selected_vs_reference"])
            for i, row in enumerate(summary["metal"])
            if row["modes"][mode]["direct_pair_available"]
        ]
        if points:
            x, values = zip(*points, strict=True)
            y = [v["candidate_over_reference_latency"] for v in values]
            errors = [
                [value - v["minimum_paired_ratio"] for value, v in zip(y, values, strict=True)],
                [v["maximum_paired_ratio"] - value for value, v in zip(y, values, strict=True)],
            ]
            axes[0].errorbar(
                x, y, yerr=errors, marker="o", capsize=3, color=color, label=mode.title()
            )
    axes[0].axhline(1, color="#667085", linestyle="--", linewidth=1)
    axes[0].set(
        ylabel="Selected Metal / reference Metal latency",
        title="Direct final paired comparison",
        xticks=range(6),
        xticklabels=WIDTHS,
        xlabel="Model width",
    )
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No qualified direct final pairs retained",
            transform=axes[0].transAxes,
            ha="center",
        )
    bottom = [0] * 6
    groups = [
        ("Qualified and measured", "#177E89", lambda key: key == "qualified_measured"),
        ("Numerical / execution rejection", "#B54745", lambda key: "rejected" in key),
        (
            "Structural / resource exclusion",
            "#98A2B3",
            lambda key: key != "qualified_measured" and "rejected" not in key,
        ),
    ]
    for label, color, predicate in groups:
        counts = [
            sum(
                count
                for mode in MODES
                for status, count in row["modes"][mode]["candidate_status_counts"].items()
                if predicate(status)
            )
            for row in summary["metal"]
        ]
        axes[1].bar(range(6), counts, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, counts, strict=True)]
    axes[1].set(
        ylabel="Candidate records across both workloads",
        title="Every scored or excluded candidate",
        xticks=range(6),
        xticklabels=WIDTHS,
        xlabel="Model width",
    )
    axes[1].set_ylim(0, max(bottom, default=1) * 1.3)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Metal engineering evidence, separate from framework qualification", weight="bold")
    fig.text(
        0.5,
        0.015,
        "Training partition only · whiskers show five-block range, not confidence intervals "
        "· no framework or CPU speedup claim",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output / "metal-engineering.png", dpi=180)
    plt.close(fig)
    statuses = {"fail": 0, "incomplete": 1, "pass": 2}
    matrix = [
        [statuses[row[key]] for key in ("exact_status", "runtime_status", "accuracy_status")]
        for row in summary["baselines"]
    ]
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(
        matrix,
        cmap=ListedColormap(["#F5C5C0", "#F5DFAB", "#BCE1D5"]),
        vmin=0,
        vmax=2,
        aspect="auto",
    )
    axis.set(
        xticks=range(3),
        xticklabels=("Exact replay", "Runtime completion", "FP64 accuracy"),
        yticks=range(len(matrix)),
        yticklabels=[f"{row['engine'].upper()} · W{row['width']}" for row in summary["baselines"]],
    )
    for i, row in enumerate(summary["baselines"]):
        for j, key in enumerate(("exact_status", "runtime_status", "accuracy_status")):
            axis.text(j, i, row[key], ha="center", va="center", fontsize=10)
    axis.set_title("Standard baseline admission remains separate", weight="bold", pad=18)
    fig.text(
        0.5,
        0.02,
        "Incomplete accuracy means required executions were unavailable; "
        "tolerance does not waive exact replay.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output / "baseline-qualification.png", dpi=180)
    plt.close(fig)
    render_profile_comparison(summary, output)


def report(baseline, profiles, searches, output, *, figures=True, attempt_id="selection-v2"):
    receipts = Receipts()
    summary = {
        "schema": 1,
        "scope": "gpu_v2_blocked_engineering_release",
        "formal_admission_opened": False,
        "formal_fits_completed": 0,
        "framework_or_cpu_speedup_claimed": False,
        "interpretation": (
            "Metal-only development latency comparisons; "
            "failed standard baseline gates remain binding"
        ),
        "baselines": baseline_rows(baseline, receipts),
        "metal": metal_rows(profiles, searches, receipts, attempt_id=attempt_id),
        "eligible_search_attempt": attempt_id,
    }
    require(
        any(not row["standard_baseline_qualified"] for row in summary["baselines"]),
        "This reporter is only for a blocked release",
    )
    summary["source_artifacts"] = receipts.sources
    summary["input_roots"] = {
        "baseline": str(Path(baseline).resolve()),
        "profiles": str(Path(profiles).resolve()),
        "searches": str(Path(searches).resolve()),
    }
    summary["reporter_sha256"] = digest(__file__)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "metal-paired-latency.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "width",
                "mode",
                "direct_pair_available",
                "candidate_over_reference",
                "minimum_block_ratio",
                "maximum_block_ratio",
                *(
                    f"{variant}_{percentile}_{unit}"
                    for variant in ("reference", "candidate")
                    for percentile in ("p50", "p95")
                    for unit in ("us", "ms")
                ),
                "reference_B16T32_targets_per_second",
                "candidate_B16T32_targets_per_second",
            )
        )
        for row in summary["metal"]:
            for mode, data in row["modes"].items():
                pair = data["selected_vs_reference"] or {}
                writer.writerow(
                    (
                        row["width"],
                        mode,
                        data["direct_pair_available"],
                        *(
                            pair.get(key, "")
                            for key in (
                                "candidate_over_reference_latency",
                                "minimum_paired_ratio",
                                "maximum_paired_ratio",
                            )
                        ),
                        *(
                            pair.get(f"{variant}_{percentile}_{unit}", "")
                            for variant in ("reference", "candidate")
                            for percentile in ("p50", "p95")
                            for unit in ("us", "ms")
                        ),
                        pair.get("reference_training_targets_per_second", ""),
                        pair.get("candidate_training_targets_per_second", ""),
                    )
                )
    write_profile_tables(summary, output)
    if figures:
        render(summary, output)
    for key, expected in receipts.sources.items():
        role, relative = key.split("/", 1)
        root = {"baseline": baseline, "profiles": profiles, "searches": searches}[role]
        require(digest(safe_file(root, relative)) == expected, "Evidence changed during reporting")
    identities = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in identities.items())
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "profiles", "searches", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--attempt-id", default="selection-v2")
    args = parser.parse_args()
    result = report(
        args.baseline,
        args.profiles,
        args.searches,
        args.output,
        figures=not args.no_figures,
        attempt_id=args.attempt_id,
    )
    print(
        json.dumps(
            {"output": str(args.output), "scope": result["scope"], "formal_admission_opened": False}
        )
    )


if __name__ == "__main__":
    main()
