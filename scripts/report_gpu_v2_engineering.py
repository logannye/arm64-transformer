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


def paired_summary(pair):
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
    return {
        "candidate_over_reference_latency": ratio,
        "latency_reduction_fraction": 1 - ratio,
        "paired_ratios": ratios,
        "minimum_paired_ratio": min(ratios),
        "maximum_paired_ratio": max(ratios),
        "candidate_median_seconds": statistics.median(pair["candidate"]["seconds"]),
        "reference_median_seconds": statistics.median(pair["reference"]["seconds"]),
        "boundary": pair.get("boundary"),
        "dispersion_label": "range of five paired block ratios; not a confidence interval",
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
                and search.get("selected_bundle_full_checks", {}).get("passed") is True
            )
            modes[mode] = {
                "candidate_status_counts": dict(Counter(row["status"] for row in candidates)),
                "candidates": details,
                "selected_policy": saved.get("selected_policy"),
                "selected_policy_sha256": saved.get("selected_policy_sha256"),
                "direct_pair_available": direct is not None and eligible,
                "selected_vs_reference": paired_summary(direct)
                if direct is not None and eligible
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
                "modes": modes,
                "profile_stage_medians": {
                    mode: profile["profiles"][mode]["median_gpu_seconds"] for mode in MODES
                },
            }
        )
    return rows


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
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Metal engineering evidence, separate from framework qualification", weight="bold")
    fig.text(
        0.5,
        0.015,
        "Training partition only · bars show five-block range, not confidence intervals "
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
                    )
                )
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
