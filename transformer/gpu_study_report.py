"""Report all 180 planned cells, including a zero-fit strict-gate outcome."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import statistics
from pathlib import Path

from transformer.gpu_study import (
    ENGINES,
    PROTOCOL,
    SEEDS,
    WIDTHS,
    digest,
    read_json,
    require,
    study_order,
    validate_manifest,
    verify_files,
    write_new,
)


def blocked_report(manifest, reasons):
    validate_manifest(manifest)
    require(
        isinstance(reasons, list)
        and bool(reasons)
        and all(isinstance(r, str) and r for r in reasons),
        "Blocking reasons required",
    )
    return {
        "schema": 1,
        "protocol": PROTOCOL,
        "status": "blocked_before_measurement",
        "planned_fits": 180,
        "completed_fits": 0,
        "blocked_fits": 180,
        "planned_test_token_predictions": 1474560,
        "retained_test_token_predictions": 0,
        "speedup_claims_qualified": False,
        "reasons": reasons,
        "cells": [{**cell, "status": "blocked_before_measurement"} for cell in study_order()],
        "by_width": {},
        "formal_admission_opened": False,
    }


def unfinished_report(manifest, manifest_sha256, campaign_directory):
    """Retain the full denominator after execution stops; emit no timing ratios."""
    from transformer.gpu_study_audit import inspect_cell

    validate_manifest(manifest)
    root = Path(campaign_directory)
    ledger = read_json(root / "campaign.json")
    require(ledger.get("manifest_sha256") == manifest_sha256, "Foreign partial campaign")
    require(
        ledger.get("status") in ("measurement_failed", "running"),
        "Complete campaigns require their full audit",
    )
    cells = ledger.get("cells", [])
    require(
        len(cells) == 180
        and [{k: c.get(k) for k in p} for c, p in zip(cells, study_order(), strict=True)]
        == study_order(),
        "Partial campaign changed the planned denominator",
    )
    statuses = {"pending", "completed", "failed", "blocked_after_failure"}
    require(all(c.get("status") in statuses for c in cells), "Unknown partial cell status")
    completed = [c for c in cells if c["status"] == "completed"]
    for cell in completed:
        planned = {k: cell[k] for k in ("id", "width", "seed", "repeat", "engine")}
        info = inspect_cell(planned, root / cell["id"], manifest_sha256)
        require(
            info["receipt_sha256"] == cell.get("receipt_sha256"),
            "Partial completed-cell evidence changed",
        )
    failed = sum(c["status"] == "failed" for c in cells)
    unrun = sum(c["status"] in ("pending", "blocked_after_failure") for c in cells)
    return {
        "schema": 1,
        "protocol": PROTOCOL,
        "manifest_sha256": manifest_sha256,
        "status": "measurement_stopped_unqualified",
        "planned_fits": 180,
        "completed_fits": len(completed),
        "failed_fits": failed,
        "blocked_fits": unrun,
        "planned_test_token_predictions": 1474560,
        "retained_test_token_predictions": len(completed) * 8192,
        "retained_prediction_scope": (
            "Authenticated completed cells; failed-cell artifacts remain preserved separately"
        ),
        "speedup_claims_qualified": False,
        "cells": cells,
        "by_width": {},
        "campaign_sha256": digest(root / "campaign.json"),
    }


def _validate_numeric(record):
    require(
        isinstance(record, dict) and isinstance(record.get("passed"), bool),
        "Missing numerical record",
    )
    require(
        type(record.get("coordinates")) is int and record["coordinates"] > 0,
        "Invalid coordinate count",
    )
    failed = record.get("failed_coordinates")
    require(type(failed) is int and 0 <= failed <= record["coordinates"], "Invalid failure count")
    require(record["passed"] == (failed == 0), "Numerical status conceals failed coordinates")
    for key in ("max_absolute_error", "max_tolerance_ratio", "atol", "rtol"):
        require(
            isinstance(record.get(key), (int, float))
            and math.isfinite(record[key])
            and record[key] >= 0,
            "Invalid numerical metric",
        )
    require(
        record["passed"] == (record["max_tolerance_ratio"] <= 1), "Numerical ratio/status mismatch"
    )


def verify_prediction_records(report, arrays, saved_logits, cell, input_hashes):
    import numpy as np

    from transformer.gpu_study_audit import prediction_delta, quality

    for split in ("validation", "test"):
        rows = arrays["oracle"].get(split + "_rows")
        require(
            rows is not None and rows.dtype == np.int32 and rows.shape == (256, 33),
            "Missing original replay input rows",
        )
        name = f"width-{cell['width']}/seed-{cell['seed']}/{split}.i32"
        require(
            hashlib.sha256(rows.astype("<i4", copy=False).tobytes()).hexdigest()
            == input_hashes.get(name),
            "Replay input rows differ from frozen corpus",
        )
    expected = {f"{e}/{s}" for e in ENGINES for s in ("validation", "test")}
    require(
        set(report.get("prediction_differences", {})) == expected, "Missing prediction differences"
    )
    for name in expected:
        engine, split = name.split("/")
        computed = prediction_delta(
            arrays[engine][split], arrays["oracle"][split], arrays["oracle"][split + "_rows"][:, 1:]
        )
        require(
            report["prediction_differences"][name] == computed, "Prediction differences changed"
        )
    require(
        report.get("quality") == quality(saved_logits, arrays["oracle"]["test_rows"][:, 1:]),
        "Saved-prediction quality changed",
    )


def load_authenticated_reports(audit_directory, campaign_directory, manifest_sha256, manifest):
    """Recompute numerical status from retained replay arrays, not report flags."""
    import numpy as np

    from transformer.gpu_study_audit import compare, inspect_cell, repeat_integrity

    validate_manifest(manifest)
    root, campaign = Path(audit_directory), Path(campaign_directory)
    aggregate = read_json(root / "audit.json")
    ledger = read_json(campaign / "campaign.json")
    require(
        aggregate.get("manifest_sha256") == ledger.get("manifest_sha256") == manifest_sha256,
        "Foreign audit/campaign",
    )
    require(
        aggregate.get("campaign_sha256") == digest(campaign / "campaign.json"),
        "Campaign changed after audit",
    )
    require(ledger.get("status") == "measurement_complete_unaudited", "Campaign did not complete")
    require(
        len(ledger.get("cells", [])) == 180
        and all(c.get("status") == "completed" for c in ledger["cells"]),
        "Missing completed cells",
    )
    require(
        aggregate.get("planned_checkpoints") == aggregate.get("audited_checkpoints") == 180,
        "Incomplete checkpoint audit",
    )
    reports = aggregate.get("reports", [])
    require(
        len(reports) == 180 and [r.get("cell") for r in reports] == study_order(),
        "Audit omitted, duplicated or reordered cells",
    )
    repeated = repeat_integrity(campaign)
    require(aggregate.get("repeat_checks") == repeated, "Repeat evidence changed")
    result = []
    for entry, planned, recorded in zip(reports, study_order(), ledger["cells"], strict=True):
        require(
            all(recorded.get(k) == v for k, v in planned.items()),
            "Campaign order/identity mismatch",
        )
        require(entry.get("path") == f"{planned['id']}/audit.json", "Unexpected audit path")
        require(digest(root / entry["path"]) == entry.get("sha256"), "Audit file changed")
        report = read_json(root / entry["path"])
        require(
            report.get("cell") == planned and report.get("manifest_sha256") == manifest_sha256,
            "Audit identity changed",
        )
        info = inspect_cell(planned, campaign / planned["id"], manifest_sha256)
        require(
            info["receipt_sha256"]
            == report.get("cell_receipt_sha256")
            == recorded.get("receipt_sha256"),
            "Cell receipt changed",
        )
        replay_root = root / planned["id"]
        require(
            set(report.get("replay_sha256", {})) == {f"{e}.npz" for e in (*ENGINES, "oracle")},
            "Missing replay engine",
        )
        verify_files(replay_root, report["replay_sha256"])
        arrays = {}
        for engine in (*ENGINES, "oracle"):
            with np.load(replay_root / f"{engine}.npz", allow_pickle=False) as data:
                arrays[engine] = {key: data[key] for key in data.files}
        expected_fp64 = {f"{e}/{s}" for e in ENGINES for s in ("validation", "test", "b1")}
        expected_compatibility = {
            f"{a}/{b}/{s}"
            for a, b in itertools.combinations(ENGINES, 2)
            for s in ("validation", "test")
        }
        require(
            set(report.get("fp64", {})) == expected_fp64
            and set(report.get("compatibility", {})) == expected_compatibility,
            "Missing numerical comparisons",
        )
        require(
            set(report.get("oracle", {})) == {"validation", "test"}, "Missing oracle cross-check"
        )
        for split in ("validation", "test"):
            require(
                report["oracle"][split]
                == compare(
                    arrays["oracle"][split][[0, -1]],
                    arrays["oracle"][split + "_torch_fp64"],
                    1e-10,
                    1e-10,
                ),
                "Oracle record changed",
            )
        verify_prediction_records(
            report, arrays, info["logits"], planned, manifest["bindings"]["inputs"]
        )
        for name, value in report["fp64"].items():
            engine, split = name.split("/")
            gold = arrays["oracle"]["test"][:1] if split == "b1" else arrays["oracle"][split]
            _validate_numeric(value)
            require(value == compare(arrays[engine][split], gold), "Accuracy record changed")
        for name, value in report["compatibility"].items():
            left, right, split = name.split("/")
            _validate_numeric(value)
            require(
                value == compare(arrays[left][split], arrays[right][split]),
                "Compatibility record changed",
            )
        own = arrays[planned["engine"]]
        checks = {
            "own_test_replay_exact": own["test"].tobytes() == info["logits"].tobytes(),
            "own_validation_history_exact": float(own["validation_nll"])
            == float(info["history"][info["best_epoch"] - 1, 1]),
        }
        b1 = campaign / planned["id"] / "inference_logits.f32"
        if b1.exists():
            checks["own_b1_replay_exact"] = own["b1"].tobytes() == b1.read_bytes()
        require(report.get("integrity") == checks, "Replay integrity record changed")
        for key in ("training_seconds", "latency_p50", "latency_p95", "memory"):
            require(report.get(key) == info[key], "Reported runtime metric changed")
        result.append(report)
    return result, repeated


def summarize(manifest, manifest_sha256, reports, repeats):
    validate_manifest(manifest)
    require(
        len(reports) == 180 and [r["cell"] for r in reports] == study_order(),
        "Complete ordered reports required",
    )
    require(len(repeats) == 90, "Complete repeat comparison required")
    require(
        all(report.get("manifest_sha256") == manifest_sha256 for report in reports),
        "Report manifest mismatch",
    )
    by_width = {}
    for width in WIDTHS:
        rows = [r for r in reports if r["cell"]["width"] == width]
        pairs = [r for r in repeats if r["first"].startswith(f"width-{width}/")]
        require(len(rows) == 30 and len(pairs) == 15, "Missing width cells")
        integrity = all(
            all(r["integrity"].values()) and all(v["passed"] for v in r["oracle"].values())
            for r in rows
        )
        repeat_ok = all(p["passed"] for p in pairs)
        accuracy = {
            e: all(
                r["fp64"][f"{e}/{s}"]["passed"] for r in rows for s in ("validation", "test", "b1")
            )
            for e in ENGINES
        }
        compatible = all(v["passed"] for r in rows for v in r["compatibility"].values())
        engines = {}
        indexed = {(r["cell"]["seed"], r["cell"]["repeat"], r["cell"]["engine"]): r for r in rows}
        for engine in ENGINES:
            selected = [r for r in rows if r["cell"]["engine"] == engine]
            seed_means = [
                statistics.mean(indexed[seed, rep, engine]["quality"]["nll"] for rep in (1, 2))
                for seed in SEEDS
            ]
            engines[engine] = {
                "training_seconds_mean": statistics.mean(r["training_seconds"] for r in selected),
                "latency_p50_mean": statistics.mean(r["latency_p50"] for r in selected),
                "latency_p95_mean": statistics.mean(r["latency_p95"] for r in selected),
                "quality_nll_seed_means": seed_means,
                "quality_nll_mean": statistics.mean(seed_means),
                "quality_nll_sample_sd": statistics.stdev(seed_means),
                "training_targets_per_second": statistics.mean(
                    524288 / r["training_seconds"] for r in selected
                ),
            }
        ratios = {}
        for baseline, candidate in (
            ("mps", "metal"),
            ("mlx", "metal"),
            ("cpu_arm64", "metal"),
            ("cpu_torch", "mps"),
            ("cpu_torch", "cpu_arm64"),
        ):
            train = [
                indexed[s, r, baseline]["training_seconds"]
                / indexed[s, r, candidate]["training_seconds"]
                for s in SEEDS
                for r in (1, 2)
            ]
            numerator = statistics.mean(
                indexed[s, r, baseline]["latency_p50"] for s in SEEDS for r in (1, 2)
            )
            denominator = statistics.mean(
                indexed[s, r, candidate]["latency_p50"] for s in SEEDS for r in (1, 2)
            )
            ratios[f"{baseline}/{candidate}"] = {
                "paired_training_ratios": train,
                "training_geometric_mean": math.exp(statistics.mean(math.log(v) for v in train)),
                "forward_ratio_of_mean_medians": numerator / denominator,
                "qualified": integrity and repeat_ok and accuracy[baseline] and accuracy[candidate],
            }
        by_width[str(width)] = {
            "evidence_integrity_passed": integrity,
            "repeatability_passed": repeat_ok,
            "fp64_accuracy_passed": accuracy,
            "mutual_fp32_compatibility_passed": compatible,
            "engines": engines,
            "ratios": ratios,
        }
    integrity = all(
        w["evidence_integrity_passed"] and w["repeatability_passed"] for w in by_width.values()
    )
    accuracy = all(all(w["fp64_accuracy_passed"].values()) for w in by_width.values())
    return {
        "schema": 1,
        "protocol": PROTOCOL,
        "manifest_sha256": manifest_sha256,
        "status": "qualified"
        if integrity and accuracy
        else "precision_limit"
        if integrity
        else "audit_failed",
        "planned_fits": 180,
        "completed_fits": 180,
        "blocked_fits": 0,
        "planned_test_token_predictions": 1474560,
        "retained_test_token_predictions": 1474560,
        "evidence_integrity_passed": integrity,
        "all_engine_fp64_accuracy_passed": accuracy,
        "by_width": by_width,
        "cells": [{**r["cell"], "status": "completed"} for r in reports],
    }


def markdown(summary):
    lines = [
        "# CPU and GPU matched experiment",
        "",
        f"Status: **{summary['status']}**.",
        "",
        f"Completed {summary['completed_fits']} of {summary['planned_fits']} planned fits; "
        f"retained {summary['retained_test_token_predictions']:,} of "
        f"{summary['planned_test_token_predictions']:,} test-token predictions.",
        "",
    ]
    if summary["status"] in ("blocked_before_measurement", "measurement_stopped_unqualified"):
        lines += ["No qualified timing or speedup claim is available.", ""] + [
            f"- {reason}" for reason in summary.get("reasons", [])
        ]
        if summary["status"] == "measurement_stopped_unqualified":
            lines.append(
                f"{summary['failed_fits']} cells failed; {summary['blocked_fits']} remain unrun. "
                "Partial artifacts are retained."
            )
    else:
        lines += [
            "| Width | Integrity/repeats | All-engine FP64 | "
            "MPS / Metal training | MLX / Metal training |",
            "|---|---|---|---|---|",
        ]
        for width, row in summary["by_width"].items():

            def ratio(key):
                value = row["ratios"][key]
                grade = "qualified" if value["qualified"] else "unqualified"
                return f"{value['training_geometric_mean']:.3f}× ({grade})"

            lines.append(
                f"| {width} | {row['evidence_integrity_passed'] and row['repeatability_passed']} | "
                f"{all(row['fp64_accuracy_passed'].values())} | "
                f"{ratio('mps/metal')} | {ratio('mlx/metal')} |"
            )
        lines += [
            "",
            "Ratios compare matched full training runs. Quality uses three seed means; "
            "timing repeats are not independent quality seeds.",
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocked-reason", action="append")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if args.blocked_reason:
        require(
            args.campaign is None and args.audit is None,
            "Blocked report cannot hide existing campaign artifacts",
        )
        summary = blocked_report(manifest, args.blocked_reason)
    elif args.campaign is not None and args.audit is None:
        summary = unfinished_report(manifest, digest(args.manifest), args.campaign)
    else:
        require(
            args.campaign is not None and args.audit is not None,
            "Campaign and complete audit required",
        )
        sha = digest(args.manifest)
        reports, repeats = load_authenticated_reports(args.audit, args.campaign, sha, manifest)
        summary = summarize(manifest, sha, reports, repeats)
    require(not args.output.exists(), "Report destination must be fresh")
    args.output.mkdir(parents=True)
    write_new(args.output / "summary.json", summary)
    with (args.output / "README.md").open("x") as handle:
        handle.write(markdown(summary))


if __name__ == "__main__":
    main()
