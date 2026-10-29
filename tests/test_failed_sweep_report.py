"""Failed-study reporting must preserve failures and authenticate descriptive data."""

import copy
import json
import subprocess

import numpy as np
import pytest

from scripts import report_failed_sweep as report


def original_failed_report():
    entries = [
        {
            "name": f"seed-{seed}-repeat-{repeat}-{impl}",
            "seed": seed,
            "repeat": repeat,
            "implementation": impl,
        }
        for seed in report.SEEDS
        for repeat in (1, 2)
        for impl in report.IMPLEMENTATIONS
    ]
    records = []
    for name in sorted(report.expected_audit_names(entries)):
        record = {"name": name, "passed": True}
        if "/selected_checkpoint/sequence_" in name:
            record.update(
                tolerance="forward",
                actual_shape=[32, 128],
                expected_shape=[32, 128],
                max_absolute_error=1e-6,
                rmse=1e-7,
            )
        records.append(record)
    failed = next(r for r in records if "-arm64/selected_checkpoint/sequence_0" in r["name"])
    failed.update(passed=False, max_absolute_error=5e-5)
    audit = {
        "passed": False,
        "error": None,
        "checks": records,
        "check_count": len(records),
        "failed_checks": [failed["name"]],
    }
    return audit, {"status": "audit_failed", "runs": entries}, entries


def test_original_failures_are_retained_undiluted():
    audit, summary, entries = original_failed_report()
    before = copy.deepcopy(audit)
    failed = report.validate_failed_audit(audit, summary, entries)
    assert len(failed) == 1 and failed[0]["passed"] is False
    assert failed[0]["tolerance"] == "forward"
    assert audit == before
    assert len(audit["checks"]) == 3411


@pytest.mark.parametrize(
    "mutation",
    ["drop_failure", "drop_pass", "duplicate", "tolerance", "shape", "nonfinite", "summary_pass"],
)
def test_original_audit_tampering_is_rejected(mutation):
    audit, summary, entries = original_failed_report()
    failed = next(r for r in audit["checks"] if not r["passed"])
    if mutation in ("drop_failure", "drop_pass"):
        target = (
            failed
            if mutation == "drop_failure"
            else next(r for r in audit["checks"] if r["passed"])
        )
        audit["checks"].remove(target)
        audit["check_count"] -= 1
        audit["failed_checks"] = [r["name"] for r in audit["checks"] if not r["passed"]]
    elif mutation == "duplicate":
        audit["checks"].append(audit["checks"][0])
        audit["check_count"] += 1
    elif mutation == "tolerance":
        failed["tolerance"] = "trajectory"
    elif mutation == "shape":
        failed["actual_shape"] = [1, 128]
    elif mutation == "nonfinite":
        failed["rmse"] = float("nan")
    else:
        summary["status"] = "completed"
    with pytest.raises(ValueError):
        report.validate_failed_audit(audit, summary, entries)


def test_non_checkpoint_failure_is_never_hidden():
    audit, summary, entries = original_failed_report()
    record = next(r for r in audit["checks"] if r["name"].endswith("optimizer_step_count"))
    record["passed"] = False
    audit["failed_checks"] = [r["name"] for r in audit["checks"] if not r["passed"]]
    with pytest.raises(ValueError, match="Unexpected failed audit class"):
        report.validate_failed_audit(audit, summary, entries)


def test_source_blobs_authenticated_while_later_readme_edits_allowed(tmp_path, monkeypatch):
    names = (
        "transformer/sweep_report.py",
        "transformer/sweep.py",
        "transformer/report.py",
        "README.md",
    )
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path).decode().strip()

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    revision = git("rev-parse", "HEAD")
    hashes = {name: report.digest(tmp_path / name) for name in names}
    monkeypatch.setattr(report, "SOURCE_FILES", ())
    (tmp_path / "README.md").write_text("post-study narrative\n")
    report.authenticate_git_sources(tmp_path, revision, hashes)
    wrong = dict(hashes, **{"README.md": "0" * 64})
    with pytest.raises(ValueError, match="Source mismatch"):
        report.authenticate_git_sources(tmp_path, revision, wrong)
    (tmp_path / "transformer/sweep.py").write_text("altered validator\n")
    with pytest.raises(ValueError, match="Changed reused validator"):
        report.authenticate_git_sources(tmp_path, revision, hashes)


def synthetic_evidence():
    runs = {}
    for width in report.WIDTHS:
        runs[width] = []
        for index, seed in enumerate(report.SEEDS):
            for repeat in (1, 2):
                for impl in report.IMPLEMENTATIONS:
                    train = (10 + 2 * index) * repeat / (2 if impl == "arm64" else 1)
                    runs[width].append(
                        {
                            "seed": seed,
                            "repeat": repeat,
                            "implementation": impl,
                            "training_seconds": train,
                            "training_cpu_seconds": train,
                            "process_wall_seconds": train + 1,
                            "peak_rss_after_bytes": 2**20,
                            "peak_rss_final_bytes": 2**20,
                            "quality": {
                                "tokens": 8192,
                                "nll": 2 + index * 0.1,
                                "bits_per_character": 3,
                                "perplexity": 8,
                                "accuracy": 0.3,
                            },
                            "forward_latency_seconds": {"p50": 0.001, "p95": 0.002, "p99": 0.003},
                            "selected_checkpoint_validation_nll": 2.2,
                        }
                    )
    return {
        "manifest": {
            "revision": "a" * 40,
            "models": {str(w): {"width": w, "context": 32} for w in report.WIDTHS},
            "runs": list(range(24)),
            "validation_contract": "v5-common-state-replay-v1",
            "source_sha256": {},
            "fixtures": {"512": {"42": {"config": {"epochs": 8, "train_sequences": 2048}}}},
        },
        "runs": runs,
        "paired_predictions": {
            w: [
                {"seed": seed, "repeat": repeat, "prediction_disagreement": 0.1}
                for seed in report.SEEDS
                for repeat in (1, 2)
            ]
            for w in report.WIDTHS
        },
        "baselines": {"uniform": {"nll": 4.8}, "train_only_unigram": {"nll": 3.0}},
        "raw_manifest_sha256": "b" * 64,
        "original_audits": {
            512: {"check_count": 3411, "failed_checks": ["a", "b"]},
            1024: {"check_count": 3411, "failed_checks": list(range(48))},
        },
        "failed_checks": [{"name": str(i), "passed": False} for i in range(50)],
        "integrity_checks": [{"name": "bytes", "passed": True}],
        "gate_counts": {"all": 21734, "auto": 7248, "forced": 14486},
        "legacy_diagnostic_count": 438,
        "legacy_failed_diagnostic_count": 18,
        "verified_files": {},
    }


def test_descriptive_math_preserves_failed_status_and_distinct_failure_counts():
    evidence = synthetic_evidence()
    summary = report.descriptive_summary(evidence)
    audit = report.failed_outcome_audit(evidence)
    assert summary["status"] == "audit_failed" and audit["passed"] is False
    assert summary["fit_count"] == 24
    assert summary["training_target_tokens_per_fit"] == 524288
    assert summary["retained_test_prediction_rows"] == 196608
    assert summary["post_run_failed_check_count"] == 50
    assert summary["legacy_failed_diagnostic_count"] == 18
    assert audit["check_count"] == 6822 and audit["integrity_check_count"] == 1
    stats = summary["descriptive_by_width"]["512"]["aggregate"]
    assert stats["by_implementation"]["python"]["training_seconds"]["seed_values"] == [15, 18, 21]
    assert stats["by_implementation"]["python"]["training_seconds"]["sample_sd"] == 3
    assert stats["geometric_mean_training_ratio_python_over_arm64"] == pytest.approx(2)
    assert summary["descriptive_by_width"]["512"]["prediction_disagreement"][
        "mean"
    ] == pytest.approx(0.1)
    text = " ".join(report.report_markdown(summary).split())
    assert "No validated speed or numerical-equivalence claim" in text
    assert "**18 failures**" in text


def test_changed_outcome_is_rejected_before_packaging(tmp_path):
    evidence = synthetic_evidence()
    summary = report.descriptive_summary(evidence)
    summary["status"] = "completed"
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Outcome summary was altered"):
        report.validate_failed_report(tmp_path, evidence)


def test_existing_outcome_directory_is_never_overwritten(tmp_path):
    marker = tmp_path / "original.txt"
    marker.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        report.create_report("unused", "unused", "unused", {}, tmp_path)
    assert marker.read_bytes() == b"preserve"


def test_saved_output_hash_changes_rejected(tmp_path):
    path = tmp_path / "logits.f32"
    np.array([1, 2, 3], dtype=np.float32).tofile(path)
    expected = report.digest(path)
    verified = {}
    report.verify_file(path, expected, verified)
    np.array([1, 2, 4], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match="Evidence hash mismatch"):
        report.verify_file(path, expected, verified)


def test_shortened_admitted_file_hash_maps_are_rejected(tmp_path):
    preflight = tmp_path / "preflights"
    preflight.mkdir()
    gate = {}
    hashes = {}
    for prefix in ("library", "executable", "generated_layout"):
        path = tmp_path / prefix
        path.write_text(prefix)
        gate[prefix + "_path"] = str(path)
        gate[prefix + "_sha256"] = report.digest(path)
        hashes[str(path)] = report.digest(path)
    gate_path = preflight / "width-512.json"
    gate_path.write_text(json.dumps(gate))
    hashes[str(gate_path)] = report.digest(gate_path)
    fixtures = {}
    for width in report.WIDTHS:
        for seed in report.SEEDS:
            folder = tmp_path / "inputs" / f"width-{width}" / f"seed-{seed}"
            folder.mkdir(parents=True)
            for name in (*report.FIXTURE_FILES, "manifest.json"):
                path = folder / name
                path.write_text(name)
                fixtures[str(path)] = report.digest(path)
    manifest = {
        "regression_preflights": {"512": {"auto": gate}},
        "preflight_directory": str(preflight),
        "authenticated_files": hashes,
        "fixture_files": fixtures,
    }
    report.authenticate_admitted_files(manifest, tmp_path / "inputs", {})
    del hashes[gate["library_path"]]
    with pytest.raises(ValueError, match="coverage changed"):
        report.authenticate_admitted_files(manifest, tmp_path / "inputs", {})
    hashes[gate["library_path"]] = gate["library_sha256"]
    fixtures.pop(next(iter(fixtures)))
    with pytest.raises(ValueError, match="fixture coverage changed"):
        report.authenticate_admitted_files(manifest, tmp_path / "inputs", {})
