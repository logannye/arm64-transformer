"""V6 reports retain numerical failures and metrics without claiming admission."""

import json

import pytest

from transformer import report, spec
from transformer.backend import validation_contract
from transformer.precision_audit import POSTRUN_CONTRACT, category_summary, comparison


@pytest.mark.parametrize(
    "failed_category,status,qualified",
    [
        (None, "completed", True),
        ("native_fp64", "native_precision_limit", False),
        ("python_fp64", "baseline_precision_limit", False),
        ("fp32_compatibility", "completed", True),
    ],
)
def test_report_keeps_complete_metrics_and_separate_failed_categories(
    tmp_path, monkeypatch, failed_category, status, qualified
):
    results = tmp_path / "results"
    results.mkdir()
    inputs = tmp_path / "inputs"
    output = tmp_path / "report"
    hashes = {f"{split}.i32": split for split in ("train", "validation", "test")}
    fixtures = {str(seed): {"sha256": hashes, "config": {}} for seed in spec.SEEDS}
    preflight = {
        "passed": True,
        "runtime_included": True,
        "check_count": 1,
        "legacy_failed_checks": [],
        "backend": {},
    }
    entries = []
    for seed in spec.SEEDS:
        for repeat in (1, 2):
            for impl in ("python", "arm64"):
                entries.append(
                    {
                        "name": f"seed-{seed}-repeat-{repeat}-{impl}",
                        "seed": seed,
                        "repeat": repeat,
                        "implementation": impl,
                        "returncode": 0,
                        "conditions_valid": True,
                        "source_unchanged": True,
                        "binary_unchanged": True,
                        "process_wall_seconds": 3.0,
                        "backend": {},
                        "backend_valid": True,
                        "backend_sha256": "hash",
                        "backend_environment": "auto",
                    }
                )
    manifest = {
        "model": {"width": spec.WIDTH},
        "protocol_version": "v6",
        "postrun_numerical_contract": POSTRUN_CONTRACT,
        "validation_contract": validation_contract(spec.WIDTH),
        "status": "completed",
        "hardware": {"stdout": "test CPU"},
        "platform": "test platform",
        "torch": "test version",
        "numpy": "test version",
        "source_sha256": {},
        "revision": "frozen source",
        "preflight": preflight,
        "backend_preflights": {"neon": preflight},
        "fixtures": fixtures,
        "runs": entries,
    }
    (results / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(report, "SOURCE_FILES", ())
    monkeypatch.setattr(report, "sha256", lambda path: "hash")
    monkeypatch.setattr(report, "validate_validation_contract", lambda gate, width: (0, 0))
    monkeypatch.setattr(report, "validate_backend_preflights", lambda *args: 1)
    monkeypatch.setattr(report, "read_backend", lambda *args, **kwargs: {})
    monkeypatch.setattr(report, "validate_runtime_backend", lambda *args, **kwargs: None)
    monkeypatch.setattr(report, "compare_repeats", lambda *args: None)
    monkeypatch.setattr(report, "count_baselines", lambda *args: {})
    monkeypatch.setattr(report, "paired_predictions", lambda *args: {})
    monkeypatch.setattr(report, "charts", lambda *args: None)

    def audit_run(fixture, run, checks, *, recompute, numerical_contract, implementation):
        assert recompute and numerical_contract == POSTRUN_CONTRACT
        seed = int(fixture.name.split("-")[1])
        checks.require(run.name + "/integrity", True)
        categories = {}
        for category in ("native_fp64", "python_fp64", "fp32_compatibility"):
            record = comparison(
                run.name + "/v6/test/sequence_0/" + category,
                [0.001 if category == failed_category else 0.0],
                [0.0],
            )
            categories[category] = category_summary([record])
        return {
            "seed": seed,
            "fixture_sha256": hashes,
            "quality": {
                "nll": 2,
                "bits_per_character": 3,
                "perplexity": 8,
                "accuracy": 0.3,
                "tokens": 8192,
            },
            "training_seconds": 1 if implementation == "arm64" else 2,
            "training_cpu_seconds": 1,
            "peak_rss_after_bytes": 2**20,
            "peak_rss_final_bytes": 2**20,
            "best_epoch": 1,
            "forward_latency_seconds": {"p50": 0.001, "p95": 0.002, "p99": 0.003},
            "selected_checkpoint_validation_nll": 2.2,
            "numerical_audit": {"contract": POSTRUN_CONTRACT, "categories": categories},
        }

    monkeypatch.setattr(report, "audit_run", audit_run)
    audit = report.create_report(results, inputs, output)
    summary = json.loads((output / "summary.json").read_text())
    text = (output / "README.md").read_text()
    assert audit["error"] is None
    assert audit["status"] == summary["status"] == status
    assert audit["passed"] is summary["numerically_qualified"] is qualified
    assert audit["evidence_integrity_passed"] is True
    assert all(record["passed"] for record in audit["checks"])
    assert len(summary["runs"]) == 12
    assert summary["aggregate"]["geometric_mean_training_ratio_python_over_arm64"] == 2
    assert audit["numerical_check_count"] == 36
    if failed_category:
        group = audit["numerical_categories"][failed_category]
        assert group["passed"] is False
        assert group["failed_logit_coordinates"] == 12
        assert len(group["failed_checks"]) == 12
        assert sum(not record["passed"] for record in audit["numerical_checks"]) == 12
    if not qualified:
        assert "no numerically qualified speedup is claimed" in text
