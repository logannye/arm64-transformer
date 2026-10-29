"""Static publication projections preserve failure meaning and source identities."""

import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/publish_sweep_view.py"
SPEC = importlib.util.spec_from_file_location("publish_sweep_view", PATH)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
RELEASE = "https://github.com/logannye/arm64-transformer/releases/tag/attention-precision-v6"


def test_compaction_keeps_all_metrics_false_flags_and_failure_counts_without_mutation():
    value = {
        "passed": False,
        "status": "baseline_precision_limit",
        "numerically_qualified": False,
        "numerical_check_count": 2,
        "numerical_checks": [{"name": "one", "passed": True}, {"name": "two", "passed": False}],
        "runs": [
            {
                "training_seconds": 4.5,
                "numerical_audit": {
                    "categories": {
                        "python_fp64": {
                            "passed": False,
                            "check_count": 1,
                            "failed_checks": ["two"],
                            "failed_coordinates": 7,
                            "checks": [{"name": "two", "passed": False}],
                        }
                    }
                },
            }
        ],
        "precision": {
            "diagnostics": [{"name": "compatibility", "passed": False}],
            "diagnostic_count": 1,
            "failed_diagnostics": ["compatibility"],
        },
    }
    original = copy.deepcopy(value)
    result = publisher.compact(value)
    assert value == original
    assert result["passed"] is False
    assert result["numerically_qualified"] is False
    assert result["status"] == "baseline_precision_limit"
    assert result["numerical_check_count"] == 2
    assert "numerical_checks" not in result
    assert result["omitted_full_record_counts"] == {"numerical_checks": 2}
    assert result["runs"][0]["training_seconds"] == 4.5
    category = result["runs"][0]["numerical_audit"]["categories"]["python_fp64"]
    assert category["passed"] is False and category["failed_coordinates"] == 7
    assert category["failed_checks"] == ["two"]
    assert category["omitted_full_record_counts"] == {"checks": 1}
    assert result["precision"]["failed_diagnostics"] == ["compatibility"]


def test_audit_projection_is_explicitly_an_index_of_unchanged_full_evidence():
    projected = publisher.project_json(
        {"passed": False, "checks": [{"passed": True}]}, "width-1024/audit.json", "a" * 64, RELEASE
    )
    assert projected["passed"] is False
    publication = projected["publication"]
    assert publication["artifact_type"] == "audit_index"
    assert publication["complete_check_records_present"] is False
    assert publication["full_source"] == {
        "archive_path": "report/width-1024/audit.json",
        "sha256": "a" * 64,
        "release_url": RELEASE,
    }


def test_readme_keeps_results_and_replaces_false_complete_local_audit_claims():
    text = (
        "# Results\n\nStatus: **`baseline_precision_limit`**. Native passes: True.\n\n"
        "All 98,304 saved test prediction rows and\n"
        "their full numerical evidence are retained in [summary.json](summary.json) and\n"
        "[audit.json](audit.json).\n"
        "See [audit.json](audit.json) for every post-run check.\n"
        "![Result](numerical-accuracy.png)\n"
    )
    projected = publisher.publication_readme(text, RELEASE)
    assert "Status: **`baseline_precision_limit`**. Native passes: True." in projected
    assert "full numerical evidence are retained in [summary.json]" not in projected
    assert "for every post-run check" not in projected
    assert "[release archive](" + RELEASE + ")" in projected
    assert "[audit.json](audit.json)" in projected
    assert "![Result](numerical-accuracy.png)" in projected
    assert "All 98,304 saved test prediction rows" in projected


@pytest.mark.parametrize(
    "release",
    [
        "http://github.com/a/b/releases/tag/v6",
        "https://example.org/releases/tag/v6",
        "https://github.com/a/b/releases/tag/v6\nunsafe",
    ],
)
def test_publication_rejects_unsafe_or_ambiguous_release_links(tmp_path, release):
    report = tmp_path / "full"
    report.mkdir()
    with pytest.raises(ValueError, match="release-tag"):
        publisher.publish(report, tmp_path / "compact", release)
    assert not (tmp_path / "compact").exists()


def test_publication_refuses_overwrite_or_source_directory_mutation(tmp_path):
    report = tmp_path / "full"
    report.mkdir()
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("user data")
    with pytest.raises(ValueError, match="exists"):
        publisher.publish(report, output, RELEASE)
    assert (output / "keep").read_text() == "user data"
    with pytest.raises(ValueError, match="inside"):
        publisher.publish(report, report / "compact", RELEASE)
