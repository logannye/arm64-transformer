"""Tiny saved-data checks; no GPU, model arithmetic, plots, or real evidence writes."""

import csv
import hashlib
import json

import pytest

from scripts import package_gpu_v2_engineering as package
from scripts import report_gpu_v2_engineering as report
from scripts.summarize_gpu_v2_baselines import parameter_layout


def tiny_archive(tmp_path, limit=127):
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(256)) * 8)
    identity, signature = package.old.file_identity(source)
    files = {"evidence/raw.bin": identity}
    manifest = {
        "schema": 1,
        "scope": package.SCOPE,
        "formal_campaign": package.FORMAL,
        "files": files,
    }
    output = tmp_path / "output"
    output.mkdir()
    index = package.write_split_archive(
        output,
        "tiny",
        limit,
        {"evidence/raw.bin": {"path": source}},
        files,
        {"evidence/raw.bin": signature},
        manifest,
    )
    index_path = output / "archive.json"
    package.write_new(index_path, index)
    return source, index_path, index


def test_split_archive_verifies_raw_member_across_part_boundaries(tmp_path):
    source, index_path, index = tiny_archive(tmp_path)
    assert len(index["parts"]) > 2
    assert all(part["bytes"] <= 127 for part in index["parts"])
    result = package.verify_archive(index_path)
    assert result["archive_integrity_verified"]
    assert result["payload_bytes"] == source.stat().st_size
    assert result["numerical_qualification_claimed"] is False
    assert source.read_bytes() == bytes(range(256)) * 8


def test_part_tampering_fails_before_decompression(tmp_path):
    _, index_path, index = tiny_archive(tmp_path)
    path = index_path.parent / index["parts"][0]["name"]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="part changed"):
        package.verify_archive(index_path)


def test_reordered_parts_fail_stream_verification(tmp_path):
    _, index_path, index = tiny_archive(tmp_path)
    index["parts"][0], index["parts"][1] = index["parts"][1], index["parts"][0]
    index_path.write_text(json.dumps(index))
    with pytest.raises((ValueError, OSError)):
        package.verify_archive(index_path)


def test_partial_write_never_overwrites_an_existing_part(tmp_path):
    writer = package.SplitWriter(tmp_path, "tiny", 4)
    writer.write(b"12345")
    writer.close()
    other = package.SplitWriter(tmp_path, "tiny", 4)
    with pytest.raises(FileExistsError):
        other.write(b"changed")
    assert (tmp_path / "tiny.tar.gz.part-00001").read_bytes() == b"1234"


def test_archive_rejects_source_mutation_between_inventory_and_stream(tmp_path):
    source = tmp_path / "original"
    source.write_bytes(b"first")
    identity, signature = package.old.file_identity(source)
    source.write_bytes(b"second")
    with pytest.raises(ValueError, match="Input changed"):
        package.write_split_archive(
            tmp_path,
            "bad",
            128,
            {"x": {"path": source}},
            {"x": identity},
            {"x": signature},
            {"files": {"x": identity}},
        )


def test_evidence_tree_preserves_failure_files_and_excludes_only_named_caches(tmp_path):
    (tmp_path / "failed").mkdir()
    (tmp_path / "failed/partial.npz").write_bytes(b"failure")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/unused.pyc").write_bytes(b"cache")
    files, excluded = package.tree_files(tmp_path)
    assert files == [tmp_path / "failed/partial.npz"]
    assert excluded == ["__pycache__"]


def pair_fixture():
    pair = {}
    for label, latency in (("reference", 2.0), ("candidate", 1.0)):
        pair[label] = {
            "warmups": 20,
            "observations": 100,
            "blocks": 5,
            "seconds": [latency] * 100,
            "block_median_seconds": [latency] * 5,
            "exact_block_replay": True,
            "exact_state_restoration": True,
            "guards": True,
        }
    pair.update(paired_latency_ratios=[0.5] * 5, median_paired_latency_ratio=0.5)
    return pair


def test_report_recomputes_direct_paired_ratios_from_raw_observations():
    result = report.paired_summary(pair_fixture())
    assert result["candidate_over_reference_latency"] == 0.5
    assert result["latency_reduction_fraction"] == 0.5
    assert "not a confidence interval" in result["dispersion_label"]


def test_report_rejects_invented_ratio_and_failed_timing_control():
    pair = pair_fixture()
    pair["median_paired_latency_ratio"] = 0.25
    with pytest.raises(ValueError, match="paired summary"):
        report.paired_summary(pair)
    pair = pair_fixture()
    pair["candidate"]["exact_state_restoration"] = False
    with pytest.raises(ValueError, match="control failed"):
        report.paired_summary(pair)


def test_exact_replay_cannot_pass_from_boolean_without_fingerprint():
    assert not report.exact_pass({"passed": True, "different_coordinates": 0})
    fingerprint = {"sha256": hashlib.sha256(b"bytes").hexdigest(), "shape": [1], "dtype": "<f4"}
    assert report.exact_pass(
        {
            "passed": True,
            "different_coordinates": 0,
            "actual": fingerprint,
            "expected": fingerprint.copy(),
        }
    )


def test_recorded_parameter_layout_matches_frozen_layout_without_model_import():
    from transformer import spec

    layout, count = parameter_layout(spec.WIDTH)
    assert layout == spec.PARAMS
    assert count == spec.PARAM_COUNT


def test_archive_index_cannot_reference_parent_path(tmp_path):
    _, index_path, index = tiny_archive(tmp_path)
    index["parts"][0]["name"] = "../outside"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Unsafe"):
        package.verify_archive(index_path)


def report_inventory_fixture(tmp_path):
    widths = (32, 64, 128, 256, 512, 1024)
    roots = {key: str(tmp_path / key) for key in ("baseline", "profiles", "searches")}
    return {
        "scope": package.SCOPE,
        "formal_admission_opened": False,
        "formal_fits_completed": 0,
        "framework_or_cpu_speedup_claimed": False,
        "baselines": [
            {"engine": engine, "width": width, "standard_baseline_qualified": False}
            for engine in ("mps", "mlx")
            for width in widths
        ],
        "metal": [{"width": width, "search_status": "completed"} for width in widths],
        "input_roots": roots,
        "source_artifacts": {"baseline/summary.json": "0" * 64},
        "eligible_search_attempt": "selection-v2",
    }


def test_finalized_blocked_report_accepts_no_improvement_and_retained_search_failure(tmp_path):
    summary = report_inventory_fixture(tmp_path)
    summary["metal"][0]["search_status"] = "failed"
    package.validate_report_inventory(summary)
    summary["metal"].pop()
    with pytest.raises(ValueError, match="all six finalized"):
        package.validate_report_inventory(summary)


def test_report_binding_requires_digest_at_the_correct_archive_path(tmp_path):
    summary = report_inventory_fixture(tmp_path)
    files = {"evidence/unrelated.json": {"sha256": "0" * 64}}
    with pytest.raises(ValueError, match="absent or changed"):
        package.bind_report_files(tmp_path, summary, files)


def test_finalization_does_not_accept_started_search_or_formal_waiver(tmp_path):
    summary = report_inventory_fixture(tmp_path)
    summary["metal"][0]["search_status"] = "started"
    with pytest.raises(ValueError, match="all six finalized"):
        package.validate_report_inventory(summary)
    summary = report_inventory_fixture(tmp_path)
    summary["formal_admission_opened"] = True
    with pytest.raises(ValueError, match="Blocked release"):
        package.validate_report_inventory(summary)


def test_invalidated_search_attempt_is_rejected_before_any_timing_read(tmp_path):
    (tmp_path / "invalidated.json").write_text("{}")
    with pytest.raises(ValueError, match="invalidated"):
        report.metal_rows(tmp_path, tmp_path, report.Receipts())


def test_report_does_not_silently_select_another_attempt(tmp_path):
    (tmp_path / "attempt.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "scope": "metal_development_search_attempt",
                "attempt_id": "selection-v1",
                "eligible_for_reporting": True,
            }
        )
    )
    with pytest.raises(ValueError, match="Explicit eligible"):
        report.metal_rows(tmp_path, tmp_path, report.Receipts())


def test_saved_sample_percentiles_and_training_throughput_are_explicit():
    assert report.percentile(list(range(1, 101)), 0.5) == 50.5
    assert report.percentile(list(range(1, 101)), 0.95) == pytest.approx(95.05)
    result = report.paired_summary(pair_fixture(), mode="training")
    assert result["candidate_p50_us"] == 1e6
    assert result["candidate_p95_ms"] == 1000
    assert result["candidate_training_targets_per_second"] == 512
    assert result["reference_training_targets_per_second"] == 256
    inference = report.paired_summary(pair_fixture(), mode="inference")
    assert inference["candidate_training_targets_per_second"] is None


def fresh_pair_fixture():
    fields = [
        "owned_buffer_count",
        "workspace_buffer_count",
        "input_buffer_count",
        "batch_buffer_count",
        "owned_buffer_bytes_including_guards",
    ]
    memory = dict(zip(fields, (102, 99, 0, 3, 4096), strict=True))
    return {
        "fresh_instances_without_qualification_allocations": True,
        "equal_initial_state": True,
        "memory_before_timing": {
            "reference": {**memory, "device_current_allocated_bytes": 10000},
            "candidate": {**memory, "device_current_allocated_bytes": 20000},
        },
        "allocation_validation": {
            "equal_owned_counts_and_bytes": True,
            "compared_fields": fields,
        },
    }


def test_fresh_pair_proof_checks_owned_buffers_and_ignores_global_device_counter():
    result = report.validate_fresh_pair(fresh_pair_fixture())
    assert result["batch_buffer_count_per_backend"] == 3
    assert result["device_global_allocation_or_process_rss_compared"] is False
    receipt = fresh_pair_fixture()
    receipt["memory_before_timing"]["candidate"]["owned_buffer_bytes_including_guards"] += 1
    with pytest.raises(ValueError, match="asymmetric"):
        report.validate_fresh_pair(receipt)


def test_fresh_pair_rejects_extra_qualification_buffers_or_missing_state_proof():
    receipt = fresh_pair_fixture()
    receipt["memory_before_timing"]["candidate"]["batch_buffer_count"] = 4
    with pytest.raises(ValueError, match="B16T32/B1T32"):
        report.validate_fresh_pair(receipt)
    receipt = fresh_pair_fixture()
    receipt["equal_initial_state"] = False
    with pytest.raises(ValueError, match="initial-state"):
        report.validate_fresh_pair(receipt)


def profile_fixture():
    stages = (
        ("qkv_forward", 3),
        ("dense_dw/qkv", 2),
        ("bias_gradient/qkv", 1),
        ("dense_forward/ff1", 4),
    )
    records = [
        {
            "stage": stage,
            "gpu_seconds": cost * repetition * 1e-6,
            "host_completion_seconds": cost * repetition * 2e-6,
        }
        for repetition in (1, 2, 3)
        for stage, cost in stages
    ]
    return {
        "scope": report.PROFILE_SCOPE,
        "repetitions": 3,
        "restored_warmups": 20,
        "records": records,
        "median_gpu_seconds": {stage: cost * 2e-6 for stage, cost in stages},
    }


def test_profile_cost_categories_are_recomputed_without_inventing_dispatch_counts():
    result = report.profile_summary(profile_fixture())
    assert result["category_sum_of_stage_medians_seconds"]["qkv"] == pytest.approx(10e-6)
    assert result["category_sum_of_stage_medians_seconds"]["reductions"] == pytest.approx(2e-6)
    assert result["total_sum_of_stage_medians_seconds"] == pytest.approx(20e-6)
    assert result["profiled_stage_command_buffers_per_repetition"] == 4
    assert result["kernel_dispatches_per_repetition"] is None
    assert result["production_runtime_attribution_claimed"] is False


def test_profile_rejects_changed_medians_and_records_optional_dispatches_only_if_retained():
    profile = profile_fixture()
    profile["median_gpu_seconds"]["qkv_forward"] *= 2
    with pytest.raises(ValueError, match="median differs"):
        report.profile_summary(profile)
    profile = profile_fixture()
    for record in profile["records"]:
        record["dispatch_count"] = 2
    assert report.profile_summary(profile)["kernel_dispatches_per_repetition"] == 8


def test_profile_comparison_requires_matching_frozen_runtime_and_inputs():
    identity = {
        name: "identity"
        for name in (
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
        )
    }
    reference = {**identity, "status": "completed", "profiles": {"training": profile_fixture()}}
    selected = {
        **identity,
        "profile_identity_verified": True,
        "selected_profiles_after_production": {"training": profile_fixture()},
    }
    result = report.compare_profiles(reference, selected, "training")
    assert result["selected_minus_reference_profiled_stage_command_buffers"] == 0
    assert result["selected_minus_reference_kernel_dispatches"] is None
    selected["runtime_identity"] = "other-runtime"
    with pytest.raises(ValueError, match="identity differs: runtime_identity"):
        report.compare_profiles(reference, selected, "training")


def test_raw_profile_csv_retains_all_repetitions_and_labels_intrusive_scope(tmp_path):
    profile = report.profile_summary(profile_fixture())
    summary = {
        "metal": [
            {
                "width": 32,
                "modes": {
                    "training": {
                        "kernel_profile_comparison": {"reference": profile, "selected": profile},
                    }
                },
            }
        ]
    }
    report.write_profile_tables(summary, tmp_path)
    with (tmp_path / "kernel-stage-costs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert {row["repetition"] for row in rows} == {"0", "1", "2"}
    assert all(row["scope"] == report.PROFILE_SCOPE for row in rows)
    assert all(row["kernel_dispatch_count_if_retained"] == "" for row in rows)
