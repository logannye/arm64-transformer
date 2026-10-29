"""Tiny saved-data checks; no GPU, model arithmetic, plots, or real evidence writes."""

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
