"""Tiny archive/provenance regressions; no frameworks, GPU, or numerical models."""

import copy
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts import package_gpu_qkv as package


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        value = package.json_bytes(value)
    path.write_bytes(value)
    return path


def tiny_archive(tmp_path, *, formal=None):
    source = save(tmp_path / "raw.bin", bytes(range(256)) * 3)
    identity, signature = package.old.file_identity(source)
    files = {"evidence/failed/raw.bin": identity}
    manifest = {
        "schema": 1,
        "scope": package.SCOPE,
        "formal_campaign": package.FORMAL if formal is None else formal,
        "files": files,
    }
    output = tmp_path / "output"
    output.mkdir()
    index = package.write_split_archive(
        output,
        "tiny",
        127,
        {"evidence/failed/raw.bin": {"path": source}},
        files,
        {"evidence/failed/raw.bin": signature},
        manifest,
    )
    index_path = output / "index.json"
    package.write_new(index_path, index)
    return source, index_path, index


def test_split_roundtrip_keeps_failure_evidence_and_closed_formal_boundary(tmp_path):
    source, path, index = tiny_archive(tmp_path)
    result = package.verify_archive(path)
    assert index["scope"] == package.SCOPE
    assert len(index["parts"]) > 2
    assert result["archive_integrity_verified"] is True
    assert result["payload_bytes"] == source.stat().st_size
    assert result["numerical_qualification_claimed"] is False


@pytest.mark.parametrize("mutation", ["byte", "missing", "duplicate", "reordered", "escape"])
def test_corrupt_or_incomplete_part_inventory_is_rejected(tmp_path, mutation):
    _, path, index = tiny_archive(tmp_path)
    if mutation == "byte":
        part = path.parent / index["parts"][0]["name"]
        data = bytearray(part.read_bytes())
        data[4] ^= 1
        part.write_bytes(data)
    elif mutation == "missing":
        index["parts"].pop()
    elif mutation == "duplicate":
        index["parts"].append(index["parts"][0])
    elif mutation == "reordered":
        index["parts"][0], index["parts"][1] = index["parts"][1], index["parts"][0]
    else:
        index["parts"][0]["name"] = "../outside"
    path.write_bytes(package.json_bytes(index))
    with pytest.raises((ValueError, OSError, EOFError, tarfile.TarError)):
        package.verify_archive(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("completed_fits", 1),
        ("numerical_gate_waived", True),
        ("reserved_admission_stream_opened", True),
        ("heldout_arrays_read_by_packager", True),
    ],
)
def test_archive_never_promotes_blocked_engineering_to_formal_admission(tmp_path, key, value):
    formal = {**package.FORMAL, key: value}
    _, path, _ = tiny_archive(tmp_path, formal=formal)
    with pytest.raises(ValueError, match="Formal boundary"):
        package.verify_archive(path)


def test_stream_rejects_mutation_after_inventory(tmp_path):
    source = save(tmp_path / "source", b"first")
    identity, signature = package.old.file_identity(source)
    source.write_bytes(b"second")
    with pytest.raises(ValueError, match="Input changed"):
        package.write_split_archive(
            tmp_path,
            "changed",
            128,
            {"x": {"path": source}},
            {"x": identity},
            {"x": signature},
            {"files": {"x": identity}},
        )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("native/metal/qkv.metal", True),
        ("native/metal/build/libtransformer_metal.dylib", False),
        ("experiments/gpu-v3-qkv/candidates.json", True),
        ("experiments/gpu-v3-qkv/results/figure.png", True),
        ("experiments/gpu-v2/policies/selected.json", True),
        ("experiments/gpu-v2/results/large-old-report.json", False),
        ("work/gpu-v2/raw.npz", False),
        ("work/gpu-v3-qkv/raw.npz", False),
        ("results/inputs-v6/width-32/seed-42/test.i32", False),
        ("transformer/__pycache__/code.pyc", False),
        ("scripts/package_gpu_qkv.py", True),
    ],
)
def test_source_inventory_is_limited_to_code_and_required_phase_metadata(name, expected):
    assert package.source_member(name) is expected


def test_evidence_tree_retains_every_failed_and_partial_attempt(tmp_path):
    failed = save(tmp_path / "search/failed/partial.npz", b"failure")
    log = save(tmp_path / "search/failed/error.log", b"failure detail")
    smoke = save(tmp_path / "uncommitted-smoke.json", {"scope": "uncommitted"})
    save(tmp_path / "__pycache__/unused.pyc", b"cache")
    paths, excluded = package.tree_files(tmp_path)
    assert set(paths) == {failed, log, smoke}
    assert excluded == ["__pycache__"]


def test_input_inventory_opens_only_seed42_manifests(tmp_path, monkeypatch):
    calls = []

    def read(path):
        calls.append(Path(path))
        width = int(Path(path).parent.parent.name.removeprefix("width-"))
        return {
            "config": {"width": width, "seed": 42},
            "sha256": {
                "initial.f32": "a" * 64,
                "train.i32": "b" * 64,
                "validation.i32": "c" * 64,
                "test.i32": "d" * 64,
            },
        }

    monkeypatch.setattr(package.old, "read_json", read)
    records = package.input_inventory(tmp_path)
    assert len(calls) == len(package.WIDTHS)
    assert all(p.name == "manifest.json" and p.parent.name == "seed-42" for p in calls)
    assert all(set(r["sha256"]) == {"initial.f32", "train.i32"} for r in records)


def completed_report():
    return {
        "scope": package.SEARCH_SCOPE,
        "status": "completed_development_report",
        "formal_campaign_allowed": False,
        "formal_fits": 0,
        "framework_or_cpu_speed_comparisons_claimed": False,
        "numerically_qualified_development_result": True,
        "repeatability": {},
        "baseline_feasibility": {"reserved_admission_opened": False},
        "widths": [{"width": w} for w in package.WIDTHS],
        "evidence_sha256": {"search/example": "a" * 64},
    }


@pytest.mark.parametrize(
    "mutation", ["interim", "missing_width", "duplicate_width", "waiver", "no_repeat", "admission"]
)
def test_incomplete_or_wrong_scope_report_cannot_be_published(mutation):
    report = completed_report()
    if mutation == "interim":
        report["status"] = "interim_repeatability_pending"
    elif mutation == "missing_width":
        report["widths"].pop()
    elif mutation == "duplicate_width":
        report["widths"][-1] = report["widths"][0]
    elif mutation == "waiver":
        report["framework_or_cpu_speed_comparisons_claimed"] = True
    elif mutation == "no_repeat":
        report["repeatability"] = None
    else:
        report["baseline_feasibility"]["reserved_admission_opened"] = True
    with pytest.raises(ValueError):
        package.validate_report(report)


def test_report_binding_rejects_removed_changed_and_traversal_evidence(tmp_path):
    path = save(tmp_path / "case.json", b"data")
    payload = {"evidence/case.json": {"path": path}}
    files = {"evidence/case.json": {"sha256": sha(b"data")}}
    summary = {"evidence_sha256": {"search/case.json": sha(b"data")}}
    roots = {"search": tmp_path}
    assert package.bind_report(summary, payload, files, roots) == {
        "search/case.json": "evidence/case.json"
    }
    for changed in (
        {"search/missing.json": sha(b"data")},
        {"search/case.json": sha(b"changed")},
        {"search/../case.json": sha(b"data")},
        {"other/case.json": sha(b"data")},
    ):
        with pytest.raises(ValueError):
            package.bind_report({"evidence_sha256": changed}, payload, files, roots)


def test_only_frozen_ledgers_define_measured_snapshots(tmp_path):
    save(tmp_path / "smoke.json", {"source_revision": "uncommitted"})
    ledger = {
        "scope": package.SEARCH_SCOPE,
        "frozen_before_performance_scoring": True,
        "source_revision": "a" * 40,
        "source_sha256": {"native/metal/qkv.metal": "b" * 64},
    }
    path = save(tmp_path / "failed/width-32.json.ledger.json", ledger)
    records = package.measured_ledgers(tmp_path)
    assert records[0]["member"] == "evidence/failed/width-32.json.ledger.json"
    ledger["source_revision"] = "uncommitted"
    path.write_bytes(package.json_bytes(ledger))
    with pytest.raises(ValueError, match="Unbound measured"):
        package.measured_ledgers(tmp_path)


def synthetic_release(tmp_path):
    """A real tiny Git repository; literal byte strings stand in for model files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in package.SOURCE_FILES | {
        "scripts/package_gpu_qkv.py",
        "scripts/report_gpu_qkv.py",
    }:
        save(repo / relative, b"{}\n")
    probe = save(repo / "transformer/probe.py", b"# measured source\n")
    save(repo / ".gitignore", b"work/\nresults/\nnative/metal/build/\n")
    for command in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "tiny source",
        ],
    ):
        subprocess.run(["git", *command], cwd=repo, check=True, capture_output=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    evidence = repo / "work/gpu-v3-qkv"
    searches, replay = evidence / "search/selection-v1", evidence / "repeatability/selected-v1"
    inputs, report = repo / "results/inputs-v6", repo / "results/gpu-v3-qkv"
    binary = save(repo / package.BINARY_PATH, b"fake Metal binary; never execute")
    runtime = save(repo / "work/gpu-v2/runtime-identity.json", {"record_integrity_passed": True})
    historical = save(repo / "work/gpu-v2/old-raw.npz", b"must not open")
    failure = save(evidence / "failed-attempt/partial.npz", b"retain failed raw bytes")
    save(evidence / "smoke-uncommitted.json", {"source_revision": "uncommitted"})
    replay_path = save(replay / "state.bin", b"fake saved replay bytes")
    summary = completed_report()
    summary["widths"], summary["evidence_sha256"] = [], {}
    forbidden = [historical]
    for width in package.WIDTHS:
        root = inputs / f"width-{width}/seed-42"
        initial = save(root / "initial.f32", f"initial-{width}".encode())
        train = save(root / "train.i32", f"train-{width}".encode())
        hashes = {"initial.f32": sha(initial.read_bytes()), "train.i32": sha(train.read_bytes())}
        save(root / "manifest.json", {"config": {"width": width, "seed": 42}, "sha256": hashes})
        forbidden += [
            save(root / n, b"held-out: never open") for n in ("validation.i32", "test.i32")
        ]
        common = {
            "width": width,
            "source_revision": revision,
            "source_sha256": {
                "transformer/probe.py": sha(probe.read_bytes()),
                package.BINARY_PATH: sha(binary.read_bytes()),
            },
            "runtime_identity": {"inventory_sha256": sha(runtime.read_bytes())},
            "input_sha256": hashes,
            "selected_policy": {"id": "selected"},
            "incumbent_policy": {"id": "incumbent"},
        }
        ledger = {
            **common,
            "scope": package.SEARCH_SCOPE,
            "frozen_before_performance_scoring": True,
        }
        name = f"width-{width}.json"
        ledger_path = save(searches / (name + ".ledger.json"), ledger)
        result = {
            **common,
            "scope": package.SEARCH_SCOPE,
            "status": "completed",
            "formal_fits": 0,
            "admission_stream_opened": False,
            "ledger_sha256": sha(ledger_path.read_bytes()),
        }
        result_path = save(searches / name, result)
        for path in (result_path, ledger_path):
            summary["evidence_sha256"]["search/" + path.name] = sha(path.read_bytes())
        summary["widths"].append(common)
    summary["evidence_sha256"]["repeatability/state.bin"] = sha(replay_path.read_bytes())
    save(report / "results.json", summary)
    args = (repo, evidence, inputs, binary, runtime, report, searches, replay, tmp_path / "archive")
    return args, forbidden, failure


def test_complete_package_preserves_sources_training_and_failures_without_heldout_reads(
    tmp_path, monkeypatch
):
    args, forbidden, failure = synthetic_release(tmp_path)
    forbidden = {p.resolve() for p in forbidden}
    actual_open, path_open = os.open, Path.open

    def guarded_os_open(path, *rest, **kwargs):
        assert Path(path).resolve() not in forbidden, "Forbidden array or historical evidence read"
        return actual_open(path, *rest, **kwargs)

    def guarded_path_open(path, *rest, **kwargs):
        assert path.resolve() not in forbidden, "Forbidden array or historical evidence read"
        return path_open(path, *rest, **kwargs)

    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    result = package.package(*args, max_part_bytes=4096)
    assert result["archive_integrity_verified"] is True
    manifest = json.loads((args[-1] / "gpu-v3-qkv-manifest.json").read_text())
    files = manifest["files"]
    assert files["evidence/failed-attempt/partial.npz"]["sha256"] == sha(failure.read_bytes())
    assert "evidence/smoke-uncommitted.json" in files
    assert len(manifest["measured_source_bindings"]) == 6
    assert manifest["historical_gpu_v2"] == package.HISTORICAL_V2
    assert manifest["formal_campaign"] == package.FORMAL
    assert not any(
        "validation.i32" in name or "test.i32" in name or "old-raw.npz" in name for name in files
    )
    assert len([n for n in files if n.startswith("training-inputs/")]) == 18
    assert len([n for n in files if n.startswith("source-snapshots/")]) == 1
    with pytest.raises(ValueError, match="Archive output must be new"):
        package.package(*args)


@pytest.mark.parametrize(
    "mutation", ["binary", "runtime", "input", "ledger", "source", "report_source"]
)
def test_package_rejects_stale_execution_identities(tmp_path, mutation):
    args, _, _ = synthetic_release(tmp_path)
    repo, _, inputs, binary, runtime, report, searches, _, _ = args
    if mutation == "binary":
        binary.write_bytes(b"changed binary")
    elif mutation == "runtime":
        runtime.write_bytes(b"changed runtime")
    elif mutation == "input":
        (inputs / "width-32/seed-42/train.i32").write_bytes(b"changed input")
    elif mutation == "ledger":
        path = searches / "width-32.json.ledger.json"
        value = json.loads(path.read_text())
        value["source_revision"] = "b" * 40
        path.write_bytes(package.json_bytes(value))
    elif mutation == "source":
        # Commit a different measured source while keeping the old byte receipt.
        save(repo / "transformer/probe.py", b"# changed measured source\n")
        subprocess.run(["git", "add", "transformer/probe.py"], cwd=repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "changed",
            ],
            cwd=repo,
            check=True,
        )
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        for path in searches.glob("*.ledger.json"):
            value = json.loads(path.read_text())
            value["source_revision"] = revision
            path.write_bytes(package.json_bytes(value))
    else:
        path = report / "results.json"
        value = json.loads(path.read_text())
        value["widths"][0]["source_revision"] = "b" * 40
        path.write_bytes(package.json_bytes(value))
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        package.package(*args, max_part_bytes=4096)


def test_runtime_and_input_bindings_are_independent_of_numerical_pass_flags():
    summary = {
        "widths": [
            {
                "width": 32,
                "runtime_identity": {"inventory_sha256": "a" * 64},
                "input_sha256": {"initial.f32": "b" * 64, "train.i32": "c" * 64},
            }
        ]
    }
    files = {
        "runtime/runtime-identity.json": {"sha256": "a" * 64},
        "training-inputs/width-32/seed-42/initial.f32": {"sha256": "b" * 64},
        "training-inputs/width-32/seed-42/train.i32": {"sha256": "c" * 64},
    }
    package.bind_runtime_inputs(summary, files)
    bad = copy.deepcopy(summary)
    bad["widths"][0]["input_sha256"]["test.i32"] = "d" * 64
    with pytest.raises(ValueError, match="Measured input inventory"):
        package.bind_runtime_inputs(bad, files)
