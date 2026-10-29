"""Small archive-integrity tests; no model execution, datasets, or timings."""

import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/package_failed_sweep.py"
SPEC = importlib.util.spec_from_file_location("package_failed_sweep", MODULE_PATH)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def tiny_failed_evidence(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.txt").write_text("frozen measured source\n")
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(["git", "add", "source.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Evidence Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "measured",
        ],
        cwd=repo,
        check=True,
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    inputs, results, report, parent, w512, w1024, gates, extra = (
        tmp_path / name
        for name in ("inputs", "results", "outcome", "parent", "w512", "w1024", "gates", "extra")
    )
    for path in (inputs, results, report, parent, w512, w1024, gates, extra):
        path.mkdir()
    (inputs / "initial.f32").write_bytes(b"original fixture bytes")
    (results / "best.f32").write_bytes(b"original trained checkpoint")
    (extra / "note.txt").write_text("Retained development failure\n")
    failed = [{"name": "width-512/selected_checkpoint/sequence_43", "passed": False}]
    for directory in (parent, w512, w1024):
        write_json(directory / "audit.json", {"passed": False, "checks": failed})
    manifest = {
        "status": "completed",
        "protocol_version": "v5",
        "widths": [512, 1024],
        "revision": revision,
        "source_sha256": {"source.txt": packager.archive.digest(repo / "source.txt")},
        "runs": [{"order": i} for i in range(24)],
        "preflight_directory": str(gates),
        "regression_preflights": {},
    }
    for width in packager.ALL_WIDTHS:
        gate = {"passed": True, "model_width": width}
        for filename, stem in (
            ("transformer-arm64", "executable"),
            ("libtransformer.dylib", "library"),
            ("layout.inc", "generated_layout"),
        ):
            relative = f"native/build/width-{width}/{filename}"
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{width} {filename}\n")
            gate[stem + "_path"] = relative
            gate[stem + "_sha256"] = packager.archive.digest(path)
        modes = {}
        for mode in ("auto", "neon", "sme"):
            modes[mode] = gate | {"policy": mode}
            suffix = "" if mode == "auto" else "-" + mode
            write_json(gates / f"width-{width}{suffix}.json", modes[mode])
        manifest["regression_preflights"][str(width)] = modes
    write_json(results / "manifest.json", manifest)
    summary = {
        "status": "audit_failed",
        "evidence_integrity_passed": True,
        "revision": revision,
        "post_run_failed_check_count": 1,
    }
    write_json(report / "summary.json", summary)
    write_json(report / "audit.json", {"passed": False, "checks": failed})
    verified = {}
    for directory in (inputs, results, report, parent, w512, w1024, gates, repo / "native"):
        for path in directory.rglob("*"):
            if path.is_file():
                verified[str(path.resolve())] = packager.archive.digest(path)
    authenticated = {
        "manifest": manifest,
        "failed_checks": failed,
        "integrity_checks": [{"name": "identities", "passed": True}],
        "measured_source_sha256": manifest["source_sha256"],
        "verified_files": verified,
        "legacy_diagnostic_count": 438,
        "legacy_failed_diagnostic_count": 18,
        "gate_counts": {"gates": 18},
    }
    reporter = tmp_path / "post-study-reporter.py"
    reporter.write_text('"""Post-study tool placeholder for archive-unit tests."""\n')
    monkeypatch.setattr(packager, "REPORTER_PATH", reporter)
    monkeypatch.setattr(packager, "authenticate", lambda args: (authenticated, summary))
    monkeypatch.setattr(
        packager.archive,
        "validate_evidence",
        lambda *args: pytest.fail("Success validator must never be invoked"),
    )
    args = SimpleNamespace(
        repo=repo,
        inputs=inputs,
        results=results,
        report=report,
        parent_failure=parent,
        width_512_report=w512,
        width_1024_report=w1024,
        output=tmp_path / "failed-study.tar.gz",
        extra_evidence=[f"diagnosis={extra}"],
    )
    return args, authenticated, summary


def test_failed_archive_retains_false_audits_full_binaries_and_separate_tool_provenance(
    tmp_path, monkeypatch
):
    args, auth, _ = tiny_failed_evidence(tmp_path, monkeypatch)
    result = packager.package(args)
    assert result["outcome"] == result["status"] == "audit_failed"
    assert result["integrity_verified"] is True
    assert result["numerical_validation_passed"] is False
    assert result["packager_is_measured_source"] is False
    assert result["reporter_is_measured_source"] is False
    hashes = packager.archive.verify_archive(args.output)
    assert sum(name.startswith("preflights/") for name in hashes) == 18
    assert sum(name.startswith("bin/") for name in hashes) == 18
    assert "development/diagnosis/note.txt" in hashes
    with tarfile.open(args.output, "r:gz") as tar:
        outcome = json.load(tar.extractfile("outcome.json"))
        audit = json.load(tar.extractfile("original-failed-reports/parent/audit.json"))
        assert audit["passed"] is False and audit["checks"] == auth["failed_checks"]
        assert outcome["numerical_validation_passed"] is False
        assert outcome["post_run_failed_check_count"] == 1
        assert tar.getmember("bin/width-1024/transformer-arm64").mode == 0o755
    with pytest.raises(FileExistsError):
        packager.package(args)


def test_failed_archive_is_deterministic_and_preserves_original_bytes(tmp_path, monkeypatch):
    args, auth, _ = tiny_failed_evidence(tmp_path, monkeypatch)
    first = packager.package(args)
    args.output = tmp_path / "second.tar.gz"
    second = packager.package(args)
    assert first["sha256"] == second["sha256"]
    for path, expected in auth["verified_files"].items():
        assert packager.archive.digest(path) == expected


@pytest.mark.parametrize(
    "mutation", ["success_status", "no_failure", "false_integrity", "wrong_source"]
)
def test_rejects_success_or_unauthenticated_outcome_before_publication(
    tmp_path, monkeypatch, mutation
):
    args, auth, summary = tiny_failed_evidence(tmp_path, monkeypatch)
    if mutation == "success_status":
        summary["status"] = "completed"
    elif mutation == "no_failure":
        auth["failed_checks"] = []
    elif mutation == "false_integrity":
        auth["integrity_checks"][0]["passed"] = False
    else:
        auth["measured_source_sha256"] = {"source.txt": "0" * 64}
    with pytest.raises(ValueError):
        packager.package(args)
    assert not args.output.exists()


def test_rejects_evidence_changed_after_authentication(tmp_path, monkeypatch):
    args, _, _ = tiny_failed_evidence(tmp_path, monkeypatch)
    original = packager.archive.write_archive

    def change_then_write(output, payload):
        (args.results / "best.f32").write_bytes(b"tampered trained checkpoint")
        return original(output, payload)

    monkeypatch.setattr(packager.archive, "write_archive", change_then_write)
    with pytest.raises(ValueError, match="changed after authentication"):
        packager.package(args)
    assert not args.output.exists()


def test_rejects_omitted_authenticated_file_and_missing_gate(tmp_path, monkeypatch):
    args, auth, _ = tiny_failed_evidence(tmp_path, monkeypatch)
    outside = tmp_path / "unmapped-evidence.bin"
    outside.write_bytes(b"must not silently omit")
    auth["verified_files"][str(outside)] = packager.archive.digest(outside)
    with pytest.raises(ValueError, match="omitted"):
        packager.package(args)
    del auth["verified_files"][str(outside)]
    del auth["manifest"]["regression_preflights"]["512"]["neon"]
    with pytest.raises(ValueError, match="18 policy gates"):
        packager.package(args)
    assert not args.output.exists()


def test_rejects_symlink_extra_and_unsafe_extra_name(tmp_path, monkeypatch):
    args, _, _ = tiny_failed_evidence(tmp_path, monkeypatch)
    target = args.inputs / "initial.f32"
    extra = tmp_path / "extra"
    (extra / "linked.bin").symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        packager.package(args)
    args.extra_evidence = [f"../bad={extra}"]
    with pytest.raises(ValueError, match="NAME=DIRECTORY"):
        packager.package(args)
    assert not args.output.exists()
