"""Bounded archive tests with mocked read-only Git; no project evidence or GPU work."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/package_gpu_development.py"
SPEC = importlib.util.spec_from_file_location("package_gpu_development", SCRIPT)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


@pytest.fixture
def project(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    evidence = repo / "work/gpu-v1"
    files = {
        "README.md": b"Development, not formal results\n",
        "requirements-gpu-development.txt": b"numpy==2.5.2\n",
        "scripts/capture_gpu_environment.py": b"# bounded mock capture source\n",
        "experiments/GPU-PLAN-v1.md": b"planned108 completed0\n",
        "experiments/PROTOCOL-gpu-v1.md": b"admission unresolved\n",
        "native/metal/kernels.metal": b"// source\n",
    }
    for name, data in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    binary = repo / "native/metal/build/libtransformer_metal.dylib"
    binary.parent.mkdir()
    binary.write_bytes(b"small synthetic dylib bytes")
    runtime = {
        "package_count": 1,
        "installed_file_count": 1,
        "record_integrity_passed": True,
        "requirements_identity": {
            "sha256": hashlib.sha256(files["requirements-gpu-development.txt"]).hexdigest()
        },
        "capture_script_identity": {
            "sha256": hashlib.sha256(files["scripts/capture_gpu_environment.py"]).hexdigest()
        },
    }
    evidence_files = {
        "discovery/runtime-identity.json": json.dumps(runtime).encode(),
        "fixtures/width-32/train.i32": b"\0" * 132,
        "integration-v1/metal-32/runtime.json": json.dumps(
            {"scope": "development_only", "status": "failed", "error": "retained failure"}
        ).encode(),
        "integration-v2/metal-32/runtime.json": json.dumps(
            {"scope": "development_only", "status": "completed"}
        ).encode(),
        "superseded/probe.json": b'{"passed": false}',
        "uv-cache/excluded.bin": b"not archived",
        "python-site/mlx/excluded.bin": b"not archived",
    }
    for name, data in evidence_files.items():
        path = evidence / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def fake_git(_repo, arguments):
        assert _repo == repo
        if arguments == ["rev-parse", "HEAD"]:
            return b"1" * 40 + b"\n"
        if arguments[0] == "status":
            return b" M README.md\0?? scripts/capture_gpu_environment.py\0"
        if arguments[0] == "ls-files":
            return b"\0".join(name.encode() for name in sorted(files)) + b"\0"
        raise AssertionError(arguments)

    monkeypatch.setattr(packager, "git_bytes", fake_git)
    return repo, evidence, binary, tmp_path / "development.tar.gz", files


def test_archive_retains_failures_and_checks_every_payload(project):
    repo, evidence, binary, output, files = project
    result = packager.package(repo, evidence, binary, output)
    assert result["archive_integrity_verified"] is True
    assert result["numerical_qualification_claimed"] is False
    assert result["formal_campaign"] == packager.FORMAL
    assert packager.verify_archive(output)["payload_files"] == len(files) + 6
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert not any("uv-cache" in name or "python-site" in name for name in names)
        assert "evidence/integration-v1/metal-32/runtime.json" in names
        assert "evidence/superseded/probe.json" in names
        manifest = json.load(archive.extractfile("MANIFEST.json"))
        assert manifest["source"]["dirty"] is True
        assert manifest["cpu_v6_references"] == packager.CPU_V6
        assert {row["status"] for row in manifest["development_attempt_receipts"]} == {
            "failed",
            "completed",
        }
    assert output.with_name("development-manifest.json").exists()
    assert output.with_name("development-verification.json").exists()
    for line in output.with_name("development-SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((output.parent / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize(
    "collision",
    [
        "development.tar.gz",
        "development-manifest.json",
        "development-verification.json",
        "development-SHA256SUMS.txt",
    ],
)
def test_refuses_output_collision_without_modification(project, collision):
    repo, evidence, binary, output, _ = project
    path = output.with_name(collision)
    path.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        packager.package(repo, evidence, binary, output)
    assert path.read_bytes() == b"preserve"


@pytest.mark.parametrize("location", ["source", "evidence", "binary", "output"])
def test_rejects_symlink_payload_or_output(project, location):
    repo, evidence, binary, output, _ = project
    target = output.parent / "target"
    target.write_bytes(b"outside")
    path = {
        "source": repo / "README.md",
        "evidence": evidence / "linked.bin",
        "binary": binary,
        "output": output,
    }[location]
    if path.exists():
        path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        packager.package(repo, evidence, binary, output)
    assert target.read_bytes() == b"outside"


def test_refuses_output_inside_inputs(project):
    repo, evidence, binary, _, _ = project
    with pytest.raises(ValueError, match="outside"):
        packager.package(repo, evidence, binary, evidence / "bad.tar.gz")


def test_refuses_formal_run_and_stale_dependency_receipt(project):
    repo, evidence, binary, output, _ = project
    path = evidence / "integration-v1/metal-32/runtime.json"
    path.write_text('{"scope":"formal", "status":"completed"}')
    with pytest.raises(ValueError, match="Non-development"):
        packager.package(repo, evidence, binary, output)
    path.write_text('{"scope":"development_only", "status":"failed"}')
    (repo / "requirements-gpu-development.txt").write_text("changed pins")
    with pytest.raises(ValueError, match="Runtime identity"):
        packager.package(repo, evidence, binary, output)
    assert not output.exists()


def test_preserves_partial_failed_receipt_without_inventing_its_scope(project):
    repo, evidence, binary, output, _ = project
    partial = evidence / "integration-v1/metal-32/runtime.json"
    partial.write_bytes(b'{"scope":')
    packager.package(repo, evidence, binary, output)
    with tarfile.open(output) as archive:
        name = "evidence/integration-v1/metal-32/runtime.json"
        assert archive.extractfile(name).read() == b'{"scope":'
        manifest = json.load(archive.extractfile("MANIFEST.json"))
        attempt = next(
            row for row in manifest["development_attempt_receipts"] if row["receipt"] == name
        )
        assert attempt["status"] == "unreadable_receipt"
        assert attempt["scope_verified"] is False


@pytest.mark.parametrize("name", [".", "../bad", "/absolute", "a//b", "a\\b", "a\nb"])
def test_rejects_unsafe_archive_names(name):
    with pytest.raises(ValueError, match="Unsafe"):
        packager.safe_name(name)


def test_input_mutation_is_rejected_and_partial_output_is_preserved(project, monkeypatch):
    repo, evidence, binary, output, _ = project
    original = packager.write_archive

    def mutate_after_archive(*args):
        result = original(*args)
        (evidence / "superseded/probe.json").write_text("changed after archiving")
        return result

    monkeypatch.setattr(packager, "write_archive", mutate_after_archive)
    with pytest.raises(ValueError, match="Input changed"):
        packager.package(repo, evidence, binary, output)
    assert output.exists()
    assert not output.with_name("development-verification.json").exists()


@pytest.mark.parametrize("corruption", ["payload", "duplicate", "symlink", "formal_claim"])
def test_independent_archive_verification_rejects_tampering(project, corruption):
    repo, evidence, binary, output, _ = project
    packager.package(repo, evidence, binary, output)
    bad = output.with_name("tampered.tar.gz")
    with tarfile.open(output) as source, tarfile.open(bad, "w:gz") as destination:
        for member in source:
            data = source.extractfile(member).read()
            if corruption == "payload" and member.name == "source/README.md":
                data = b"x" * len(data)
            if corruption == "formal_claim" and member.name == "MANIFEST.json":
                manifest = json.loads(data)
                manifest["formal_campaign"]["completed_fits"] = 108
                data = json.dumps(manifest).encode()
            info = packager.tar_info(member.name, len(data), member.mode)
            destination.addfile(info, io.BytesIO(data))
        if corruption == "duplicate":
            destination.addfile(packager.tar_info("source/README.md", 1), io.BytesIO(b"x"))
        if corruption == "symlink":
            info = tarfile.TarInfo("linked")
            info.type, info.linkname = tarfile.SYMTYPE, "../../outside"
            destination.addfile(info)
    with pytest.raises(ValueError):
        packager.verify_archive(bad)
