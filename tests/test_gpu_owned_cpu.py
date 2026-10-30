"""Frozen CPU provenance launcher controls; no Torch imports or numerical runs."""

import copy
import json
from unittest import mock

import pytest

from transformer import gpu_owned_cpu as cpu
from transformer import gpu_study as study


@pytest.fixture
def authority(tmp_path):
    manifest = study.planned_manifest("owned")
    manifest.update(phase="frozen", admission={"sha256": "1" * 64})
    runtime = {
        "runtime/torch/__init__.py": "a" * 64,
        "runtime/torch/_C.fixture.so": "b" * 64,
        "runtime/torch/lib/libtorch_cpu.dylib": "c" * 64,
        "runtime/torch/_ops.py": "4" * 64,
        "runtime/torch/_classes.py": "5" * 64,
    }
    source = {
        cpu.LAUNCHER: "d" * 64,
        cpu.HELPER: "e" * 64,
        "transformer/gpu_study_cpu_admission.py": "f" * 64,
        "transformer/gpu_study_replay.py": "0" * 64,
    }
    frozen = {
        "transformer/" + name + ".py": str(index) * 64
        for index, name in enumerate(("__init__", "reference", "spec", "python_train"))
    }
    manifest["bindings"] = {
        "runtime": runtime,
        "runtime_python_paths": [],
        "source": source,
        "cpu": frozen,
        "torch_distribution_sha256": cpu.identity(runtime),
        "runtime_layout": {"packages": {"torch": ["/runtime/torch/__init__.py", "/runtime/torch"]}},
    }
    output = tmp_path / "fit"
    value = {
        "schema": "gpu-owned-cpu-authority-v1",
        "protocol": cpu.PROTOCOL,
        "manifest": manifest,
        "manifest_identity_sha256": cpu.identity(manifest),
        "kind": "fit",
        "engine": "cpu_torch",
        "output": str(output),
        "fixture": str(tmp_path / "fixture"),
        "arguments": [str(tmp_path / "fixture"), str(output)],
        "entrypoint": {
            "role": "cpu",
            "path": "transformer/python_train.py",
            "sha256": frozen["transformer/python_train.py"],
        },
        "runtime_identity_sha256": cpu.identity(runtime),
        "expected_steps": 1024,
        "roots": {"source": "/source", "cpu": "/cpu", "runtime": "/"},
        "admission_receipt": str(tmp_path / "admission.json"),
    }
    return value


def test_frozen_cpu_authority_requires_same_runtime_and_original_entrypoint(authority):
    assert cpu.validate_authority(authority) is authority
    assert cpu.identity(authority["manifest"]) == study.identity(authority["manifest"])


@pytest.mark.parametrize(
    "mutation",
    [
        "entrypoint",
        "engine",
        "namespace",
        "seed",
        "runtime",
        "manifest",
        "steps",
        "args",
        "unsealed",
        "helper",
    ],
)
def test_cpu_launch_rejects_mutations_before_importing_torch(authority, mutation):
    if mutation == "entrypoint":
        authority["entrypoint"]["role"] = "source"
    elif mutation == "engine":
        authority["engine"] = "mps"
    elif mutation == "namespace":
        authority["protocol"] = "gpu-v2-five-engine-v2"
    elif mutation == "seed":
        authority["manifest"]["admission_seed"] = 20261104
    elif mutation == "runtime":
        authority["runtime_identity_sha256"] = "0" * 64
    elif mutation == "manifest":
        authority["manifest"]["workload"]["epochs"] = 1
    elif mutation == "steps":
        authority["expected_steps"] = 1
    elif mutation == "args":
        authority["arguments"][-1] += "-substituted"
    elif mutation == "unsealed":
        authority["manifest"]["admission"] = None
        authority["manifest_identity_sha256"] = cpu.identity(authority["manifest"])
    else:
        del authority["manifest"]["bindings"]["source"][cpu.HELPER]
        authority["manifest_identity_sha256"] = cpu.identity(authority["manifest"])
    with pytest.raises(ValueError):
        cpu.validate_authority(authority)


def test_stale_child_authority_rejected_before_import_or_backend(tmp_path):
    path = tmp_path / "authority.json"
    path.write_text('{"passed": true}')
    with pytest.raises(ValueError, match="trust anchor"):
        cpu.authenticate(path, "0" * 64)


def test_cpu_guard_revalidates_raw_admission_in_separate_control_plane(authority, tmp_path):
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority))
    with (
        mock.patch.object(cpu, "guard_controls"),
        mock.patch.object(cpu, "file_at"),
        mock.patch.object(cpu.sys, "path", list(cpu.sys.path)),
        mock.patch("transformer.gpu_study.verify_admission") as verified,
    ):
        cpu.guard(path, cpu.digest(path))
        verified.assert_called_once_with(
            authority["manifest"], authority["roots"], authority["admission_receipt"]
        )
    with (
        mock.patch.object(cpu, "guard_controls"),
        mock.patch.object(cpu, "file_at"),
        mock.patch.object(cpu.sys, "path", list(cpu.sys.path)),
        mock.patch("transformer.gpu_study.verify_admission", side_effect=ValueError("raw failure")),
    ):
        with pytest.raises(ValueError, match="raw failure"):
            cpu.guard(path, cpu.digest(path))


@pytest.fixture
def proof(authority, tmp_path):
    manifest = authority["manifest"]
    core = "/runtime/torch/_C.fixture.so"
    origins = {
        "framework": {
            "schema": "gpu-owned-loaded-framework-origins-v1",
            "framework": "pytorch",
            "module_origins": {
                "torch": "/runtime/torch/__init__.py",
                "torch._C": core,
                "torch._ops": "/runtime/torch/_ops.py",
                "torch._classes": "/runtime/torch/_classes.py",
            },
            "synthetic_origins": {
                name: {
                    "backing_module": backing,
                    "backing_file": "/runtime/torch/" + label,
                    "backing_sha256": manifest["bindings"]["runtime"]["runtime/torch/" + label],
                    "exact_class": backing + "." + cls,
                    "singleton_identity_verified": True,
                }
                for name, backing, label, cls in (
                    ("torch.ops", "torch._ops", "_ops.py", "_Ops"),
                    ("torch.classes", "torch._classes", "_classes.py", "_Classes"),
                )
            },
            "native_aliases": {"torch._C._fixture": {"extension": core, "sha256": "b" * 64}},
            "namespace_packages": {},
            "module_count": 7,
        },
        "core": core,
        "source_revision": manifest["owned_frameworks"]["frameworks"]["pytorch"]["base"],
        "operator_libraries": [],
        "native_libraries": {"runtime/torch/lib/libtorch_cpu.dylib": "c" * 64},
        "cpu_modules": {
            name: {
                "path": name.replace(".", "/") + ".py",
                "sha256": manifest["bindings"]["cpu"][name.replace(".", "/") + ".py"],
            }
            for name in ("transformer.reference", "transformer.spec")
        },
        "deterministic": True,
        "warn_only": False,
        "host_threads": 1,
        "interop_threads": 1,
    }
    origins["cpu_modules"]["transformer"] = {"path": "transformer/__init__.py", "sha256": "0" * 64}
    value = {
        "schema": cpu.SCHEMA,
        "protocol": cpu.PROTOCOL,
        "kind": "fit",
        "engine": "cpu_torch",
        "manifest_identity_sha256": cpu.identity(manifest),
        "runtime_identity_sha256": authority["runtime_identity_sha256"],
        "source_sha256": manifest["bindings"]["source"],
        "cpu_source_sha256": manifest["bindings"]["cpu"],
        "entrypoint": authority["entrypoint"],
        "complete": True,
        "integrity_after": True,
        "output_sha256": {"best.f32": "5" * 64},
        "input_sha256": {"/old/fixture/config.bin": "6" * 64},
        "expected_steps": 1024,
        "actual_steps": 1024,
        "runtime_root": "/",
        "origins_before": copy.deepcopy(origins),
        "origins_after": copy.deepcopy(origins),
    }
    path = tmp_path / "proof.json"
    return path, value, manifest


def verify(proof):
    path, value, manifest = proof
    path.write_text(json.dumps(value))
    return cpu.verify_proof(
        path,
        manifest,
        engine="cpu_torch",
        kind="fit",
        output_hashes={"best.f32": "5" * 64},
        input_hashes={"config.bin": "6" * 64},
    )


def test_cpu_proof_revalidated_without_live_runtime(proof):
    assert verify(proof)["actual_steps"] == 1024


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "runtime",
        "input",
        "checkpoint",
        "source",
        "entrypoint",
        "steps",
        "error",
        "native",
        "core",
        "modules",
        "warn",
        "threads",
    ],
)
def test_missing_or_substituted_cpu_evidence_cannot_qualify(proof, mutation):
    path, value, manifest = proof
    if mutation == "missing":
        with pytest.raises(FileNotFoundError):
            cpu.verify_proof(path, manifest, engine="cpu_torch", kind="fit", output_hashes={})
        return
    if mutation == "runtime":
        value["runtime_identity_sha256"] = "0" * 64
    elif mutation == "input":
        value["input_sha256"]["/old/fixture/config.bin"] = "7" * 64
    elif mutation == "checkpoint":
        value["output_sha256"]["best.f32"] = "7" * 64
    elif mutation == "source":
        value["source_sha256"] = {}
    elif mutation == "entrypoint":
        value["entrypoint"] = {
            "role": "source",
            "path": "transformer/python_train.py",
            "sha256": "3" * 64,
        }
    elif mutation == "steps":
        value["actual_steps"] = 1023
    elif mutation == "error":
        value["error"] = "late integrity failure"
    elif mutation == "native":
        value["origins_after"]["native_libraries"] = {
            "runtime/torch/lib/libtorch_cpu.dylib": "8" * 64
        }
    elif mutation == "core":
        value["origins_after"]["core"] = "/unselected/torch/_C.fixture.so"
    elif mutation == "modules":
        value["origins_after"]["cpu_modules"] = {}
    elif mutation == "warn":
        value["origins_after"]["warn_only"] = True
    else:
        value["origins_after"]["interop_threads"] = 8
    with pytest.raises(ValueError):
        verify(proof)


@pytest.mark.parametrize(
    "mutation",
    ["section", "count", "synthetic", "backing", "singleton", "alias", "duplicate", "namespace"],
)
def test_removed_or_forged_framework_origin_records_fail(proof, mutation):
    framework = proof[1]["origins_after"]["framework"]
    if mutation == "section":
        del framework["native_aliases"]
    elif mutation == "count":
        framework["module_count"] -= 1
    elif mutation == "synthetic":
        del framework["synthetic_origins"]["torch.ops"]
        framework["module_count"] -= 1
    elif mutation == "backing":
        framework["synthetic_origins"]["torch.ops"]["backing_sha256"] = "0" * 64
    elif mutation == "singleton":
        framework["synthetic_origins"]["torch.classes"]["singleton_identity_verified"] = False
    elif mutation == "alias":
        framework["native_aliases"]["torch._C._fixture"]["sha256"] = "0" * 64
    elif mutation == "duplicate":
        framework["native_aliases"]["torch._C"] = framework["native_aliases"].pop(
            "torch._C._fixture"
        )
    else:
        framework["namespace_packages"] = {"torch": {}}
    with pytest.raises(ValueError):
        verify(proof)


@pytest.mark.parametrize("expected", ["Malformed CPU parameters", "Different failure", None])
def test_expected_rejection_keeps_postexecution_identity_checks(authority, tmp_path, expected):
    script = tmp_path / "frozen.py"
    script.write_text("raise ValueError('Malformed CPU parameters')\n")
    authority.update(
        kind="admission",
        proof=str(tmp_path / "proof.json"),
        input_sha256={},
        expected_steps=None,
        expected_rejection=expected,
    )
    with (
        mock.patch.object(cpu, "authenticate", return_value=authority),
        mock.patch.object(cpu, "torch_proof", return_value={"authenticated": True}) as origins,
        mock.patch.object(cpu, "file_at", return_value=script),
    ):
        if expected == "Malformed CPU parameters":
            result = cpu.execute("authority.json", "0" * 64)
            assert result["complete"] and result["integrity_after"]
            assert result["output_sha256"] == {}
            assert result["rejection"] == {"type": "ValueError", "message": expected}
            assert origins.call_count == 2
        else:
            with pytest.raises(ValueError, match="Malformed CPU parameters"):
                cpu.execute("authority.json", "0" * 64)
            result = json.loads((tmp_path / "proof.json").read_text())
            assert not result["complete"] and "error" in result


def test_expected_failure_cannot_qualify_as_successful_execution(proof):
    proof[1]["expected_rejection"] = "Malformed CPU parameters"
    proof[1]["rejection"] = {"type": "ValueError", "message": "Malformed CPU parameters"}
    with pytest.raises(ValueError, match="rejection contract"):
        verify(proof)
