"""Owned diagnostic resource and provenance controls, without framework execution."""

import copy
from types import SimpleNamespace
from unittest import mock

import pytest

from transformer import gpu_owned_repeatability as repetition
from transformer import gpu_study as study


def test_disk_budget_blocks_before_launch_and_preserves_existing_evidence(tmp_path):
    evidence = tmp_path / "failed-worker.json"
    evidence.write_text('{"failed":true}')
    with mock.patch.object(repetition.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
        with pytest.raises(ValueError, match="reserve"):
            repetition.check_storage(tmp_path, initial=True)
    with (
        mock.patch.object(
            repetition.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)
        ),
        mock.patch.dict(repetition.LIMITS, output_bytes=1),
    ):
        with pytest.raises(ValueError, match="output exceeded"):
            repetition.check_storage(tmp_path)
    assert evidence.read_text() == '{"failed":true}'


def test_resource_abort_keeps_partial_log_and_terminates_its_single_worker(tmp_path):
    process = mock.Mock(pid=123, returncode=None)
    process.poll.return_value = None
    process.wait.side_effect = lambda **kwargs: setattr(process, "returncode", -15)
    with (
        mock.patch.object(repetition.subprocess, "Popen", return_value=process),
        mock.patch.object(repetition, "check_storage", side_effect=ValueError("disk exhausted")),
        mock.patch.object(repetition.os, "killpg") as stop,
    ):
        result = repetition.bounded_worker(
            ["never-executed"],
            cwd=tmp_path,
            environment={},
            log=tmp_path / "worker.log",
            output=tmp_path,
        )
    assert result["returncode"] == -15 and "disk exhausted" in result["error"]
    stop.assert_called_once_with(123, repetition.signal.SIGTERM)
    assert (tmp_path / "worker.log").is_file()


def capability_fixture():
    manifest = study.planned_manifest("owned")
    manifest["bindings"] = {
        "runtime": {"runtime/python": "0" * 64},
        "runtime_layout": {"interpreter": "/runtime/python"},
        "baseline": {"owned-runtime.json": "1" * 64},
    }
    capabilities = {
        "baseline_class": "project_maintained_full_source_build",
        "standard_release_claim": False,
        "numerical_admission_claim": False,
        "runtime_identity_sha256": study.identity(manifest["bindings"]["runtime"]),
        "framework_identity": manifest["owned_frameworks"]["frameworks"]["pytorch"],
        "runtime_receipt_sha256": "1" * 64,
        "storage_dtype": "float32",
        "cpu_tensor_fallback": False,
        "deterministic_algorithms": True,
        "loaded_origin_proof": {
            "module_origins": ["/runtime/torch/__init__.py", "/runtime/torch/_C.so"],
            "framework_module_origins": {
                "pytorch": {
                    "module_origins": {
                        "torch": "/runtime/torch/__init__.py",
                        "torch._C": "/runtime/torch/_C.so",
                    }
                }
            },
            "project_module_origins": {
                name: "bound-fixture.py"
                for name in (
                    "transformer",
                    "transformer.spec",
                    "transformer.reference",
                    "transformer.gpu_owned_backend",
                    "transformer.gpu_mps",
                )
            },
        },
    }
    return manifest, {
        "engine": "mps",
        "runtime_root": "/",
        "capabilities": capabilities,
        "capabilities_after": copy.deepcopy(capabilities),
    }


def test_both_actual_capability_snapshots_reach_nested_provenance_auditor():
    manifest, worker = capability_fixture()
    with mock.patch(
        "transformer.gpu_owned_backend.validate_loaded_origin_proof", create=True
    ) as validate:
        assert repetition.validate_capabilities(worker, manifest, {})
    assert validate.call_count == 2


@pytest.mark.parametrize(
    "mutation", ["omit", "runtime", "storage", "fallback", "determinism", "origins"]
)
def test_initial_capability_cannot_hide_changed_postexecution_runtime(mutation):
    manifest, worker = capability_fixture()
    after = worker["capabilities_after"]
    if mutation == "omit":
        del worker["capabilities_after"]
    elif mutation == "runtime":
        after["runtime_identity_sha256"] = "2" * 64
    elif mutation == "storage":
        after["storage_dtype"] = "float16"
    elif mutation == "fallback":
        after["cpu_tensor_fallback"] = True
    elif mutation == "determinism":
        after["deterministic_algorithms"] = False
    else:
        after["loaded_origin_proof"] = {}

    def nested(proof, **kwargs):
        if not proof:
            raise ValueError("Missing actual loaded origins")

    with mock.patch(
        "transformer.gpu_owned_backend.validate_loaded_origin_proof",
        side_effect=nested,
        create=True,
    ):
        with pytest.raises(ValueError):
            repetition.validate_capabilities(worker, manifest, {})
