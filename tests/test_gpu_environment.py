"""Control-plane regressions; no numerical framework imports or GPU dispatch."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from transformer.gpu_environment import (
    ENVIRONMENT,
    child_environment,
    configure_environment,
    require_environment,
)

FORBIDDEN = ("MLX_DISABLE_COMPILE", "PYTORCH_MPS_PREFER_METAL")


@pytest.mark.parametrize("name", FORBIDDEN)
@pytest.mark.parametrize("value", ("0", "1", "", "false"))
def test_presence_based_controls_rejected_in_process(name, value):
    environment = {name: value}
    with pytest.raises(ValueError, match="must be absent"):
        configure_environment(environment=environment)
    assert environment == {name: value}


def test_manifest_null_means_absence_and_subprocess_values_are_strings():
    pins = json.loads(json.dumps(ENVIRONMENT))
    base = {"KEEP": "yes", **{name: "0" for name in FORBIDDEN}}
    actual = child_environment(pins, base=base, ARM64_TRANSFORMER_WIDTH="1024")
    assert all(name not in actual and pins[name] is None for name in FORBIDDEN)
    assert actual["KEEP"] == "yes" and actual["ARM64_TRANSFORMER_WIDTH"] == "1024"
    assert all(isinstance(value, str) for value in actual.values())
    assert base == {"KEEP": "yes", **{name: "0" for name in FORBIDDEN}}
    require_environment(pins, actual)


def test_environment_configuration_keeps_value_based_controls_disabled():
    actual = {}
    configure_environment(environment=actual)
    assert all(name not in actual for name in FORBIDDEN)
    for name in (
        "MLX_ENABLE_TF32",
        "MLX_SDPA_VJP_FALLBACK",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_FAST_MATH",
    ):
        assert actual[name] == "0"
    require_environment(ENVIRONMENT, actual)


@pytest.mark.parametrize("name", FORBIDDEN)
def test_child_cannot_reintroduce_forbidden_control(name):
    with pytest.raises(ValueError, match="Cannot override"):
        child_environment(base={}, **{name: "0"})


@pytest.mark.parametrize(
    "module,name",
    (("gpu_mlx", "MLX_DISABLE_COMPILE"), ("gpu_mps", "PYTORCH_MPS_PREFER_METAL")),
)
@pytest.mark.parametrize("value", ("0", "1", ""))
def test_backend_rejects_presence_before_importing_framework(module, name, value):
    code = f"""
import importlib, sys
try:
    importlib.import_module('transformer.{module}')
except (ValueError, RuntimeError) as error:
    assert 'must be absent' in str(error), str(error)
    assert 'mlx.core' not in sys.modules
    assert 'torch' not in sys.modules
    assert 'numpy' not in sys.modules
else:
    raise AssertionError('backend import accepted a forbidden variable')
"""
    environment = child_environment(base=os.environ)
    environment[name] = value
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_new_contracts_bind_absence_and_reject_old_manifest():
    from transformer import gpu_repeatability, gpu_study

    assert gpu_repeatability.CONTRACT.endswith("-v2")
    assert gpu_study.PROTOCOL.endswith("-v2")
    manifest = gpu_study.planned_manifest()
    assert all(manifest["environment"][name] is None for name in FORBIDDEN)
    gpu_study.validate_manifest(manifest)
    manifest["environment"]["MLX_DISABLE_COMPILE"] = "0"
    with pytest.raises(ValueError):
        gpu_study.validate_manifest(manifest)


def test_repeatability_binds_environment_and_probe_sources():
    from transformer.gpu_repeatability import source_identity

    mlx = source_identity("mlx")
    mps = source_identity("mps")
    assert "transformer/gpu_environment.py" in mlx
    assert "transformer/gpu_environment.py" in mps
    assert "transformer/gpu_mlx_compile.py" in mlx


def test_replay_and_cpu_bridge_remove_inherited_presence_controls(tmp_path, monkeypatch):
    from transformer import gpu_study as study
    from transformer import gpu_study_audit as audit
    from transformer import gpu_study_cpu_admission as cpu

    for name in FORBIDDEN:
        monkeypatch.setenv(name, "0")
    captured = []

    def record(command, **kwargs):
        captured.append(kwargs["env"])
        assert all(name not in kwargs["env"] for name in FORBIDDEN)
        assert kwargs["env"]["ARM64_TRANSFORMER_WIDTH"] == "32"

    monkeypatch.setattr(audit.subprocess, "run", record)
    monkeypatch.setattr(study, "verify_bindings", lambda *args: None)
    monkeypatch.setattr(study, "python_path", lambda *args, **kwargs: "frozen-python")
    monkeypatch.setattr(audit, "python_path", lambda *args, **kwargs: "frozen-python")
    monkeypatch.setattr(audit, "policy_path", lambda *args: tmp_path / "policy.json")
    roots = {"source": tmp_path, "cpu": tmp_path}
    cpu.run_cpu_operation("cpu_torch", {}, roots, tmp_path / "input", tmp_path / "output", width=32)
    audit.replay_engine(
        "mlx",
        roots,
        32,
        tmp_path / "fixture",
        tmp_path / "checkpoint",
        tmp_path / "replay",
        manifest={},
    )
    assert len(captured) == 2


@pytest.mark.parametrize("name", FORBIDDEN)
@pytest.mark.parametrize(
    "script,function,arguments",
    (
        ("gpu_study_replay.py", "replay", ["cpu_torch", "absent", "absent", "absent"]),
        ("gpu_study_cpu_admission.py", "execute", ["cpu_torch", "absent", "absent"]),
    ),
)
def test_standalone_cpu_workers_validate_before_any_numerical_import(
    name, script, function, arguments
):
    source = Path(__file__).resolve().parents[1] / "transformer" / script
    code = f"""
import runpy, sys
worker = runpy.run_path({str(source)!r})
try:
    worker[{function!r}](*{arguments!r})
except ValueError as error:
    assert 'must be absent' in str(error), str(error)
    assert 'torch' not in sys.modules
    assert 'numpy' not in sys.modules
else:
    raise AssertionError('worker accepted a forbidden variable')
"""
    environment = child_environment(base=os.environ)
    environment[name] = "0"
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
