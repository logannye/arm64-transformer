"""Frozen-control metadata mutations without importing a runtime or opening RNGs."""

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_frozen_library_authority", ROOT / "integration/library-study-v1/authority.py"
)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
C = A.C


def test_frozen_numerical_source_plan_retains_every_gate_and_fit(monkeypatch):
    plan = C.plan()
    assert C.validate_plan(plan) is plan
    assert (
        plan["math_revision"] == plan["cpu_revision"] == "9f57a0d58011dbb27cf0cd8f7a4862fc538df65b"
    )
    assert plan["protocol"] == "gpu-v5-library-five-engine-v2"
    assert plan["source_roles"] == ["control", "math"]
    assert plan["admission_seed"] == 20261112
    for field in ("admission_seed", "math_revision", "source_roles"):
        broken = copy.deepcopy(plan)
        broken[field] = True
        with pytest.raises(ValueError, match="contract changed"):
            C.validate_plan(broken)
    assert plan["formal_execution_supported"] is C.EXECUTION_SUPPORTED
    monkeypatch.setattr(C, "EXECUTION_SUPPORTED", False)
    with pytest.raises(ValueError, match="not implemented"):
        C.require_execution_support()
    assert not plan["formal_eligible"]


def test_control_metadata_rejects_duplicates_nonfinite_values_and_aliases(tmp_path):
    path = tmp_path / "authority.json"
    for text, message in (('{"x":1,"x":2}', "Duplicate"), ('{"x":NaN}', "Nonfinite")):
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            C.read(path)
    path.write_text('{"x":1}')
    bound = C.binding(path)
    assert C.read(path, bound["sha256"]) == {"x": 1}
    with pytest.raises(ValueError, match="trust anchor"):
        C.read(path, "0" * 64)
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="canonical"):
        C.binding(alias)
    with pytest.raises(ValueError, match="Malformed"):
        C.bound(dict(bound, bytes=True))


def test_source_hash_rejects_metadata_races_even_when_bytes_are_unchanged(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"unchanged source")
    original = C.hashlib.file_digest

    def racing_digest(stream, algorithm):
        result = original(stream, algorithm)
        metadata = path.stat()
        os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000))
        return result

    with mock.patch.object(C.hashlib, "file_digest", side_effect=racing_digest):
        with pytest.raises(ValueError, match="changed during read"):
            C.binding(path)
    assert path.read_bytes() == b"unchanged source"


def test_execution_environment_completes_frozen_pins_without_mutating_parent():
    parent = {
        "ARM64_TRANSFORMER_BACKEND": "neon",
        "PYTORCH_MPS_PREFER_METAL": "0",
        "MLX_DISABLE_COMPILE": "0",
        "OMP_NUM_THREADS": "8",
        "METAL_DEBUG_ERROR_MODE": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": "/untrusted/bytecode",
    }
    unchanged = parent.copy()
    child = A.execution_environment(parent)
    assert parent == unchanged
    assert child["ARM64_TRANSFORMER_BACKEND"] == "auto"
    assert child["METAL_DEBUG_ERROR_MODE"] == "0"
    assert {n for n in child if n.startswith("PYTHON")} == {
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
    }
    for name, value in C.comparison()["environment"].items():
        if value is None:
            assert name not in child
        else:
            assert child[name] == value


def test_two_source_roles_allow_only_explicitly_sealed_files(tmp_path):
    control, math = tmp_path / "control", tmp_path / "math"
    control.mkdir()
    math.mkdir()
    a, b = control / "dispatcher.py", math / "model.py"
    a.write_text("control")
    b.write_text("numerical source")
    roles = {
        "control": dict(root=str(control), files={a.name: C.binding(a)["sha256"]}),
        "math": dict(root=str(math), files={b.name: C.binding(b)["sha256"]}),
    }
    project = A.merged_project(roles)
    assert project["root"] == str(tmp_path)
    assert set(project["files"]) == {"control/dispatcher.py", "math/model.py"}
    assert "control/unsealed-probe.py" not in project["files"]
    for name in ("../escape.py", "/absolute.py", "bad\\path.py"):
        with pytest.raises(ValueError, match="Unsafe"):
            A.relative(name)


def test_independent_oracle_cannot_resolve_an_editable_shadow_checkout(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "test_frozen_audit_imports", ROOT / "integration/library-study-v1/audit.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    math, shadow = tmp_path / "math", tmp_path / "editable"
    math.mkdir()
    shadow.mkdir()
    source, wrong = math / "reference.py", shadow / "reference.py"
    source.write_text("frozen oracle")
    wrong.write_text("frozen oracle")
    role = dict(root=str(math), files={source.name: C.binding(source)["sha256"]})
    result = audit.project_origins(
        role, {"transformer.reference": SimpleNamespace(__file__=str(source))}
    )
    assert result["transformer.reference"]["sha256"] == role["files"][source.name]
    with pytest.raises(ValueError, match="another project checkout"):
        audit.project_origins(role, {"transformer.reference": SimpleNamespace(__file__=str(wrong))})
    source.write_text("changed oracle")
    with pytest.raises(ValueError, match="Unbound"):
        audit.project_origins(
            role, {"transformer.reference": SimpleNamespace(__file__=str(source))}
        )


@pytest.mark.parametrize("tamper", [False, True])
def test_activation_authenticates_bytes_before_exposing_dependency_paths(
    tmp_path, monkeypatch, tamper
):
    math, control, dependency = (tmp_path / n for n in ("math", "control", "dependency"))
    for directory in (math, control, dependency):
        directory.mkdir()
    payload = dependency / "payload.bin"
    payload.write_bytes(b"selected runtime bytes")
    payload_binding = C.binding(payload)
    roles = {
        name: dict(root=str(root), revision="0" * 40, files={})
        for name, root in (("control", control), ("math", math))
    }

    def write(name, value):
        path = tmp_path / (name + ".json")
        C.write(path, value)
        return C.binding(path)

    capture = dict(
        schema="gpu-owned-runtime-capture-v1",
        status="identity_captured_not_qualified",
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
        runtime_root=str(tmp_path),
        runtime_inventory=write("inventory", dict(python_paths=["dependency"])),
        runtime_receipt=write("receipt", dict(project=dict(root=str(math), files={}))),
        framework_selection=write("selection", C.comparison()["framework_selection"]),
    )
    binding = write("capture", capture)
    process = SimpleNamespace(
        flags=SimpleNamespace(
            safe_path=True, no_site=True, dont_write_bytecode=True, optimize=0, hash_randomization=0
        ),
        modules={},
        path=["trusted-stdlib"],
        pycache_prefix=str(tmp_path / "unused"),
    )
    monkeypatch.setattr(A, "sys", process)
    monkeypatch.setattr(
        A,
        "os",
        SimpleNamespace(environ=A.execution_environment({"PYTHONNOUSERSITE": "1"}), path=os.path),
    )
    checked_roles, authentications = [], []
    monkeypatch.setattr(A, "verify_role", lambda role, **kw: checked_roles.append(role))
    authority = object()

    def authenticate(*args):
        assert str(dependency) not in process.path
        assert checked_roles == [roles["control"], roles["math"]]
        C.bound(payload_binding)
        authentications.append(args)
        return authority

    modules = {
        "transformer": {},
        "transformer.gpu_owned_backend": dict(authenticate_runtime=authenticate),
        "transformer.gpu_environment": dict(require_environment=lambda value: None),
        "transformer.gpu_study": dict(ENVIRONMENT={}),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    if tamper:
        payload.write_bytes(b"changed runtime bytes!")
        with pytest.raises(ValueError, match="bytes changed"):
            A.activate(roles, binding)
        assert str(dependency) not in process.path
        assert not authentications
    else:
        runtime = A.activate(roles, binding)
        assert runtime.authority is authority
        assert len(authentications) == 1
        assert str(dependency) in process.path
