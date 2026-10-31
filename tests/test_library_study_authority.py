"""Frozen-control metadata mutations without importing a runtime or opening RNGs."""

import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_frozen_library_authority", ROOT / "integration/library-study-v1/authority.py"
)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
C = A.C


def test_frozen_numerical_source_plan_retains_every_gate_and_fit():
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
    with pytest.raises(ValueError, match="not implemented"):
        C.require_execution_support()
    assert not plan["formal_execution_supported"] and not plan["formal_eligible"]


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
