"""Output-only quiescence may retry metadata, never corrupted bytes or inputs."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "integration/library-study-v1"


def load(name):
    spec = importlib.util.spec_from_file_location("test_export_" + name, ROOT / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


IO, C, ADAPTER, AUDIT = (
    load(n) for n in ("artifact_io", "contract", "optimizer_adapter", "audit_optimizer")
)


def test_fresh_export_is_durable_exact_and_independently_stable_twice(tmp_path):
    data = b"exact state bytes" * 1024
    path = tmp_path / "checkpoint.f32"
    bound, proof = IO.write_exact(path, data, C)
    assert path.read_bytes() == data
    assert bound["sha256"] == hashlib.sha256(data).hexdigest()
    assert proof["fsync"] and proof["stable_reads"] == 2 and proof["metadata_retries"] == 0
    with pytest.raises(ValueError, match="Fresh canonical"):
        IO.write_exact(path, b"overwrite", C)
    assert path.read_bytes() == data


def test_only_metadata_races_are_retried_with_exact_source_bytes(tmp_path, monkeypatch):
    original, calls = C.binding, []

    def binding(path):
        calls.append(path)
        if len(calls) == 1:
            raise ValueError("Bound file changed during read")
        return original(path)

    monkeypatch.setattr(C, "binding", binding)
    bound, proof = IO.write_exact(tmp_path / "state.f32", b"unchanged", C)
    assert bound["sha256"] == hashlib.sha256(b"unchanged").hexdigest()
    assert proof["metadata_retries"] == 1 and len(calls) == 3


def test_corrupt_export_is_not_rescued_by_retry(tmp_path, monkeypatch):
    original = C.binding
    monkeypatch.setattr(C, "binding", lambda path: dict(original(path), sha256="0" * 64))
    with pytest.raises(ValueError, match="in-memory source bytes"):
        IO.write_exact(tmp_path / "corrupt.f32", b"expected", C)


def test_permanent_race_exhausts_a_bounded_budget(tmp_path, monkeypatch):
    def race(path):
        raise ValueError("Bound file changed during read")

    monkeypatch.setattr(C, "binding", race)
    with pytest.raises(ValueError, match="within export budget"):
        IO.write_exact(tmp_path / "unstable.f32", b"expected", C)


def test_scoped_optimizer_uses_original_bytecode_without_mutating_module_globals():
    namespace = dict(
        SUITE="old-suite", REPLAY="old-replay", save_state=lambda value: "old", update=object()
    )
    exec("def run(value):\n return SUITE, REPLAY, save_state(value), update\n", namespace)
    original = namespace["run"]
    overrides = dict(
        SUITE="new-suite", REPLAY="new-replay", save_state=lambda value: "exact-export"
    )
    function = ADAPTER.scoped_function(original, overrides)
    assert function.__code__ is original.__code__
    assert function(None) == ("new-suite", "new-replay", "exact-export", namespace["update"])
    assert original(None)[:3] == ("old-suite", "old-replay", "old")
    with pytest.raises(ValueError, match="only declared control"):
        ADAPTER.scoped_function(original, dict(overrides, update=lambda: None))


def test_export_auditor_requires_every_saved_file_and_source_hash(tmp_path):
    bound, proof = IO.write_exact(tmp_path / "step-000-parameters.f32", b"exact", C)
    value = dict(
        saved=[dict(files=dict(parameters=bound))], control_adapter=dict(writer_proofs=[proof])
    )
    AUDIT.validate_exports(value)
    value["control_adapter"]["writer_proofs"] = []
    with pytest.raises(ValueError, match="inventory differs"):
        AUDIT.validate_exports(value)
