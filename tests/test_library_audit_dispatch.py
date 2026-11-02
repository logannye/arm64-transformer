"""Independent audit dispatch cannot change program or open a reserved stream."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_audit_dispatch", ROOT / "integration/library-study-v1/audit_dispatch.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def test_only_four_frozen_raw_auditors_are_dispatchable():
    assert set(D.AUDITORS) == {"primitive", "optimizer", "provenance", "collection"}
    assert all(
        path.startswith("integration/") and path.endswith(".py") for path in D.AUDITORS.values()
    )
    with pytest.raises(ValueError, match="independent audit Python"):
        D.execute(SimpleNamespace(kind="formal_fit", width=32))


def test_auditor_stdout_cannot_grow_without_bound(monkeypatch):
    monkeypatch.setattr(D.C, "MAX_JSON_BYTES", 16)
    output = D.BoundedOutput()
    output.write('{"result":true}')
    assert output.getvalue() == '{"result":true}'
    with pytest.raises(ValueError, match="exceeds bound"):
        output.write("too much")
