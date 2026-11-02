"""Saved import maps must remain confined to the sealed runtime and math role."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_suite_audit", ROOT / "integration/library-study-v1/qualify_suites.py"
)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def test_retained_imports_reject_shadow_files_and_changed_bytes(tmp_path):
    runtime, math, shadow = (tmp_path / name for name in ("runtime", "math", "shadow"))
    for root in (runtime, math, shadow):
        root.mkdir()
        (root / "source.py").write_text("same source")
    digest = Q.C.binding(runtime / "source.py")["sha256"]
    inventory = dict(root=str(runtime), files={"source.py": digest})
    project = dict(root=str(math), files={"source.py": digest})
    saved = {
        "numpy": dict(path=str(runtime / "source.py"), sha256=digest),
        "transformer": dict(path=str(math / "source.py"), sha256=digest),
    }
    Q.imports(saved, inventory, project)
    with pytest.raises(ValueError, match="authority"):
        Q.imports(
            {"numpy": dict(path=str(shadow / "source.py"), sha256=digest)}, inventory, project
        )
    (runtime / "source.py").write_text("changed source")
    with pytest.raises(ValueError, match="changed"):
        Q.imports(saved, inventory, project)
