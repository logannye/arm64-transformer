"""Cross-engine gradient proofs and audit-only reuse preserve producer identity."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_development_audit", ROOT / "integration/library-study-v1/audit_development.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


@pytest.mark.parametrize("width", [32, 64, 128, 256, 512, 1024])
def test_frozen_native_adam_gates_require_all_gpu_gradient_sources(monkeypatch, width):
    def gradient_sources(comparison):
        return {
            g["id"].split("/")[3].removeprefix("from-")
            for g in comparison["gate_inventory"]
            if g["id"].split("/")[0] in ("cpu_arm64", "metal")
            and g["id"].split("/")[1:3] == [str(width), "adam"]
        }

    sources = gradient_sources(D.C.original_comparison())
    assert sources == {"cpu_arm64", "cpu_torch", "mps", "mlx", "metal"}
    assert gradient_sources(D.C.comparison()) == (sources if width in D.C.WIDTHS else set())
    capabilities = {e: object() for e in sources - {"cpu_arm64", "cpu_torch"}}
    checked = {}
    monkeypatch.setattr(D.E, "validate_capabilities", lambda e, cap: checked.update({e: cap}))
    D.validate_capabilities(capabilities)
    assert checked == capabilities
    for omitted in capabilities:
        with pytest.raises(ValueError, match="capability proof"):
            D.validate_capabilities({e: cap for e, cap in capabilities.items() if e != omitted})
    with pytest.raises(ValueError, match="capability proof"):
        D.validate_capabilities(dict(capabilities, stock_torch={}))


def test_reaudit_preserves_producer_and_rejects_changed_execution_dependencies(monkeypatch):
    files = {
        D.A.CONTROL_PREFIX + name: "a" * 64
        for name in ("audit_development.py", "qualify.py", "worker.py", "contract.py", "context.py")
    }
    files[D.A.METAL_LIBRARY] = "a" * 64
    producer = dict(root="/original", revision="a" * 40, files=files)
    auditor = dict(root="/corrected", revision="b" * 40, files=dict(files))
    for name in ("audit_development.py", "qualify.py"):
        auditor["files"][D.A.CONTROL_PREFIX + name] = "b" * 64
    calls = []

    def source_role(root, revision, *, control):
        calls.append((root, revision, control))
        return producer

    monkeypatch.setattr(D.A, "source_role", source_role)
    assert D.producer_role(auditor) is auditor
    assert D.producer_role(auditor, "/original", "a" * 40) is producer
    assert calls == [("/original", "a" * 40, True)]
    for root, revision in (("/original", None), (None, "a" * 40)):
        with pytest.raises(ValueError, match="Both producer"):
            D.producer_role(auditor, root, revision)
    for name in ("worker.py", "contract.py", "context.py", D.A.METAL_LIBRARY):
        key = name if name == D.A.METAL_LIBRARY else D.A.CONTROL_PREFIX + name
        altered = dict(auditor, files=dict(auditor["files"], **{key: "c" * 64}))
        with pytest.raises(ValueError, match="unchanged worker"):
            D.producer_role(altered, "/original", "a" * 40)
