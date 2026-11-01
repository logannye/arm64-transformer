"""The repaired host binding cannot replace frozen model or shader code."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

from transformer import gpu_metal

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_selected_metal_host", ROOT / "integration/library-study-v1/metal_host.py"
)
H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H)
SPEC = importlib.util.spec_from_file_location(
    "test_selected_metal_contract", ROOT / "integration/library-study-v1/contract.py"
)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def test_host_path_selection_preserves_constructor_and_all_numerical_globals(tmp_path):
    original = gpu_metal.Backend.__init__
    namespace = original.__globals__.copy()
    selected = H.constructor(gpu_metal, tmp_path)
    assert selected.__code__ is original.__code__
    assert selected.__defaults__ == original.__defaults__
    assert selected.__kwdefaults__ == original.__kwdefaults__
    assert selected.__closure__ is original.__closure__
    assert original.__globals__ == namespace
    changed = {k for k in namespace if selected.__globals__[k] is not namespace[k]}
    assert changed == {"ROOT", "LIBRARY", "_library"}
    loader = selected.__globals__["_library"]
    assert loader.__code__ is gpu_metal._library.__code__
    assert {k for k in namespace if loader.__globals__[k] is not namespace[k]} == {"LIBRARY"}
    assert selected.__globals__["ROOT"] == tmp_path
    assert loader.__globals__["LIBRARY"] == selected.__globals__["LIBRARY"] == tmp_path / H.LIBRARY


@pytest.mark.parametrize("changed", ["host", "binary"])
def test_host_and_binary_must_both_match_selected_bytes(tmp_path, monkeypatch, changed):
    host, binary = tmp_path / H.HOST_SOURCE, tmp_path / H.LIBRARY
    binary.parent.mkdir(parents=True)
    host.write_bytes(b"host source")
    binary.write_bytes(b"host binary")
    expected = {
        H.HOST_SOURCE: hashlib.sha256(host.read_bytes()).hexdigest(),
        H.LIBRARY: hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(H, "HOST_SHA256", expected[H.HOST_SOURCE])
    monkeypatch.setattr(H, "BINARY_SHA256", expected[H.LIBRARY])
    assert H.bind_assets(tmp_path, C) == expected
    (host if changed == "host" else binary).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Unselected repaired Metal host asset"):
        H.bind_assets(tmp_path, C)
