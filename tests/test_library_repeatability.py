"""Mutation controls for retained raw repetitions and explicit factory isolation."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from transformer import gpu_owned_backend, gpu_owned_repeatability, gpu_repeatability, spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_repeatability_auditor",
    ROOT / "integration/trained-precision/audit_library_repeatability.py",
)
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)
NAMESPACE = (
    "trained-precision-development-v1",
    "maintained-library-repeatability-development-v1",
)


def retained_case():
    arrays = {"output": np.array([0.0, -0.0, 1.0], dtype=np.float32)}
    entries = [
        dict(
            execution=i,
            finite=True,
            exact={key: gpu_repeatability.exact_comparison(v, v) for key, v in arrays.items()},
        )
        for i in range(10)
    ]
    return arrays, dict(complete=True, exact_passed=True, accuracy_passed=True, executions=entries)


def test_independent_auditor_requires_raw_bytes_for_all_ten_repetitions():
    arrays, case = retained_case()
    AUDITOR.check_repetitions(case, arrays)
    mutated = {"output": arrays["output"].copy()}
    mutated["output"][1] = 0.0
    with pytest.raises(ValueError, match="digest differs"):
        AUDITOR.check_repetitions(case, mutated)
    for i in range(10):
        broken = copy.deepcopy(case)
        broken["executions"][i]["exact"]["output"]["actual"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="digest differs"):
            AUDITOR.check_repetitions(broken, arrays)
    with pytest.raises(ValueError, match="ten-execution"):
        AUDITOR.check_repetitions(dict(case, executions=case["executions"][:-1]), arrays)


@pytest.mark.parametrize("explicit", [False, True])
def test_repeatability_selects_factory_without_mutating_stock_module(
    monkeypatch, tmp_path, explicit
):
    from transformer import reference

    state = dict(
        parameters=np.zeros(2, np.float32),
        m=np.zeros(2, np.float32),
        v=np.zeros(2, np.float32),
        step=0,
    )
    backend = Mock()
    backend.export_state.return_value = state
    backend.capabilities.return_value = {}
    selected = Mock(return_value=backend)
    ordinary = Mock(return_value=backend)
    monkeypatch.setattr(gpu_owned_backend, "create_runtime_backend", ordinary)
    monkeypatch.setattr(gpu_owned_repeatability, "STAGES", ())
    monkeypatch.setattr(gpu_owned_repeatability, "check_storage", lambda *a, **k: {})
    monkeypatch.setattr(gpu_repeatability, "require_environment", lambda *a: None, raising=False)
    monkeypatch.setattr(
        reference,
        "Transformer",
        lambda **k: SimpleNamespace(flat_parameters=lambda: state["parameters"]),
    )
    monkeypatch.setattr("transformer.gpu_environment.require_environment", lambda *a: None)
    authority = SimpleNamespace(
        verify=Mock(), inventory_json='{"files":{}}', receipt_sha256="0" * 64, root=tmp_path
    )
    options = dict(backend_factory=selected, namespace=NAMESPACE) if explicit else {}
    result = gpu_owned_repeatability.run_worker(
        "mps", spec.WIDTH, tmp_path, authority=authority, **options
    )
    (selected if explicit else ordinary).assert_called_once()
    (ordinary if explicit else selected).assert_not_called()
    backend.close.assert_called_once()
    assert result["protocol"] == (NAMESPACE[0] if explicit else "gpu-v4-owned-five-engine-v1")
    assert gpu_owned_backend.create_runtime_backend is ordinary


def test_candidate_factory_cannot_claim_historical_owned_namespace(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_owned_repeatability, "check_storage", lambda *a, **k: {})
    factory = Mock()
    with pytest.raises(ValueError, match="separate development namespace"):
        gpu_owned_repeatability.run_worker(
            "mps",
            spec.WIDTH,
            tmp_path,
            authority=SimpleNamespace(verify=Mock()),
            backend_factory=factory,
            namespace=("gpu-v4-owned-five-engine-v1", "gpu-v4-owned-baseline-repeatability-v1"),
        )
    factory.assert_not_called()
