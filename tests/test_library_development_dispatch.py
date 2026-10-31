"""Development dispatch cannot grant formal stream access or select a stock arm."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_library_development_dispatch", ROOT / "integration/library-study-v1/worker.py"
)
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)


def test_development_route_rejects_formal_fields_before_runtime_activation(monkeypatch):
    job = dict(
        schema=W.SCHEMA,
        kind="native_metal_gates",
        roles={},
        capture={},
        width=32,
        native_assets={},
        raw_store="/raw",
        output="/output",
        parent_pid=123,
    )
    assert W.validate_job(job) is job
    for field, value in (("seed", 20261112), ("formal_eligible", True), ("admission", "passed")):
        with pytest.raises(ValueError, match="fields differ"):
            W.validate_job(dict(job, **{field: value}))
    with pytest.raises(ValueError, match="Unknown"):
        W.validate_job(dict(job, kind="formal_fit"))
    with pytest.raises(ValueError, match="Invalid"):
        W.validate_job(dict(job, width=True))
    repeat = dict(job, kind="repeat_training", engine="cpu_torch", operands={})
    assert W.validate_job(repeat) is repeat
    with pytest.raises(ValueError, match="selected engine"):
        W.validate_job(dict(repeat, engine="stock_torch"))
    with pytest.raises(ValueError, match="selected engine"):
        W.validate_job(dict(job, kind="checkpoint_tie", engine="mps"))
    assert W.CONTEXT.DEVELOPMENT_SEED == 20261103
    common = dict(job, kind="native_metal_common_state", engine="metal", trajectory={})
    assert W.validate_job(common) is common
    with pytest.raises(ValueError, match="selected engine"):
        W.validate_job(dict(common, engine="mps"))
    with pytest.raises(ValueError, match="fields differ"):
        W.validate_job({k: v for k, v in common.items() if k != "trajectory"})
    monkeypatch.setattr(W.C, "EXECUTION_SUPPORTED", False)
    with pytest.raises(ValueError, match="not implemented"):
        W.C.require_execution_support()
