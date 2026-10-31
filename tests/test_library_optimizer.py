"""Prospective optimizer state and raw checkpoint auditor checks."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from transformer import spec
from transformer.reference import Transformer

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, relative, package=False):
    path = ROOT / relative
    specification = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


WORKER = module_at("tested_library_optimizer", "integration/trained-precision/library_optimizer.py")
AUDIT = module_at(
    "tested_library_optimizer_audit", "integration/trained-precision/audit_library_optimizer.py"
)


def test_cpu_optimizer_restores_parameters_both_moments_and_step_exactly(monkeypatch):
    api = module_at(
        "optimizer_prototype_api", "integration/precision-library-v1/torch_api.py", True
    )
    monkeypatch.setitem(sys.modules, "torch.nn.precise_fp32", api)
    initial = Transformer(seed=20261103).flat_parameters()
    first, second = WORKER.CPUTraining(initial), WORKER.CPUTraining(initial)
    tokens = np.arange(10, dtype="i4").reshape(2, 5)
    first.train_step(tokens, (tokens + 1) % 128)
    state = first.export_state()
    assert state["step"] == 1
    assert all(np.any(state[key] != 0) for key in ("m", "v"))
    WORKER.validate_state(np, state, spec.PARAM_COUNT)
    second.load_state(state)
    assert WORKER.state_identity(second.export_state()) == WORKER.state_identity(state)
    loss_a = first.train_step(tokens + 13, (tokens + 14) % 128)
    loss_b = second.train_step(tokens + 13, (tokens + 14) % 128)
    assert np.float64(loss_a).tobytes() == np.float64(loss_b).tobytes()
    assert WORKER.state_identity(first.export_state()) == WORKER.state_identity(
        second.export_state()
    )
    for model in (first.model, second.model):
        assert all(p.dtype == torch.float32 for p in model.parameters())


def test_independent_raw_checkpoint_audit_detects_replaced_and_nonfinite_moments(tmp_path):
    count = 7
    state = dict(
        parameters=np.arange(count, dtype="f4"),
        m=np.ones(count, "f4"),
        v=np.full(count, 2, "f4"),
        step=32,
    )
    bound = WORKER.save_state(tmp_path, state)
    assert AUDIT.raw_state(bound, count) == WORKER.state_identity(state)
    path = Path(bound["files"]["m"]["path"])
    path.write_bytes(np.zeros(count, "f4").tobytes())
    with pytest.raises(ValueError, match="File binding changed"):
        AUDIT.raw_state(bound, count)
    bound["files"]["m"] = AUDIT.SUITE.binding(path)
    assert AUDIT.raw_state(bound, count) != WORKER.state_identity(state)
    for bad, message in ((-1, "Negative"), (np.nan, "Nonfinite")):
        path = Path(bound["files"]["v"]["path"])
        path.write_bytes(np.full(count, bad, "f4").tobytes())
        bound["files"]["v"] = AUDIT.SUITE.binding(path)
        with pytest.raises(ValueError, match=message):
            AUDIT.raw_state(bound, count)


@pytest.mark.parametrize("change", ["dtype", "shape", "negative", "step", "missing"])
def test_worker_rejects_malformed_optimizer_states(change):
    state = dict(parameters=np.zeros(7, "f4"), m=np.zeros(7, "f4"), v=np.zeros(7, "f4"), step=0)
    if change == "dtype":
        state["m"] = state["m"].astype("f8")
    elif change == "shape":
        state["parameters"] = state["parameters"][:-1]
    elif change == "negative":
        state["v"][0] = -1
    elif change == "step":
        state["step"] = True
    else:
        del state["m"]
    with pytest.raises(ValueError):
        WORKER.validate_state(np, state, 7)
