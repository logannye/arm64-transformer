"""Raw common-state binding and independent accuracy boundary regressions."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "integration/trained-precision" / file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKER = load("test_common_worker", "library_common_state.py")
AUDIT = load("test_common_auditor", "audit_library_common_state.py")


def trajectory_at(path):
    state = {"parameters": np.arange(7, dtype="f4"), "m": np.zeros(7, "f4"), "v": np.ones(7, "f4")}
    files = {}
    for key, value in state.items():
        file = path / (key + ".f32")
        value.tofile(file)
        files[key] = AUDIT.SUITE.binding(file)
    states = [dict(step=i) for i in range(65)]
    states[8].update({key: bound["sha256"] for key, bound in files.items()})
    return dict(saved=[dict(step=8, files=files)], states=states), state


def test_common_checkpoint_requires_complete_bound_state_and_trajectory_digest(tmp_path):
    trajectory, state = trajectory_at(tmp_path)
    actual = AUDIT.checkpoint(trajectory, 8, 7)
    assert all(actual[k].tobytes() == v.tobytes() for k, v in state.items())
    broken = copy.deepcopy(trajectory)
    broken["states"][8]["m"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint binding"):
        AUDIT.checkpoint(broken, 8, 7)
    broken = copy.deepcopy(trajectory)
    broken["saved"].append(broken["saved"][0])
    with pytest.raises(ValueError, match="Incomplete common checkpoint"):
        AUDIT.checkpoint(broken, 8, 7)
    file = Path(trajectory["saved"][0]["files"]["v"]["path"])
    np.full(7, -1, "f4").tofile(file)
    trajectory["saved"][0]["files"]["v"] = AUDIT.SUITE.binding(file)
    trajectory["states"][8]["v"] = trajectory["saved"][0]["files"]["v"]["sha256"]
    with pytest.raises(ValueError, match="Negative second moment"):
        AUDIT.checkpoint(trajectory, 8, 7)
    with pytest.raises(ValueError, match="Negative second moment"):
        WORKER.checkpoint(np, trajectory, 8, 7)


def test_common_accuracy_requires_each_coordinate_and_byte_exact_integer_step():
    gold = np.array([0.0, 1.0], "f8")
    assert AUDIT.accuracy(np.array([1.9e-5, 1.0002]), gold, "forward")["passed"]
    for bad in (np.array([2.1e-5, 1.0]), np.array([np.nan, 1.0]), np.array([0.0, np.inf])):
        assert not AUDIT.accuracy(bad, gold, "forward")["passed"]
    assert AUDIT.accuracy(np.asarray(74, "i8"), np.asarray(74, "i8"), "exact")["passed"]
    assert not AUDIT.accuracy(np.asarray(74, "f8"), np.asarray(74, "i8"), "exact")["passed"]
    with pytest.raises(ValueError, match="shape"):
        AUDIT.accuracy(np.zeros(1), gold, "gradient")


def test_common_inventory_retains_each_exposed_state_batch_and_optimizer_count():
    assert WORKER.cases() == [
        f"model/step-{s}/b{b}-t32" for s in (0, 8, 64) for b in (1, 3, 16)
    ] + [f"adam/step-{s}/count-{n}" for s in (0, 64) for n in (1, 2, 10)]
