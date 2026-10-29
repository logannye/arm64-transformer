"""Control-loop fault injection against an independent four-update oracle."""

import numpy as np
import pytest
import torch

from transformer import runtime_replay as replay
from transformer import validate
from transformer.validate import Checks


class ToyNative:
    """Constant sequence gradients make schedule and denominator effects explicit."""

    def __init__(self, *, overwrite=False, broken_adam=False):
        self.overwrite = overwrite
        self.broken_adam = broken_adam

    def forward(self, params, tokens):
        return params.copy()

    def backward(self, params, tokens, targets, workspace, gradients):
        value = np.float32(int(tokens[0]) + 1)
        contribution = value * np.array([1, 2, -1, 0.5], np.float32)
        if self.overwrite:
            gradients[:] = contribution
        else:
            gradients += contribution
        return float(value), gradients

    def call(self, name, *args):
        if name == "cross_entropy":
            return 2.0
        assert name == "adam"
        params, gradient, m, v, step, count = args
        assert count == 4
        point = torch.tensor(params, requires_grad=True)
        optimizer = torch.optim.Adam([point], lr=0.001, foreach=False, fused=False)
        optimizer.state[point] = {
            "step": torch.tensor(float(step - 1)),
            "exp_avg": torch.tensor(m),
            "exp_avg_sq": torch.tensor(v),
        }
        point.grad = torch.tensor(gradient)
        optimizer.step()
        params[:] = point.detach().numpy()
        m[:] = optimizer.state[point]["exp_avg"].numpy()
        v[:] = optimizer.state[point]["exp_avg_sq"].numpy()
        if self.broken_adam:
            params[0] += 0.01


@pytest.fixture
def oracle(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(replay.spec, "PARAM_COUNT", 4)
    monkeypatch.setattr(replay.spec, "CONTEXT", 2)
    monkeypatch.setattr(replay.spec, "VOCAB", 8)
    monkeypatch.setattr(validate, "workspace_tensor", lambda workspace, name: workspace)
    initial = np.array([0.2, -0.3, 0.1, 0.4], np.float32)
    data = {
        "config": {
            "epochs": 2,
            "batch": 3,
            "train_sequences": 5,
            "validation_sequences": 2,
            "test_sequences": 2,
            "seed": 1729,
        },
        "initial": initial,
        "train": np.array([[index, 0, 1] for index in range(5)], np.int32),
        "validation": np.array([[0, 1, 2], [1, 2, 3]], np.int32),
        "schedule": np.array([[2, 0, 4, 1, 3], [3, 1, 0, 4, 2]], np.int32),
    }
    # The declared batches have sums9,6,7,8 and lengths3,2,3,2. This
    # external oracle never calls the replay's batching or normalization code.
    batch_means = [np.float32(3), np.float32(3), np.float32(7) * np.float32(1 / 3), np.float32(4)]
    point = torch.tensor(initial, requires_grad=True)
    optimizer = torch.optim.Adam([point], lr=0.001, foreach=False, fused=False)
    best = None
    for step, mean in enumerate(batch_means, 1):
        point.grad = torch.tensor(mean * np.array([1, 2, -1, 0.5], np.float32))
        optimizer.step()
        if step == 2:
            best = point.detach().numpy().copy()
    state = optimizer.state[point]
    run = tmp_path / "native"
    run.mkdir()
    for name, values in {
        "best.f32": best,
        "last.f32": point.detach().numpy(),
        "adam_m.f32": state["exp_avg"].numpy(),
        "adam_v.f32": state["exp_avg_sq"].numpy(),
    }.items():
        np.asarray(values, dtype="<f4").tofile(run / name)
    np.array([[3, 2, 0.1], [3, 2, 0.1]], "<f8").tofile(run / "history.f64")
    np.array([1, 1, 1, 1, 1, 1, 4, 1], "<f8").tofile(run / "performance.f64")
    return data, tmp_path


def test_replay_matches_independent_schedule_partial_batch_and_first_tie(oracle):
    data, directory = oracle
    checks = Checks()
    result = replay._replay_native(ToyNative(), checks, data)
    replay._check_saved(checks, "native", result, data, directory)
    assert checks.passed, [record for record in checks.records if not record["passed"]]
    assert result["batch_sizes"] == [3, 2, 3, 2]
    assert result["scheduled_rows"] == [2, 0, 4, 1, 3, 3, 1, 0, 4, 2]
    assert result["best_epoch"] == 1


def test_wrong_partial_batch_normalization_is_blocking(oracle, monkeypatch):
    data, directory = oracle

    def wrong_configured_batch(gradient, actual_count):
        gradient *= np.float32(1.0) / np.float32(3)

    monkeypatch.setattr(replay, "_mean_gradient", wrong_configured_batch)
    checks = Checks()
    result = replay._replay_native(ToyNative(), checks, data)
    replay._check_saved(checks, "native", result, data, directory)
    assert not checks.passed
    assert any(
        not record["passed"] and record["name"] == "runtime/replay/native/adam_m.f32"
        for record in checks.records
    )
    assert all(record["passed"] for record in checks.records if "/adam/" in record["name"])


@pytest.mark.parametrize("fault", ["overwrite_gradient", "schedule", "adam"])
def test_replay_rejects_accumulation_schedule_and_update_faults(oracle, fault):
    data, directory = oracle
    wrong_data = dict(data)
    if fault == "schedule":
        wrong_data["schedule"] = data["schedule"][:, ::-1].copy()
    native = ToyNative(overwrite=fault == "overwrite_gradient", broken_adam=fault == "adam")
    checks = Checks()
    result = replay._replay_native(native, checks, wrong_data)
    replay._check_saved(checks, "native", result, data, directory)
    assert not checks.passed
    if fault == "adam":
        assert any(
            not record["passed"] and "/adam/parameters" in record["name"]
            for record in checks.records
        )


def test_runtime_replay_rejects_measured_campaign_fixture(monkeypatch):
    monkeypatch.setattr(
        replay,
        "load_fixture",
        lambda path: {
            "config": {
                "epochs": 8,
                "batch": 16,
                "train_sequences": 2048,
                "validation_sequences": 256,
                "test_sequences": 256,
            },
        },
    )
    with pytest.raises(ValueError, match="smoke fixture"):
        replay.check_runtime_updates(None, Checks(), None, None)
