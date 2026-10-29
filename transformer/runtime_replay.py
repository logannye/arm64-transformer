"""Independent smoke-training control loops; no timing or production training entry point.

Native arithmetic is already checked against independent references. This replay
checks its standalone orchestration: schedule, accumulation, actual batch size,
updates, validation, selection, and saved states. Every native update additionally
uses a fresh same-input PyTorch Adam comparison before advancing the native state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from transformer import spec
from transformer.prepare import load_fixture
from transformer.reference import Transformer


def _mean_gradient(gradient, actual_count):
    """Independent FP32 mean, including the final incomplete batch."""
    gradient *= np.float32(1.0) / np.float32(actual_count)


def _finite(*values):
    return all(np.isfinite(np.asarray(value)).all() for value in values)


def _same_gradient_adam(native, checks, params, gradient, moment_m, moment_v, step):
    tag = f"runtime/replay/native/step-{step}/adam"
    point = torch.tensor(params, requires_grad=True)
    optimizer = torch.optim.Adam(
        [point],
        lr=spec.LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        foreach=False,
        fused=False,
    )
    optimizer.state[point] = {
        "step": torch.tensor(float(step - 1), dtype=torch.float32),
        "exp_avg": torch.tensor(moment_m),
        "exp_avg_sq": torch.tensor(moment_v),
    }
    point.grad = torch.tensor(gradient)
    optimizer.step()
    # Advance with the same native operation as the executable, avoiding a
    # separate reference trajectory whose rounding could obscure control bugs.
    native.call("adam", params, gradient, moment_m, moment_v, step, spec.PARAM_COUNT)
    state = optimizer.state[point]
    checks.require(tag + "/step", int(state["step"].item()) == step)
    checks.close(tag + "/parameters", params, point, "optimizer")
    checks.close(tag + "/m", moment_m, state["exp_avg"], "optimizer")
    checks.close(tag + "/v", moment_v, state["exp_avg_sq"], "optimizer")


def _native_validation(native, params, rows):
    from transformer.validate import workspace_tensor

    total = 0.0
    dlogits = np.empty((spec.CONTEXT, spec.VOCAB), np.float32)
    for row in rows:
        workspace = native.forward(params, np.ascontiguousarray(row[:-1]))
        loss = native.call(
            "cross_entropy",
            workspace_tensor(workspace, "logits"),
            np.ascontiguousarray(row[1:]),
            dlogits,
            spec.CONTEXT,
            spec.VOCAB,
        )
        total += float(loss)
    return total / len(rows)


def _replay_native(native, checks, data):
    config = data["config"]
    params = data["initial"].copy()
    moment_m, moment_v = np.zeros_like(params), np.zeros_like(params)
    best, best_loss, best_epoch = None, float("inf"), 0
    history, batch_sizes, visited = [], [], []
    step = 0
    for epoch in range(config["epochs"]):
        total = 0.0
        schedule = data["schedule"][epoch]
        for start in range(0, len(schedule), config["batch"]):
            indices = schedule[start : start + config["batch"]]
            gradient = np.zeros_like(params)
            for index in indices:
                row = data["train"][index]
                tokens, targets = (np.ascontiguousarray(part) for part in (row[:-1], row[1:]))
                workspace = native.forward(params, tokens)
                loss, gradient = native.backward(params, tokens, targets, workspace, gradient)
                total += float(loss)
                visited.append(int(index))
            _mean_gradient(gradient, len(indices))
            batch_sizes.append(len(indices))
            step += 1
            tag = f"runtime/replay/native/step-{step}"
            checks.require(tag + "/gradient_finite", _finite(gradient, total))
            _same_gradient_adam(native, checks, params, gradient, moment_m, moment_v, step)
            checks.require(
                tag + "/state_finite",
                _finite(params, moment_m, moment_v) and np.all(moment_v >= 0),
            )
        validation = _native_validation(native, params, data["validation"])
        history.append((total / config["train_sequences"], validation))
        if validation < best_loss:
            best, best_loss, best_epoch = params.copy(), validation, epoch + 1
    return {
        "best.f32": best,
        "last.f32": params,
        "adam_m.f32": moment_m,
        "adam_v.f32": moment_v,
        "history": np.asarray(history, np.float64),
        "best_epoch": best_epoch,
        "steps": step,
        "batch_sizes": batch_sizes,
        "scheduled_rows": visited,
    }


def _flat_state(model, optimizer, key):
    return (
        torch.cat(
            [
                optimizer.state[getattr(model, name)][key].detach().reshape(-1)
                for name in spec.PARAM_SHAPES
            ]
        )
        .numpy()
        .copy()
    )


def _replay_python(checks, data):
    config = data["config"]
    model = Transformer(config["seed"]).load_flat(data["initial"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=spec.LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        foreach=False,
        fused=False,
    )
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(0.0, dtype=torch.float32),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    rows = {name: torch.tensor(data[name].astype(np.int64)) for name in ("train", "validation")}
    best, best_loss, best_epoch = None, float("inf"), 0
    history, batch_sizes, visited = [], [], []
    step = 0
    for epoch in range(config["epochs"]):
        total = 0.0
        schedule = data["schedule"][epoch]
        for start in range(0, len(schedule), config["batch"]):
            indices = schedule[start : start + config["batch"]]
            selected = rows["train"][torch.tensor(indices.astype(np.int64))]
            optimizer.zero_grad(set_to_none=True)
            logits = model(selected[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, spec.VOCAB), selected[:, 1:].reshape(-1))
            loss.backward()
            step += 1
            tag = f"runtime/replay/python/step-{step}"
            checks.require(tag + "/gradient_finite", _finite(model.flat_gradients(), loss.item()))
            optimizer.step()
            m, v = (_flat_state(model, optimizer, key) for key in ("exp_avg", "exp_avg_sq"))
            checks.require(
                tag + "/state_finite", _finite(model.flat_parameters(), m, v) and np.all(v >= 0)
            )
            total += float(loss.detach()) * len(indices)
            batch_sizes.append(len(indices))
            visited.extend(int(index) for index in indices)
        validation = 0.0
        with torch.no_grad():
            for start in range(0, len(rows["validation"]), config["batch"]):
                selected = rows["validation"][start : start + config["batch"]]
                logits = model(selected[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, spec.VOCAB), selected[:, 1:].reshape(-1))
                validation += float(loss) * len(selected)
        validation /= len(rows["validation"])
        history.append((total / config["train_sequences"], validation))
        if validation < best_loss:
            best, best_loss, best_epoch = model.flat_parameters(), validation, epoch + 1
    return {
        "best.f32": best,
        "last.f32": model.flat_parameters(),
        "adam_m.f32": _flat_state(model, optimizer, "exp_avg"),
        "adam_v.f32": _flat_state(model, optimizer, "exp_avg_sq"),
        "history": np.asarray(history, np.float64),
        "best_epoch": best_epoch,
        "steps": step,
        "batch_sizes": batch_sizes,
        "scheduled_rows": visited,
    }


def _check_saved(checks, owner, replay, data, directory):
    from transformer.audit import read_exact

    tag = "runtime/replay/" + owner
    run = Path(directory) / owner
    config = data["config"]
    expected_batches = [
        min(config["batch"], config["train_sequences"] - start)
        for _ in range(config["epochs"])
        for start in range(0, config["train_sequences"], config["batch"])
    ]
    checks.require(
        tag + "/batch_sizes", replay["batch_sizes"] == expected_batches, expected_batches
    )
    checks.require(
        tag + "/scheduled_rows",
        replay["scheduled_rows"] == data["schedule"].reshape(-1).tolist(),
        replay["scheduled_rows"],
    )
    artifacts = ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32")
    checks.require(
        tag + "/state_finite",
        _finite(*(replay[name] for name in artifacts), replay["history"])
        and np.all(replay["adam_v.f32"] >= 0),
    )
    for name in artifacts:
        saved = read_exact(run / name, "<f4", (spec.PARAM_COUNT,))
        checks.close(tag + "/" + name, replay[name], saved, "trajectory")
        checks.require(
            tag + "/" + name + "_exact",
            np.array_equal(replay[name].view(np.uint32), saved.view(np.uint32)),
        )
    history = read_exact(run / "history.f64", "<f8", (config["epochs"], 3))[:, :2]
    checks.close(tag + "/history", replay["history"], history, "trajectory")
    checks.require(tag + "/history_exact", np.array_equal(replay["history"], history))
    performance = read_exact(run / "performance.f64", "<f8", (8,))
    checks.require(tag + "/steps", replay["steps"] == len(expected_batches) == performance[6])
    checks.require(
        tag + "/best_epoch",
        replay["best_epoch"] == performance[5] == 1 + replay["history"][:, 1].argmin(),
    )


def check_runtime_updates(native, checks, fixture: Path, directory: Path):
    """Require both standalone runs to reproduce their complete four-update smoke history."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = load_fixture(fixture)
    config = data["config"]
    # Bound this audit to its declared smoke workload, never the measured campaign.
    keys = ("epochs", "batch", "train_sequences", "validation_sequences", "test_sequences")
    if tuple(config[key] for key in keys) != (2, 3, 5, 2, 2):
        raise ValueError(
            "Runtime replay requires the declared two-epoch partial-batch smoke fixture"
        )
    native_replay = _replay_native(native, checks, data)
    _check_saved(checks, "native", native_replay, data, directory)
    del native_replay
    python_replay = _replay_python(checks, data)
    _check_saved(checks, "python", python_replay, data, directory)
    checks.require("runtime/replay/complete", True)
