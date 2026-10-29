"""Raw-array producers for prospective, explicitly frozen GPU admission.

Import and inventory inspection never instantiate a backend or open a random
stream. The reserved stream is available only through ``produce_gate`` after
the frozen binding and baseline-repeatability preconditions are verified.
Unsupported checks remain missing handlers; they cannot become implicit passes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ADMISSION_SEED = 20261104
_CONTEXTS = {}
UNSUPPORTED = set()


def producer_inventory():
    from transformer.gpu_study_admission import gate_inventory

    families = {
        "attention",
        "layernorm_qkv",
        "full",
        "common_state",
        "adam",
        "trajectory",
        "independent_fp64",
        "primitive",
        "framework_attention",
        "framework_layernorm_qkv",
    }
    controls = {
        "causal_prefix",
        "sequence_isolation",
        "repeat_embedding_gradient",
        "repeat_forward",
        "repeat_training_same_process",
        "repeat_training_fresh_process",
        "warmup_restore",
        "partial_batch",
        "tail_bounds",
        "input_immutability",
        "checkpoint_first_tie",
        "checkpoint_replay",
        "finite_state",
        "malformed_input",
        "output_collision",
        "pending_work_cleanup",
        "device_execution",
        "state_restore",
        "runtime_failure",
        "guards",
        "barriers",
        "borrowed_result_lifetime",
        "borrowed_batch_lifetime",
        "finite_flag",
    }
    return [
        gate["id"]
        for gate in gate_inventory()
        if gate["id"] not in UNSUPPORTED
        and (
            gate["id"].split("/")[2] in families
            or (gate["id"].split("/")[2] == "control" and gate["id"].split("/")[3] in controls)
        )
    ]


def missing_handlers():
    from transformer.gpu_study_admission import gate_inventory

    missing = {gate["id"] for gate in gate_inventory()} - set(producer_inventory())
    return {name: "No executable handler is registered" for name in sorted(missing)}


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)


def _npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **values)


def _read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _flatten(state):
    return np.concatenate(
        [np.asarray(state[key], np.float64).reshape(-1) for key in ("parameters", "m", "v", "step")]
    )


def _exact(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def precision_fixtures(width, family, seed):
    """Reuse existing generator bytecode with a private RNG, never mutate v1 globals."""
    if seed == ADMISSION_SEED:
        raise ValueError("Reserved admission stream requires a verified frozen context")
    return _private_fixtures(width, family, seed)


def _private_fixtures(width, family, seed):
    from transformer import gpu_precision

    function = {
        "attention": gpu_precision.attention_stress_cases,
        "layernorm_qkv": gpu_precision.layernorm_qkv_stress_cases,
    }[family]
    private = dict(function.__globals__)
    private["_rng"] = lambda w, split, group, case: np.random.default_rng(
        np.random.SeedSequence([seed, w, group, case])
    )
    copied = types.FunctionType(
        function.__code__, private, function.__name__, function.__defaults__, function.__closure__
    )
    copied.__kwdefaults__ = function.__kwdefaults__
    return list(copied(width, split="development"))


def numpy_model(initial, tokens, targets, *, backward=True):
    """Independent FP64 model and full analytic backward on exact stored operands."""
    from transformer import spec

    parameters = {
        name: np.asarray(initial[start : start + count], np.float64).reshape(
            spec.PARAM_SHAPES[name]
        )
        for name, (start, count) in spec.PARAMS.items()
    }
    gradients = {name: np.zeros_like(value) for name, value in parameters.items()}
    batch, length = tokens.shape
    cache = {}

    def linear(x, prefix):
        cache[prefix] = x
        return x @ parameters[prefix + "weight"].T + parameters[prefix + "bias"]

    def linear_back(dy, prefix):
        x = cache[prefix]
        gradients[prefix + "weight"] = dy.reshape(-1, dy.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        gradients[prefix + "bias"] = dy.sum(axis=tuple(range(dy.ndim - 1)))
        return dy @ parameters[prefix + "weight"]

    def norm(x, i):
        centered = x - x.mean(-1, keepdims=True)
        inv = 1 / np.sqrt((centered * centered).mean(-1, keepdims=True) + spec.EPS)
        normalized = centered * inv
        cache[f"norm{i}"] = normalized, inv
        return normalized * parameters[f"ln{i}_weight"] + parameters[f"ln{i}_bias"]

    def norm_back(dy, i):
        normalized, inv = cache[f"norm{i}"]
        gradients[f"ln{i}_weight"] = (dy * normalized).sum(axis=(0, 1))
        gradients[f"ln{i}_bias"] = dy.sum(axis=(0, 1))
        dn = dy * parameters[f"ln{i}_weight"]
        dc = inv * (dn - normalized * (dn * normalized).mean(-1, keepdims=True))
        return dc - dc.mean(-1, keepdims=True)

    x = parameters["token"][tokens] + parameters["position"][:length]
    qkv = linear(norm(x, 1), "qkv_")
    q, k, v = [
        part.reshape(batch, length, 4, spec.WIDTH // 4).transpose(0, 2, 1, 3)
        for part in np.split(qkv, 3, axis=-1)
    ]
    scale = 1 / np.sqrt(spec.WIDTH // 4)
    scores = (q @ k.swapaxes(-1, -2)) * scale
    causal = np.tril(np.ones((length, length), bool))
    scores = np.where(causal, scores, -np.inf)
    probabilities = np.exp(scores - scores.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    attention = (probabilities @ v).transpose(0, 2, 1, 3).reshape(x.shape)
    r1 = x + linear(attention, "attention_")
    hidden = linear(norm(r1, 2), "ff1_")
    r2 = r1 + linear(np.maximum(hidden, 0), "ff2_")
    logits = linear(norm(r2, 3), "head_")
    maximum = logits.max(-1, keepdims=True)
    exp = np.exp(logits - maximum)
    denominator = exp.sum(-1, keepdims=True)
    chosen = np.take_along_axis(logits, targets[..., None], axis=-1)
    loss = np.mean(np.log(denominator) + maximum - chosen)
    result = {"logits": logits, "loss": np.asarray(loss)}
    if not backward:
        return result
    dl = exp / denominator
    flat = dl.reshape(-1, 128)
    flat[np.arange(batch * length), targets.reshape(-1)] -= 1
    dl /= batch * length
    dr2 = norm_back(linear_back(dl, "head_"), 3)
    dh = linear_back(dr2, "ff2_") * (hidden > 0)
    dr1 = dr2 + norm_back(linear_back(dh, "ff1_"), 2)
    da = (
        linear_back(dr1, "attention_")
        .reshape(batch, length, 4, spec.WIDTH // 4)
        .transpose(0, 2, 1, 3)
    )
    dp = da @ v.swapaxes(-1, -2)
    dv = probabilities.swapaxes(-1, -2) @ da
    ds = probabilities * (dp - (dp * probabilities).sum(-1, keepdims=True)) * scale
    ds = np.where(causal, ds, 0)
    dq, dk = ds @ k, ds.swapaxes(-1, -2) @ q
    dqkv = np.concatenate(
        [value.transpose(0, 2, 1, 3).reshape(batch, length, spec.WIDTH) for value in (dq, dk, dv)],
        axis=-1,
    )
    dx = dr1 + norm_back(linear_back(dqkv, "qkv_"), 1)
    np.add.at(gradients["token"], tokens.reshape(-1), dx.reshape(-1, spec.WIDTH))
    gradients["position"][:length] = dx.sum(0)
    result["gradient"] = np.concatenate([gradients[name].reshape(-1) for name in spec.PARAM_SHAPES])
    return result


def adam_reference(state, gradient, count):
    result = {
        key: np.asarray(value).copy() if key != "step" else int(value)
        for key, value in state.items()
    }
    g = np.asarray(gradient, np.float64)
    for _ in range(count):
        step = result["step"] + 1
        m = (0.9 * result["m"].astype(np.float64) + 0.1 * g).astype(np.float32)
        v = (0.999 * result["v"].astype(np.float64) + 0.001 * g * g).astype(np.float32)
        update = (
            0.001
            * (m.astype(np.float64) / (1 - 0.9**step))
            / (np.sqrt(v.astype(np.float64) / (1 - 0.999**step)) + 1e-8)
        )
        result = {
            "parameters": (result["parameters"].astype(np.float64) - update).astype(np.float32),
            "m": m,
            "v": v,
            "step": step,
        }
    return result


class Context:
    def __init__(self, manifest, roots, output, width):
        from transformer import spec
        from transformer.gpu_study import require, validate_manifest, verify_bindings

        validate_manifest(manifest)
        require(
            manifest["phase"] == "frozen" and manifest["admission"] is None,
            "Producer requires a frozen pre-admission contract",
        )
        verify_bindings(manifest, roots)
        from transformer.gpu_study_admission import verify_baseline_repeatability

        verify_baseline_repeatability(manifest, roots)
        # The control plane includes authenticated, complete standard-repeatability
        # evidence in these bindings; absence must never grant seed access.
        require(
            "baseline" in roots and "baseline" in manifest["bindings"],
            "Baseline repeatability must be authenticated before admission",
        )
        require(
            width == spec.WIDTH and os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(width),
            "Admission width must be selected before process imports",
        )
        self.manifest, self.roots, self.output, self.width = manifest, roots, Path(output), width
        self.backends, self.cache, self.counter = {}, {}, 0
        from transformer.reference import Transformer

        initial = Transformer(seed=ADMISSION_SEED).flat_parameters()
        self.initial = {
            "parameters": initial,
            "m": np.zeros_like(initial),
            "v": np.zeros_like(initial),
            "step": 0,
        }
        self.rng = np.random.default_rng(np.random.SeedSequence([ADMISSION_SEED, width, 99]))
        self.tokens = self.rng.integers(0, 128, (16, 32), dtype=np.int32)
        self.tokens[:, ::3] = 65
        self.targets = self.rng.integers(0, 128, (16, 32), dtype=np.int32)
        _npz(
            self.path("initial-operands"),
            {**self.initial, "tokens": self.tokens, "targets": self.targets},
        )

    def path(self, label):
        self.counter += 1
        return self.output / "operands" / f"width-{self.width}" / f"{self.counter:06}-{label}.npz"

    def backend(self, engine):
        if engine not in self.backends:
            self.backends[engine] = importlib.import_module(f"transformer.gpu_{engine}").Backend(
                self.initial["parameters"]
            )
        return self.backends[engine]

    def cpu(self, engine, mode, state, **values):
        from transformer.gpu_study_cpu_admission import run_cpu_operation

        operands, output = self.path("cpu-input"), self.path("cpu-output")
        _npz(operands, {**state, **values, "mode": np.asarray(mode)})
        run_cpu_operation(engine, self.manifest, self.roots, operands, output, width=self.width)
        return _read_npz(output)

    def model(self, engine, state=None, tokens=None, targets=None):
        state = self.initial if state is None else state
        tokens = self.tokens[:3] if tokens is None else tokens
        targets = self.targets[: len(tokens)] if targets is None else targets
        if engine.startswith("cpu_"):
            return self.cpu(engine, "forward_gradient", state, tokens=tokens, targets=targets)
        backend = self.backend(engine)
        backend.load_state({**state, "step": int(state["step"])})
        batch = backend.upload_inputs(tokens, targets)
        logits = backend.to_numpy(backend.forward(batch)).copy()
        loss, gradient = backend.loss_and_grad(batch)
        return {
            "logits": logits,
            "loss": backend.to_numpy(loss).copy(),
            "gradient": backend.to_numpy(gradient).copy(),
        }

    def advance(self, engine, state=None, count=3, tokens=None, targets=None):
        state = self.initial if state is None else state
        tokens = self.tokens[:3] if tokens is None else tokens
        targets = self.targets[: len(tokens)] if targets is None else targets
        if engine.startswith("cpu_"):
            result = dict(state)
            for _ in range(count):
                result = self.cpu(
                    engine,
                    "advance",
                    result,
                    tokens=tokens,
                    targets=targets,
                    step=np.asarray(int(result["step"]) + 1, np.int64),
                )
                result = {key: result[key] for key in self.initial}
            return result
        backend = self.backend(engine)
        backend.load_state({**state, "step": int(state["step"])})
        batch = backend.upload_inputs(tokens, targets)
        for _ in range(count):
            backend.to_numpy(backend.train_step(batch))
            if not backend.check_finite():
                raise RuntimeError("Admission training produced nonfinite state")
        return backend.export_state()

    def adam(self, engine, gradient, count):
        state = self.initial
        if engine.startswith("cpu_"):
            for _ in range(count):
                state = self.cpu(
                    engine,
                    "adam",
                    state,
                    gradient=gradient,
                    step=np.asarray(int(state["step"]) + 1, np.int64),
                )
            return state
        backend = self.backend(engine)
        backend.load_state(state)
        for _ in range(count):
            backend.apply_gradient(gradient)
        return backend.export_state()


def _capture(backend, result):
    return {name: backend.to_numpy(value).copy() for name, value in result.items()}


def _fixture_family(context, family):
    key = ("fixture", family)
    if key not in context.cache:
        context.cache[key] = {}
        for case in _private_fixtures(context.width, family, ADMISSION_SEED):
            context.cache[key][case["name"]] = case
            operands = {
                name: value for name, value in case.items() if isinstance(value, np.ndarray)
            }
            _npz(context.path(f"{family}-{case['name']}"), operands)
    return context.cache[key]


def _precision(context, family, name):
    from transformer.gpu_precision import layernorm_qkv_reference
    from transformer.precision import attention_reference

    case = _fixture_family(context, family)[name]
    backend = context.backend("metal")
    if family == "attention":
        observed = _capture(backend, backend.attention(case["qkv"], case["dy"]))
        expected = attention_reference(case["qkv"], 4, case["dy"])
        saved = attention_reference(
            case["qkv"], 4, case["dy"], probabilities=observed["probabilities"]
        )
        return {
            "output": (observed["output"], expected["output"]),
            "probabilities": (observed["probabilities"], expected["probabilities"]),
            "dqkv": (observed["dqkv"], expected["dqkv"]),
            "same_saved_probabilities_gradient": (observed["dqkv"], saved["dqkv"]),
            "probability_mass": (
                observed["probabilities"].sum(-1, dtype=np.float64),
                np.ones(observed["probabilities"].shape[:-1]),
            ),
        }
    values = [case[key] for key in ("x", "gamma", "beta", "weight", "bias", "dqkv")]
    expected = layernorm_qkv_reference(*values, epsilon=case["epsilon"])
    norm = _capture(backend, backend.layernorm(*values[:3]))
    linear = _capture(backend, backend.linear(norm["output"], *values[3:]))
    backward = _capture(backend, backend.layernorm(*values[:3], linear["dx"]))
    result = {
        "normalized": (norm["norm"], expected["normalized"]),
        "inverse_std": (norm["inv"], expected["inverse_std"].reshape(-1)),
        "ln": (norm["output"], expected["ln"]),
        "qkv": (linear["output"], expected["qkv"]),
        "dln": (linear["dx"], expected["dln"]),
    }
    result.update({key: (backward[key], expected[key]) for key in ("dx", "dgamma", "dbeta")})
    result.update({key: (linear[key], expected[key]) for key in ("dweight", "dbias")})
    rounded, weight, bias, dy = [
        np.asarray(value, np.float64) for value in (norm["output"], *values[3:])
    ]
    incoming = linear["dx"].astype(np.float64)
    dn = incoming * case["gamma"].astype(np.float64)
    dc = expected["inverse_std"] * (
        dn - expected["normalized"] * (dn * expected["normalized"]).mean(-1, keepdims=True)
    )
    result.update(
        same_operands_output=(linear["output"], rounded @ weight.T + bias),
        same_operands_dln=(linear["dx"], dy @ weight),
        same_operands_dweight=(linear["dweight"], dy.T @ rounded),
        same_operands_dbias=(linear["dbias"], dy.sum(0)),
        same_incoming_dx=(backward["dx"], dc - dc.mean(-1, keepdims=True)),
        same_incoming_dgamma=(backward["dgamma"], (incoming * expected["normalized"]).sum(0)),
        same_incoming_dbeta=(backward["dbeta"], incoming.sum(0)),
    )
    return result


def _bools(values):
    actual = np.asarray(values, np.int32).reshape(-1)
    return {"observations": (actual, np.ones_like(actual))}


def _framework_precision(context, engine, family, name):
    from transformer.gpu_precision import layernorm_qkv_reference
    from transformer.precision import attention_reference

    family = family.removeprefix("framework_")
    case = _fixture_family(context, family)[name]
    backend = context.backend(engine)
    if family == "attention":
        outputs = backend.attention_probe(case["qkv"], case["dy"])
        expected = attention_reference(case["qkv"], 4, case["dy"])
        # Expanded diagnostic probabilities are not fused-kernel saved buffers.
        return {
            name: (backend.to_numpy(outputs[name]).copy(), expected[name])
            for name in ("output", "dqkv")
        }
    arrays = [case[key] for key in ("x", "gamma", "beta", "weight", "bias", "dqkv")]
    outputs = backend.layernorm_qkv_probe(*arrays)
    expected = layernorm_qkv_reference(*arrays, epsilon=case["epsilon"])
    return {
        name: (backend.to_numpy(outputs[name]).copy(), expected[name])
        for name in ("ln", "qkv", "dx", "dgamma", "dbeta", "dweight", "dbias")
    }


def _reject(operation):
    try:
        operation()
    except (ValueError, RuntimeError, TypeError, subprocess.CalledProcessError):
        return True
    return False


def _fresh_training(context, engine):
    if engine.startswith("cpu_"):
        # The CPU bridge creates a fresh frozen-source process for every step.
        state = context.advance(engine, count=6)
        return {**state, "logits": context.model(engine, state=state)["logits"]}
    operands, output = context.path("fresh-training-input"), context.path("fresh-training-output")
    _npz(
        operands, {**context.initial, "tokens": context.tokens[:3], "targets": context.targets[:3]}
    )
    command = [
        sys.executable,
        "-m",
        "transformer.gpu_admission_checks",
        "repeat-worker",
        "--engine",
        engine,
        "--input",
        str(operands.resolve()),
        "--output",
        str(output.resolve()),
    ]
    subprocess.run(
        command,
        cwd=context.roots["source"],
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return _read_npz(output)


def _control(context, engine, name):
    initial = context.initial
    if name in ("causal_prefix", "sequence_isolation"):
        tokens = context.tokens[:3].copy()
        before = context.model(engine, tokens=tokens)["logits"]
        if name == "causal_prefix":
            tokens[:, -1] = (tokens[:, -1] + 1) % 128
            after = context.model(engine, tokens=tokens)["logits"]
            return {"observations": (after[:, :-1], before[:, :-1])}
        tokens[1:] = (tokens[1:] + 17) % 128
        after = context.model(engine, tokens=tokens)["logits"]
        return {"observations": (after[:1], before[:1])}
    if name in ("repeat_forward", "repeat_embedding_gradient"):
        from transformer import spec

        first, second = context.model(engine), context.model(engine)
        key = "logits" if name == "repeat_forward" else "gradient"
        if key == "gradient":
            start, count = spec.PARAMS["token"]
            return {
                "observations": (
                    second[key][start : start + count],
                    first[key][start : start + count],
                )
            }
        return {"observations": (second[key], first[key])}
    if name in ("repeat_training_same_process", "repeat_training_fresh_process"):
        if engine.startswith("cpu_") and name == "repeat_training_same_process":
            replay = context.cpu(
                engine,
                "repeat_training",
                initial,
                tokens=context.tokens[:3],
                targets=context.targets[:3],
            )
            return _bools(
                [
                    _exact(replay["first_" + key], replay["second_" + key])
                    for key in (*initial, "logits")
                ]
            )

        def same_process():
            state = context.advance(engine, count=6)
            return {**state, "logits": context.model(engine, state=state)["logits"]}

        operation = (
            same_process
            if name.endswith("same_process")
            else lambda: _fresh_training(context, engine)
        )
        first, second = operation(), operation()
        return _bools([_exact(first[key], second[key]) for key in (*initial, "logits")])
    if name == "checkpoint_first_tie":
        if engine.startswith("cpu_"):
            from transformer.gpu_study_cpu_admission import run_cpu_checkpoint_tie

            return {
                "observations": run_cpu_checkpoint_tie(
                    engine,
                    context.manifest,
                    context.roots,
                    context.path("cpu-first-tie").with_suffix(""),
                    width=context.width,
                )
            }
        from transformer.gpu_study_worker import checkpoint_improves

        best, selected = float("inf"), -1
        for index, loss in enumerate((4.0, 2.0, 2.0, 3.0)):
            if checkpoint_improves(loss, best):
                best, selected = loss, index
        return {"observations": (np.array([selected], np.int32), np.array([1], np.int32))}
    if name in ("state_restore", "checkpoint_replay", "warmup_restore"):
        checkpoint = context.advance(engine, count=3) if name == "checkpoint_replay" else initial
        if engine.startswith("cpu_"):
            before = context.model(engine, state=checkpoint)
            context.advance(engine, count=2)
            after = context.model(engine, state=checkpoint)
            return {"observations": (after["logits"], before["logits"])}
        backend = context.backend(engine)
        backend.load_state(checkpoint)
        batch = backend.upload_inputs(context.tokens[:3], context.targets[:3])
        before = backend.to_numpy(backend.forward(batch)).copy()
        if name == "warmup_restore":
            backend.warmup(batch, iterations=2)
        else:
            backend.to_numpy(backend.train_step(batch))
            backend.load_state(checkpoint)
        state = backend.export_state()
        after = backend.to_numpy(backend.forward(batch)).copy()
        return _bools(
            [_exact(state[key], checkpoint[key]) for key in checkpoint] + [_exact(after, before)]
        )
    if name in ("partial_batch", "tail_bounds"):
        # Frozen CPU executables expose T32 only. GPU lifecycle gates cover the
        # singleton and sequence tails separately from the B1/B3 partial batches.
        lengths = (
            (1, 3, 31, 32) if name == "tail_bounds" and not engine.startswith("cpu_") else (32,)
        )
        checks = []
        for batch in (1, 3):
            for length in lengths:
                actual = context.model(
                    engine,
                    tokens=context.tokens[:batch, :length],
                    targets=context.targets[:batch, :length],
                )
                checks.extend(
                    [
                        actual["logits"].shape == (batch, length, 128),
                        actual["gradient"].shape == initial["parameters"].shape,
                        all(np.isfinite(value).all() for value in actual.values()),
                    ]
                )
        return _bools(checks)
    if name == "finite_state":
        state = context.advance(engine, count=2)
        return _bools([np.isfinite(value).all() for value in state.values()])
    if name == "input_immutability":
        tokens = context.tokens[:3].copy()
        params = initial["parameters"].copy()
        context.model(engine, tokens=tokens)
        return _bools([_exact(tokens, context.tokens[:3]), _exact(params, initial["parameters"])])
    if name == "malformed_input":
        if engine.startswith("cpu_"):
            return _bools(
                [
                    _reject(lambda: context.model(engine, tokens=np.full((3, 32), 128, np.int32))),
                    _reject(
                        lambda: context.model(
                            engine,
                            tokens=np.zeros((17, 32), np.int32),
                            targets=np.zeros((17, 32), np.int32),
                        )
                    ),
                ]
            )
        backend = context.backend(engine)
        return _bools(
            [
                _reject(lambda value=value: backend.upload_inputs(value))
                for value in (
                    np.full((1, 32), 128, np.int32),
                    np.zeros((17, 32), np.int32),
                    np.zeros((1, 33), np.int32),
                    np.zeros((1, 32), np.int64),
                )
            ]
        )
    if name == "output_collision":
        path = context.path("collision")
        _npz(path, {"sentinel": np.array([17], np.int32)})
        before = path.read_bytes()
        try:
            _npz(path, {"sentinel": np.array([99], np.int32)})
        except FileExistsError:
            return _bools([path.read_bytes() == before])
        return _bools([False])
    if name == "device_execution":
        actual = context.model(engine)
        if engine.startswith("cpu_"):
            values = np.asarray([actual["backend_width"], actual["backend_kind"]], np.int64)
            return {
                "observations": (
                    values,
                    np.asarray([context.width, 1 if engine == "cpu_arm64" else 2], np.int64),
                )
            }
        backend = context.backend(engine)
        capabilities = backend.capabilities()
        if engine == "mps":
            return _bools(
                [str(backend.device) == "mps", capabilities["cpu_tensor_fallback"] is False]
            )
        if engine == "mlx":
            return _bools(
                [
                    str(importlib.import_module("mlx.core").default_device())
                    == str(importlib.import_module("mlx.core").gpu),
                    capabilities["cpu_tensor_fallback"] is False,
                ]
            )
        return _bools(
            [
                capabilities["framework_numerical_calls"] == 0,
                bool(capabilities["device"]),
                capabilities["unified_memory"],
            ]
        )
    if name in ("pending_work_cleanup", "runtime_failure"):
        if engine.startswith("cpu_"):
            if name == "runtime_failure":
                state = {**initial, "parameters": np.array([0], np.float32)}
                return _bools([_reject(lambda: context.model(engine, state=state))])
            return _bools([np.isfinite(context.model(engine)["logits"]).all()])
        backend = context.backend(engine)
        batch = backend.upload_inputs(context.tokens[:3], context.targets[:3])
        backend.forward(batch)  # Intentionally leave completion to close().
        backend.close()
        context.backends.pop(engine)
        if name == "runtime_failure":
            return _bools([_reject(lambda: backend.forward(batch))])
        return _bools([np.isfinite(context.model(engine)["logits"]).all()])
    backend = context.backend("metal")
    if name in ("guards", "barriers"):
        checks = []
        for m, n, k in ((1, 1, 1), (5, 19, 17), (31, 65, 33), (33, 129, 65)):
            x = context.rng.normal(0, 0.2, (m, k)).astype(np.float32)
            w = context.rng.normal(0, 0.2, (n, k)).astype(np.float32)
            bias = np.zeros(n, np.float32)
            values = _capture(backend, backend.linear(x, w, bias))
            _npz(
                context.path(f"{name}-tail-{m}-{n}-{k}"),
                {"x": x, "weight": w, "bias": bias, "output": values["output"]},
            )
            checks.extend(
                [
                    backend.check_guards(),
                    np.allclose(
                        values["output"],
                        x.astype(np.float64) @ w.astype(np.float64).T,
                        atol=2e-5,
                        rtol=2e-4,
                    ),
                ]
            )
        return _bools(checks)
    if name == "borrowed_result_lifetime":
        batch = backend.upload_inputs(context.tokens[:1])
        value = backend.forward(batch)
        backend.forward(batch)
        return _bools([_reject(lambda: backend.to_numpy(value))])
    if name == "borrowed_batch_lifetime":
        dataset = backend.upload_dataset(
            np.concatenate((context.tokens[:3], context.targets[:3, -1:]), 1)
        )
        indices = backend.upload_indices(np.arange(3, dtype=np.int32))
        first = backend.gather_batch(dataset, indices, 0, 1)
        backend.gather_batch(dataset, indices, 1, 1)
        return _bools([_reject(lambda: backend.forward(first))])
    if name == "finite_flag":
        backend.load_state(initial)
        batch = backend.upload_inputs(context.tokens[:1], context.targets[:1])
        backend.forward(batch)
        normal = backend.check_finite()
        extreme = {**initial, "parameters": np.full_like(initial["parameters"], 1e20)}
        backend.load_state(extreme)
        backend.forward(batch)
        rejected = not backend.check_finite()
        backend.load_state(initial)
        return _bools([normal, rejected])
    raise NotImplementedError(f"No implemented control: {engine}/{name}")


def _primitive(context, name):
    backend = context.backend("metal")
    if name in ("forward", "dx", "dw", "db"):
        x = context.rng.normal(0, 0.2, (33, 17)).astype(np.float32)
        w = context.rng.normal(0, 0.2, (65, 17)).astype(np.float32)
        bias = context.rng.normal(0, 0.1, 65).astype(np.float32)
        dy = context.rng.normal(0, 0.2, (33, 65)).astype(np.float32)
        _npz(context.path("primitive-" + name), {"x": x, "weight": w, "bias": bias, "dy": dy})
        values = _capture(backend, backend.linear(x, w, bias, dy))
        x, w, bias, dy = [value.astype(np.float64) for value in (x, w, bias, dy)]
        actual_key, expected = {
            "forward": ("output", x @ w.T + bias),
            "dx": ("dx", dy @ w),
            "dw": ("dweight", dy.T @ x),
            "db": ("dbias", dy.sum(0)),
        }[name]
        return {"values": (values[actual_key], expected)}
    backend.load_state(context.initial)
    tokens, targets = context.tokens[:3], context.targets[:3]
    batch = backend.upload_inputs(tokens, targets)
    loss, gradient = backend.loss_and_grad(batch)
    observed_loss, observed_gradient = (
        backend.to_numpy(loss).copy(),
        backend.to_numpy(gradient).copy(),
    )
    trace = _capture(backend, backend.trace(batch, include_gradients=True))
    if name == "relu":
        return {"values": (trace["relu"], np.maximum(trace["ff1"], np.float32(0)))}
    if name == "cross_entropy":
        logits = trace["logits"].astype(np.float64)
        maximum = logits.max(-1)
        expected = (
            np.log(np.exp(logits - maximum[..., None]).sum(-1))
            + maximum
            - np.take_along_axis(logits, targets[..., None], -1)[..., 0]
        ).mean()
        return {"values": (observed_loss, np.asarray(expected))}
    if name == "embedding":
        from transformer import spec

        expected = np.zeros((128, context.width), np.float64)
        np.add.at(
            expected, tokens.reshape(-1), trace["dx"].reshape(-1, context.width).astype(np.float64)
        )
        start, count = spec.PARAMS["token"]
        return {
            "values": (observed_gradient[start : start + count].reshape(expected.shape), expected)
        }
    raise NotImplementedError(name)


def _produce(context, engine, family, suffix):
    from transformer import spec

    if family in ("attention", "layernorm_qkv"):
        return _precision(context, family, suffix)
    if family in ("framework_attention", "framework_layernorm_qkv"):
        return _framework_precision(context, engine, family, suffix)
    if family == "control":
        return _control(context, engine, suffix)
    if family == "primitive":
        return _primitive(context, suffix)
    if family == "independent_fp64":
        from transformer.gpu_repeatability import _torch_reference

        actual = numpy_model(context.initial["parameters"], context.tokens[:3], context.targets[:3])
        expected = _torch_reference(
            context.initial["parameters"], context.tokens[:3], context.targets[:3], backward=True
        )
        return {name: (actual[name], expected[name]) for name in ("logits", "gradient")}
    if family == "full":
        batch = int(suffix.split("T")[0][1:])
        actual = context.model(
            engine, tokens=context.tokens[:batch], targets=context.targets[:batch]
        )
        expected = numpy_model(
            context.initial["parameters"], context.tokens[:batch], context.targets[:batch]
        )
        result = {name: (actual[name], expected[name]) for name in ("logits", "loss")}
        for name, (start, count) in spec.PARAMS.items():
            result[f"gradient/{name}"] = (
                actual["gradient"][start : start + count],
                expected["gradient"][start : start + count],
            )
        return result
    if family == "common_state":
        state = context.initial
        if suffix == "evolved":
            if "common_evolved" not in context.cache:
                context.cache["common_evolved"] = context.advance("cpu_torch", count=3)
            state = context.cache["common_evolved"]
        actual = context.model(engine, state)
        expected = numpy_model(state["parameters"], context.tokens[:3], context.targets[:3])
        return {name: (actual[name], expected[name]) for name in ("logits", "gradient")}
    if family == "adam":
        source, count = suffix.split("/")
        source, count = source.removeprefix("from-"), int(count.removeprefix("step-"))
        key = "gradient-from-" + source
        if key not in context.cache:
            context.cache[key] = context.model(source)["gradient"]
        gradient = context.cache[key]
        actual = context.adam(engine, gradient, count)
        expected = adam_reference(context.initial, gradient, count)
        return {
            name: (np.asarray(actual[name]), np.asarray(expected[name])) for name in context.initial
        }
    if family == "trajectory":
        actual, expected = context.initial, context.initial
        records = {
            side: {key: [] for key in ("parameters", "m", "v", "loss", "gradient")}
            for side in ("actual", "expected")
        }
        for step in range(3):
            tokens, targets = context.tokens[step : step + 1], context.targets[step : step + 1]
            for side, implementation, state in (
                ("actual", engine, actual),
                ("expected", "cpu_torch", expected),
            ):
                probe = context.model(implementation, state, tokens, targets)
                updated = context.advance(implementation, state, 1, tokens, targets)
                for key in records[side]:
                    records[side][key].append(
                        np.asarray(probe[key] if key in probe else updated[key]).copy()
                    )
                if side == "actual":
                    actual = updated
                else:
                    expected = updated
        return {
            key: (np.stack(records["actual"][key]), np.stack(records["expected"][key]))
            for key in records["actual"]
        }
    raise NotImplementedError(f"No execution handler for {engine}/{family}/{suffix}")


def produce_gate(specification, manifest, roots, output_directory):
    from transformer.gpu_study import digest, identity, require, write_new
    from transformer.gpu_study_admission import binding_identity, evaluate_member

    identifier = specification["id"]
    require(identifier in producer_inventory(), f"Unimplemented admission handler: {identifier}")
    pieces = identifier.split("/")
    engine, width, family = pieces[:3]
    width = int(width)
    suffix = "/".join(pieces[3:])
    key = (identity(manifest), str(Path(output_directory).resolve()), width)
    if key not in _CONTEXTS:
        _CONTEXTS[key] = Context(manifest, roots, output_directory, width)
    try:
        pairs = _produce(_CONTEXTS[key], engine, family, suffix)
    except Exception as error:
        if (
            specification.get("family") != "diagnostic_precision"
            or specification.get("blocking") is not False
        ):
            raise
        path = Path(output_directory) / "errors" / identifier / "runtime.json"
        evidence = {
            "gate": identifier,
            "binding_sha256": binding_identity(manifest),
            "exception_type": type(error).__name__,
            "message": str(error) or repr(error),
        }
        # Failed lazy VJP graphs may poison a framework's pending queue. Preserve
        # the error and retire that instance before independent later checks.
        failed_backend = getattr(_CONTEXTS[key], "backends", {}).pop(engine, None)
        if failed_backend is not None:
            try:
                failed_backend.close()
            except Exception as cleanup_error:
                evidence["cleanup_error"] = repr(cleanup_error)
            evidence["next_case_requires_fresh_backend"] = True
        write_new(path, evidence)
        return {
            "id": identifier,
            "status": "runtime_error",
            "passed": False,
            "members": {},
            "error_artifact": {
                "path": path.relative_to(output_directory).as_posix(),
                "sha256": digest(path),
            },
        }
    require(set(pairs) == set(specification["members"]), "Producer/member inventory mismatch")
    root = Path(output_directory)
    record = {"id": identifier, "members": {}}
    if family == "control":
        scopes = {
            "runtime_failure": (
                "Rejected runtime/API operation propagates an error; no GPU fault injection"
            ),
            "barriers": (
                "Guarded-tail numerical executions and buffer sentinels; "
                "not a proof of race freedom"
            ),
            "output_collision": (
                "Actual admission evidence writer rejects an existing output artifact"
            ),
        }
        if suffix in scopes:
            record["execution_scope"] = scopes[suffix]
        if engine.startswith("cpu_") and suffix in (
            "state_restore",
            "warmup_restore",
            "checkpoint_replay",
        ):
            record["execution_scope"] = (
                "Explicit supplied-state replay through authenticated frozen CPU arithmetic; "
                "the standalone CPU runner has no device-state export/load API"
            )
    if family.startswith("framework_"):
        record["execution_scope"] = (
            "Framework fused primitive outputs and gradients; reconstructed probabilities "
            "and normalization statistics are excluded"
        )
    for name, (actual, expected) in pairs.items():
        members = {}
        for label, value in (("actual", actual), ("expected", expected)):
            path = root / "arrays" / identifier / name / f"{label}.npy"
            _save(path, value)
            members[label] = {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}
        record["members"][name] = members
    record["passed"] = all(
        evaluate_member(root, record["members"][name], tolerance)
        for name, tolerance in specification["members"].items()
    )
    return record


def close_contexts():
    """Synchronize/release cached backends after a width worker, even on failure."""
    errors = []
    for context in _CONTEXTS.values():
        for backend in context.backends.values():
            try:
                backend.close()
            except Exception as error:
                errors.append(error)
    _CONTEXTS.clear()
    if errors:
        raise RuntimeError(f"Admission backend cleanup failed: {errors!r}")


def repeat_worker(engine, input_file, output_file):
    """Replay supplied arrays; this child never generates random fixtures."""
    values = _read_npz(input_file)
    runtime = importlib.import_module(f"transformer.gpu_{engine}")
    backend = runtime.Backend(values["parameters"])
    try:
        backend.load_state(
            {
                key: int(values[key]) if key == "step" else values[key]
                for key in ("parameters", "m", "v", "step")
            }
        )
        batch = backend.upload_inputs(values["tokens"], values["targets"])
        for _ in range(6):
            backend.to_numpy(backend.train_step(batch))
        result = backend.export_state()
        result["logits"] = backend.to_numpy(backend.forward(batch)).copy()
        _npz(Path(output_file), result)
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "repeat-worker"))
    parser.add_argument("--engine", choices=("mps", "mlx", "metal"))
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.action == "inventory":
        print(
            json.dumps(
                {"implemented": producer_inventory(), "missing": missing_handlers()}, indent=2
            )
        )
    else:
        if not all((args.engine, args.input, args.output)):
            parser.error("Repeat worker needs engine/input/output")
        repeat_worker(args.engine, args.input, args.output)


if __name__ == "__main__":
    main()
