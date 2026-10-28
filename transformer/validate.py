"""Independent numerical and executable audits; never builds the native code.

This module uses PyTorch primitives as mathematical references and NumPy for
artifact checks. Native training is only invoked as a standalone child process.
The tolerances below are fixed before the first native validation run.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transformer import spec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / "native/build/libtransformer.dylib"
DEFAULT_EXECUTABLE = ROOT / "native/build/transformer-arm64"
SOURCE_FILES = (
    "native/kernels.S",
    "native/model.S",
    "native/runtime.S",
    "native/layout.inc",
    "transformer/spec.py",
    "transformer/reference.py",
    "transformer/prepare.py",
    "transformer/python_train.py",
    "transformer/validate.py",
)

# Fixed admission limits, not selected from observed discrepancies.
TOLERANCES = {
    "forward": {"atol": 2e-5, "rtol": 2e-4},
    "gradient": {"atol": 3e-5, "rtol": 3e-3},
    "optimizer": {"atol": 3e-6, "rtol": 3e-4},
    "math": {"atol": 2e-7, "rtol": 2e-6},
    "trajectory": {"atol": 2e-4, "rtol": 2e-3},
    "finite_difference": {"atol": 2e-5, "rtol": 5e-3},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arr(value, dtype=np.float32):
    return np.ascontiguousarray(value, dtype=dtype)


def tensor(value, *, grad=False, dtype=torch.float32):
    return torch.tensor(value, dtype=dtype, requires_grad=grad)


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class Native:
    """Typed FFI into the independently built handwritten assembly library."""

    def __init__(self, path: Path | str = DEFAULT_LIBRARY):
        self.path = Path(path).resolve()
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Native validation requires macOS ARM64")
        if not self.path.is_file():
            raise FileNotFoundError(f"Native library missing: {self.path}; build it explicitly")
        self.lib = ct.CDLL(str(self.path))
        p, n = ct.c_void_p, ct.c_uint64
        signatures = {
            "linear_forward": [p] * 5,
            "linear_backward": [p] * 7,
            "ln_forward": [p] * 6 + [n, n],
            "ln_backward": [p] * 8,
            "attention_forward": [p] * 4,
            "attention_backward": [p] * 5,
            "relu_forward": [p, p, n],
            "relu_backward": [p, p, p, n],
            "cross_entropy": [p, p, p, n, n],
            "adam": [p] * 4 + [n, n],
            "forward": [p] * 3,
            "backward": [p] * 5,
            "scale": [p, n, ct.c_float],
            "finite": [p, n],
            "exp": [ct.c_float],
            "log": [ct.c_float],
        }
        for name, arguments in signatures.items():
            fn = getattr(self.lib, "tf_" + name)
            fn.argtypes = arguments
            if name in {"cross_entropy", "backward", "exp", "log"}:
                fn.restype = ct.c_float
            elif name == "finite":
                fn.restype = ct.c_uint64
            else:
                fn.restype = None

    def call(self, name, *args):
        converted = []
        for value in args:
            if isinstance(value, np.ndarray):
                if not value.flags.c_contiguous:
                    raise ValueError("Native arguments must be contiguous")
                converted.append(ct.c_void_p(value.ctypes.data))
            else:
                converted.append(value)
        return getattr(self.lib, "tf_" + name)(*converted)

    def forward(self, params, tokens):
        workspace = np.full(spec.WORK_COUNT, np.nan, dtype=np.float32)
        self.call("forward", params, tokens, workspace)
        return workspace

    def backward(self, params, tokens, targets, workspace, gradients=None):
        if gradients is None:
            gradients = np.zeros(spec.PARAM_COUNT, dtype=np.float32)
        loss = self.call("backward", params, tokens, targets, workspace, gradients)
        return loss, gradients


def workspace_tensor(workspace, name):
    start, count = spec.WORK[name]
    return workspace[start : start + count].reshape(spec.WORK_SHAPES[name])


class Checks:
    def __init__(self):
        self.records = []

    def require(self, name, condition, detail=None):
        self.records.append({"name": name, "passed": bool(condition), "detail": detail})

    def close(self, name, actual, expected, tolerance="forward"):
        actual = as_numpy(actual)
        expected = as_numpy(expected)
        thresholds = TOLERANCES[tolerance]
        same_shape = actual.shape == expected.shape
        passed = same_shape and np.allclose(actual, expected, equal_nan=False, **thresholds)
        finite = np.isfinite(actual) & np.isfinite(expected) if same_shape else np.array([])
        if finite.any():
            errors = np.abs(actual[finite].astype(np.float64) - expected[finite])
            max_error = float(errors.max())
            rmse = float(np.sqrt(np.mean(errors**2)))
        else:
            max_error = rmse = None
        self.records.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape),
                "tolerance": tolerance,
                "max_absolute_error": max_error,
                "rmse": rmse,
            }
        )

    @property
    def passed(self):
        return bool(self.records) and all(record["passed"] for record in self.records)


def reference_forward(flat, tokens, *, trace=False):
    """Autograd reference written independently of the production Python model."""
    parameters = {
        name: flat[start : start + count].view(spec.PARAM_SHAPES[name])
        for name, (start, count) in spec.PARAMS.items()
    }
    x = F.embedding(tokens, parameters["token"]) + parameters["position"]
    values = {"x": x}

    def norm(inp, index):
        out = F.layer_norm(
            inp,
            (spec.WIDTH,),
            parameters[f"ln{index}_weight"],
            parameters[f"ln{index}_bias"],
            spec.EPS,
        )
        variance, mean = torch.var_mean(inp, dim=-1, correction=0, keepdim=True)
        inv = torch.rsqrt(variance + spec.EPS)
        values[f"norm{index}"] = (inp - mean) * inv
        values[f"inv{index}"] = inv.squeeze(-1)
        return out

    def linear(inp, name):
        return F.linear(inp, parameters[f"{name}_weight"], parameters[f"{name}_bias"])

    n1 = norm(x, 1)
    qkv = linear(n1, "qkv")
    q, k, v = [
        piece.reshape(spec.CONTEXT, spec.HEADS, -1).transpose(0, 1)
        for piece in qkv.chunk(3, dim=-1)
    ]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(spec.WIDTH // spec.HEADS)
    mask = torch.ones(spec.CONTEXT, spec.CONTEXT, dtype=torch.bool).tril()
    probs = scores.masked_fill(~mask, -torch.inf).softmax(-1)
    attention = (probs @ v).transpose(0, 1).reshape(spec.CONTEXT, spec.WIDTH)
    projected = linear(attention, "attention")
    r1 = x + projected
    n2 = norm(r1, 2)
    ff1 = linear(n2, "ff1")
    relu = F.relu(ff1)
    ff2 = linear(relu, "ff2")
    r2 = r1 + ff2
    n3 = norm(r2, 3)
    logits = linear(n3, "head")
    values.update(
        n1=n1,
        qkv=qkv,
        attention=attention,
        projected=projected,
        r1=r1,
        n2=n2,
        ff1=ff1,
        relu=relu,
        ff2=ff2,
        r2=r2,
        n3=n3,
        logits=logits,
        probs=probs,
    )
    return (logits, values) if trace else logits


def initial_parameters(rng):
    flat = rng.normal(0, 0.02, spec.PARAM_COUNT).astype(np.float32)
    for name, (start, count) in spec.PARAMS.items():
        if name.endswith("bias"):
            flat[start : start + count] = 0
        if name.startswith("ln") and name.endswith("weight"):
            flat[start : start + count] = 1
    return flat


def check_linear(native, checks, rng):
    for rows, width, output in ((1, 4, 8), (7, 32, 96), (32, 64, 32), (32, 32, 128)):
        tag = f"linear/{rows}x{width}x{output}"
        x = rng.normal(0, 0.3, (rows, width)).astype(np.float32)
        w = rng.normal(0, 0.2, (output, width)).astype(np.float32)
        b = rng.normal(0, 0.1, output).astype(np.float32)
        dy = rng.normal(0, 0.2, (rows, output)).astype(np.float32)
        y = np.full((rows, output), np.nan, np.float32)
        shape = arr([rows, width, output], np.int32)
        native.call("linear_forward", x, w, b, y, shape)
        tx, tw, tb = [tensor(value, grad=True) for value in (x, w, b)]
        expected = F.linear(tx, tw, tb)
        (expected * tensor(dy)).sum().backward()
        checks.close(tag + "/forward", y, expected)
        dx = np.full_like(x, np.nan)
        # Nonzero initial accumulators test the ADD versus overwrite contract.
        dw = np.full_like(w, 0.125)
        db = np.full_like(b, -0.125)
        native.call("linear_backward", x, w, dy, dx, dw, db, shape)
        checks.close(tag + "/dx", dx, tx.grad, "gradient")
        checks.close(tag + "/dw_add", dw, tw.grad + 0.125, "gradient")
        checks.close(tag + "/db_add", db, tb.grad - 0.125, "gradient")


def check_norm(native, checks, rng):
    for rows, width, constant in ((1, 4, False), (7, 32, False), (3, 64, True)):
        tag = f"layernorm/{rows}x{width}/constant={constant}"
        x = rng.normal(0, 0.4, (rows, width)).astype(np.float32)
        if constant:
            x.fill(0.25)
        gamma = rng.uniform(0.5, 1.5, width).astype(np.float32)
        beta = rng.normal(0, 0.1, width).astype(np.float32)
        dy = rng.normal(0, 0.2, x.shape).astype(np.float32)
        y, normalized, inv = np.empty_like(x), np.empty_like(x), np.empty(rows, np.float32)
        native.call("ln_forward", x, gamma, beta, y, normalized, inv, rows, width)
        tx, tg, tb = [tensor(value, grad=True) for value in (x, gamma, beta)]
        expected = F.layer_norm(tx, (width,), tg, tb, spec.EPS)
        (expected * tensor(dy)).sum().backward()
        checks.close(tag + "/forward", y, expected)
        variance, mean = torch.var_mean(tx, dim=-1, correction=0, keepdim=True)
        checks.close(tag + "/normalized", normalized, (tx - mean) / torch.sqrt(variance + spec.EPS))
        checks.close(tag + "/inv", inv, torch.rsqrt(variance + spec.EPS).squeeze(-1))
        dx = np.full_like(x, np.nan)
        dg, db = np.full(width, 0.125, np.float32), np.full(width, -0.125, np.float32)
        native.call(
            "ln_backward", dy, normalized, inv, gamma, dx, dg, db, arr([rows, width], np.int32)
        )
        checks.close(tag + "/dx", dx, tx.grad, "gradient")
        checks.close(tag + "/dgamma_add", dg, tg.grad + 0.125, "gradient")
        checks.close(tag + "/dbeta_add", db, tb.grad - 0.125, "gradient")


def check_attention(native, checks, rng):
    for tokens in (1, 7, spec.CONTEXT):
        width, heads = spec.WIDTH, spec.HEADS
        tag = f"attention/{tokens}x{width}x{heads}"
        qkv = rng.normal(0, 0.5, (tokens, 3 * width)).astype(np.float32)
        output = np.full((tokens, width), np.nan, np.float32)
        probs = np.full((heads, tokens, tokens), np.nan, np.float32)
        shape = arr([tokens, width, heads], np.int32)
        native.call("attention_forward", qkv, output, probs, shape)
        tqkv = tensor(qkv, grad=True)
        q, k, v = [
            part.reshape(tokens, heads, width // heads).transpose(0, 1)
            for part in tqkv.chunk(3, dim=-1)
        ]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // heads)
        mask = torch.ones(tokens, tokens, dtype=torch.bool).tril()
        tprob = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        tout = (tprob @ v).transpose(0, 1).reshape(tokens, width)
        checks.close(tag + "/forward", output, tout)
        checks.close(tag + "/probabilities", probs, tprob)
        checks.close(tag + "/probability_mass", probs.sum(-1), np.ones((heads, tokens)))
        checks.require(tag + "/masked_exact_zero", bool(np.all(probs[:, ~mask.numpy()] == 0)))
        dy = rng.normal(0, 0.3, output.shape).astype(np.float32)
        (tout * tensor(dy)).sum().backward()
        dqkv = np.full_like(qkv, np.nan)
        native.call("attention_backward", qkv, probs, dy, dqkv, shape)
        checks.close(tag + "/dqkv", dqkv, tqkv.grad, "gradient")
        if tokens > 1:
            changed = qkv.copy()
            changed[-1] += 2
            out2, probs2 = np.empty_like(output), np.empty_like(probs)
            native.call("attention_forward", changed, out2, probs2, shape)
            checks.require(tag + "/future_invariance", np.array_equal(output[:-1], out2[:-1]))


def check_elementwise(native, checks, rng):
    x = arr([-3, -0.1, 0, 0.1, 1, 4, -2, 0])
    dy = rng.normal(size=len(x)).astype(np.float32)
    out, dx = np.empty_like(x), np.empty_like(x)
    native.call("relu_forward", x, out, len(x))
    native.call("relu_backward", x, dy, dx, len(x))
    checks.close("relu/forward", out, np.maximum(x, 0))
    checks.close("relu/backward_zero_convention", dx, dy * (x > 0))
    exp_input = arr([-100, -80, -20, -1, -0.001, 0, 0.001, 1, 20, 80])
    exp_out = np.array([native.call("exp", float(x)) for x in exp_input])
    checks.close("math/exp", exp_out, np.exp(exp_input.astype(np.float64)), "math")
    log_input = arr(
        [np.nextafter(np.float32(0), np.float32(1)), 1e-20, 0.001, 0.5, 1, 2, 100, 1e20]
    )
    log_out = np.array([native.call("log", float(x)) for x in log_input])
    checks.close("math/log", log_out, np.log(log_input.astype(np.float64)), "math")
    checks.require("math/log_zero", native.call("log", 0.0) == -math.inf)
    checks.require("math/log_negative", math.isnan(native.call("log", -1.0)))
    checks.require("math/exp_large", native.call("exp", 1000.0) == math.inf)
    checks.require("math/exp_underflow", native.call("exp", -1000.0) == 0.0)
    finite = arr([0, -1, np.finfo(np.float32).max, np.finfo(np.float32).tiny])
    checks.require("finite/valid", native.call("finite", finite, len(finite)) == 1)
    for invalid in (np.nan, np.inf, -np.inf):
        value = finite.copy()
        value[2] = invalid
        checks.require(f"finite/reject_{invalid}", native.call("finite", value, len(value)) == 0)
    native.call("scale", finite, len(finite), 0.5)
    checks.close(
        "scale", finite, arr([0, -0.5, np.finfo(np.float32).max / 2, np.finfo(np.float32).tiny / 2])
    )


def check_loss(native, checks, rng):
    for rows, kind in ((1, "constant"), (5, "random"), (3, "extreme")):
        logits = rng.normal(size=(rows, spec.VOCAB)).astype(np.float32)
        if kind == "constant":
            logits.fill(0.125)
        if kind == "extreme":
            logits *= 500
        targets = rng.integers(0, spec.VOCAB, rows, dtype=np.int32)
        dlogits = np.full_like(logits, np.nan)
        loss = native.call("cross_entropy", logits, targets, dlogits, rows, spec.VOCAB)
        tlogits = tensor(logits, grad=True)
        expected = F.cross_entropy(tlogits, tensor(targets, dtype=torch.long))
        expected.backward()
        checks.close(f"loss/{kind}/value", loss, expected)
        checks.close(f"loss/{kind}/gradient", dlogits, tlogits.grad, "gradient")
        checks.close(f"loss/{kind}/gradient_mass", dlogits.sum(-1), np.zeros(rows))


def check_adam(native, checks, rng):
    params = rng.normal(0, 0.1, 128).astype(np.float32)
    tp = tensor(params, grad=True)
    optimizer = torch.optim.Adam([tp], lr=spec.LR, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    m, v = np.zeros_like(params), np.zeros_like(params)
    for step in range(1, 6):
        gradients = rng.normal(0, 0.1, len(params)).astype(np.float32)
        gradients[0] = 0
        tp.grad = tensor(gradients)
        optimizer.step()
        native.call("adam", params, gradients, m, v, step, len(params))
        checks.close(f"adam/{step}/weights", params, tp, "optimizer")
        checks.close(f"adam/{step}/m", m, optimizer.state[tp]["exp_avg"], "optimizer")
        checks.close(f"adam/{step}/v", v, optimizer.state[tp]["exp_avg_sq"], "optimizer")


def check_complete_model(native, checks, rng):
    params = initial_parameters(rng)
    tokens = rng.integers(0, spec.VOCAB, spec.CONTEXT, dtype=np.int32)
    tokens[::3] = 32  # Repeated indices must ADD their embedding gradients.
    targets = rng.integers(0, spec.VOCAB, spec.CONTEXT, dtype=np.int32)
    tp = tensor(params, grad=True)
    tt = tensor(tokens, dtype=torch.long)
    expected, traces = reference_forward(tp, tt, trace=True)
    workspace = native.forward(params, tokens)
    for name, expected_tensor in traces.items():
        checks.close("model/trace/" + name, workspace_tensor(workspace, name), expected_tensor)
    loss, gradients = native.backward(params, tokens, targets, workspace)
    expected_loss = F.cross_entropy(expected, tensor(targets, dtype=torch.long))
    expected_loss.backward()
    checks.close("model/loss", loss, expected_loss)
    for name, (start, count) in spec.PARAMS.items():
        checks.close(
            "model/gradient/" + name,
            gradients[start : start + count],
            tp.grad[start : start + count],
            "gradient",
        )
    _, doubled = native.backward(params, tokens, targets, workspace, gradients.copy())
    checks.close("model/gradient_accumulation", doubled, gradients * 2, "gradient")
    changed = tokens.copy()
    changed[16:] = (changed[16:] + 17) % spec.VOCAB
    changed_workspace = native.forward(params, changed)
    checks.require(
        "model/future_token_invariance",
        np.array_equal(
            workspace_tensor(workspace, "logits")[:16],
            workspace_tensor(changed_workspace, "logits")[:16],
        ),
    )

    # Independent double central differences, away from the ReLU boundary.
    double_parameters = tensor(params, dtype=torch.float64)
    with torch.no_grad():
        _, double_traces = reference_forward(double_parameters, tt, trace=True)
    margin = float(double_traces["ff1"].abs().min())
    checks.require("finite_difference/relu_nonzero_margin", margin > 1e-8, margin)
    indices = [
        spec.PARAMS[name][0] + min(3, spec.PARAMS[name][1] - 1)
        for name in (
            "position",
            "ln1_weight",
            "qkv_weight",
            "attention_weight",
            "ln2_bias",
            "ff1_weight",
            "ff2_weight",
            "ln3_weight",
            "head_weight",
        )
    ]
    # An actually used embedding row exercises embedding scatter as well.
    indices.append(spec.PARAMS["token"][0] + int(tokens[0]) * spec.WIDTH + 1)
    epsilon = min(1e-5, max(1e-7, margin * 0.01))
    for index in indices:
        losses = []
        signs_stable = True
        for sign in (-1, 1):
            shifted = double_parameters.clone()
            shifted[index] += sign * epsilon
            with torch.no_grad():
                logits, shifted_traces = reference_forward(shifted, tt, trace=True)
                signs_stable &= bool(
                    torch.equal(shifted_traces["ff1"] > 0, double_traces["ff1"] > 0)
                )
                losses.append(float(F.cross_entropy(logits, tensor(targets, dtype=torch.long))))
        checks.require(f"finite_difference/{index}/relu_signs_stable", signs_stable)
        derivative = (losses[1] - losses[0]) / (2 * epsilon)
        checks.close(
            f"finite_difference/{index}/gradient", gradients[index], derivative, "finite_difference"
        )

    # Complete repeated update trajectory, with independent native moment state.
    params = initial_parameters(rng)
    tp = tensor(params, grad=True)
    optimizer = torch.optim.Adam([tp], lr=spec.LR, foreach=False)
    m, v = np.zeros_like(params), np.zeros_like(params)
    for step in range(1, 4):
        tokens = rng.integers(0, spec.VOCAB, spec.CONTEXT, dtype=np.int32)
        targets = rng.integers(0, spec.VOCAB, spec.CONTEXT, dtype=np.int32)
        workspace = native.forward(params, tokens)
        native_loss, gradients = native.backward(params, tokens, targets, workspace)
        optimizer.zero_grad(set_to_none=True)
        reference_loss = F.cross_entropy(
            reference_forward(tp, tensor(tokens, dtype=torch.long)),
            tensor(targets, dtype=torch.long),
        )
        reference_loss.backward()
        checks.close(f"trajectory/{step}/loss", native_loss, reference_loss, "trajectory")
        for name, (start, count) in spec.PARAMS.items():
            checks.close(
                f"trajectory/{step}/gradient/{name}",
                gradients[start : start + count],
                tp.grad[start : start + count],
                "trajectory",
            )
        optimizer.step()
        native.call("adam", params, gradients, m, v, step, spec.PARAM_COUNT)
        checks.close(f"trajectory/{step}/parameters", params, tp, "trajectory")
        checks.close(f"trajectory/{step}/m", m, optimizer.state[tp]["exp_avg"], "trajectory")
        checks.close(f"trajectory/{step}/v", v, optimizer.state[tp]["exp_avg_sq"], "trajectory")


def numerical_checks(native, checks):
    torch.set_num_threads(1)
    rng = np.random.default_rng(1729)
    check_linear(native, checks, rng)
    check_norm(native, checks, rng)
    check_attention(native, checks, rng)
    check_elementwise(native, checks, rng)
    check_loss(native, checks, rng)
    check_adam(native, checks, rng)
    check_complete_model(native, checks, rng)


def run_child(command, *, timeout=180):
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    return {
        "command": [str(item) for item in command],
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "wall_seconds": time.perf_counter() - started,
    }


def runtime_checks(checks, directory, executable=DEFAULT_EXECUTABLE):
    from transformer.prepare import prepare_fixture

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "synthetic-smoke.txt"
    source.write_text(
        (
            "The quick brown fox jumps over the lazy dog.\n"
            "A tiny transformer reads one character at a time.\n"
        )
        * 120
    )
    fixture = directory / "fixture"
    prepare_fixture(
        source,
        fixture,
        seed=1729,
        epochs=2,
        batch=3,
        train_sequences=5,
        validation_sequences=2,
        test_sequences=2,
    )
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Native executable missing: {executable}")
    records = []
    for name, command in (
        ("python", [sys.executable, "-m", "transformer.python_train"]),
        ("native", [str(executable)]),
        ("native-repeat", [str(executable)]),
    ):
        result = run_child([*command, str(fixture), str(directory / name)])
        records.append(result)
        checks.require("runtime/" + name + "/exit", result["returncode"] == 0, result)
        if result["returncode"] != 0:
            raise RuntimeError(f"{name} smoke failed: {result}")
    for filename in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32", "test_logits.f32"):
        python_values = np.fromfile(directory / "python" / filename, dtype="<f4")
        native_values = np.fromfile(directory / "native" / filename, dtype="<f4")
        checks.close("runtime/matched/" + filename, native_values, python_values, "trajectory")
        checks.require(
            "runtime/repeat/" + filename,
            (directory / "native" / filename).read_bytes()
            == (directory / "native-repeat" / filename).read_bytes(),
        )
    python_history = np.fromfile(directory / "python/history.f64", "<f8").reshape(2, 3)
    native_history = np.fromfile(directory / "native/history.f64", "<f8").reshape(2, 3)
    checks.close("runtime/history", native_history[:, :2], python_history[:, :2], "trajectory")
    native_repeat = np.fromfile(directory / "native-repeat/history.f64", "<f8").reshape(2, 3)
    checks.require(
        "runtime/repeat/history", np.array_equal(native_history[:, :2], native_repeat[:, :2])
    )
    for name in ("python", "native"):
        performance = np.fromfile(directory / name / "performance.f64", "<f8")
        checks.require(f"runtime/{name}/performance_shape", performance.shape == (8,))
        checks.require(f"runtime/{name}/four_optimizer_steps", performance[6] == 4)
        checks.require(f"runtime/{name}/positive_timers", np.all(performance[[0, 1, 7]] > 0))
        checks.require(
            f"runtime/{name}/best_epoch", performance[5] == 1 + native_history[:, 1].argmin()
        )
        timings = np.fromfile(directory / name / "timings.f64", "<f8")
        checks.require(
            f"runtime/{name}/timings",
            timings.shape == (100,) and bool(np.all(np.isfinite(timings) & (timings > 0))),
        )

    # Preserve the output bytes across a rejected duplicate invocation.
    before = {
        path.name: sha256(path) for path in (directory / "native").iterdir() if path.is_file()
    }
    result = run_child([str(executable), str(fixture), str(directory / "native")])
    records.append(result)
    after = {path.name: sha256(path) for path in (directory / "native").iterdir() if path.is_file()}
    checks.require("runtime/reject_existing_output", result["returncode"] != 0 and before == after)

    corruptions = {
        "truncated_weights": ("initial.f32", lambda data: data[:-4]),
        "extra_weights": ("initial.f32", lambda data: data + b"\x00\x00\x00\x00"),
        "nonfinite_weights": ("initial.f32", lambda data: struct.pack("<f", math.nan) + data[4:]),
        "token_out_of_range": ("train.i32", lambda data: struct.pack("<i", 128) + data[4:]),
        "token_negative": ("train.i32", lambda data: struct.pack("<i", -1) + data[4:]),
        "truncated_validation": ("validation.i32", lambda data: data[:-4]),
        "invalid_schedule": ("schedule.i32", lambda data: struct.pack("<i", 5) + data[4:]),
        "duplicate_schedule": ("schedule.i32", lambda data: data[4:8] + data[4:]),
        "zero_batch": ("config.bin", lambda data: data[:12] + struct.pack("<I", 0) + data[16:]),
        "bad_magic": ("config.bin", lambda data: b"BAD!" + data[4:]),
    }
    for name, (filename, corrupt) in corruptions.items():
        bad = directory / ("bad-" + name)
        shutil.copytree(fixture, bad)
        path = bad / filename
        path.write_bytes(corrupt(path.read_bytes()))
        result = run_child([str(executable), str(bad), str(directory / ("rejected-" + name))])
        records.append(result)
        checks.require(
            "runtime/reject/" + name,
            result["returncode"] > 0,
            {"returncode": result["returncode"], "stderr": result["stderr"]},
        )
    (directory / "child-processes.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Audit output already exists; choose a fresh evidence path")
    if args.runtime and args.work_dir is None:
        parser.error("--runtime requires a fresh --work-dir")
    checks = Checks()
    error = None
    source_before = {
        name: sha256(ROOT / name) if (ROOT / name).is_file() else None for name in SOURCE_FILES
    }
    try:
        native = Native(args.library)
        numerical_checks(native, checks)
        if args.runtime:
            runtime_checks(checks, args.work_dir, args.executable)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        checks.require("audit/exception", False, error)
    source_after = {
        name: sha256(ROOT / name) if (ROOT / name).is_file() else None for name in SOURCE_FILES
    }
    checks.require("audit/source_files_present", all(source_after.values()))
    checks.require("audit/source_unchanged_during_checks", source_before == source_after)
    result = {
        "schema_version": 1,
        "passed": checks.passed,
        "check_count": len(checks.records),
        "failed_checks": [record["name"] for record in checks.records if not record["passed"]],
        "tolerances": TOLERANCES,
        "library_sha256": sha256(args.library) if args.library.is_file() else None,
        "executable_sha256": sha256(args.executable) if args.executable.is_file() else None,
        "runtime_included": args.runtime,
        "source_sha256": source_after,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "checks": checks.records,
        "error": error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("passed", "check_count", "failed_checks", "error")}
        )
    )
    return 0 if checks.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
