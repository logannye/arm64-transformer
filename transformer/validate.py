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
import os
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
from transformer.backend import (
    library_backend,
    read_backend,
    validate_runtime_backend,
    validation_contract,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = ROOT / f"native/build/width-{spec.WIDTH}"
DEFAULT_LIBRARY = DEFAULT_BUILD / "libtransformer.dylib"
DEFAULT_EXECUTABLE = DEFAULT_BUILD / "transformer-arm64"
GENERATED_LAYOUT = DEFAULT_BUILD / "layout.inc"
SOURCE_FILES = (
    "native/kernels.S",
    "native/linear.S",
    "native/linear_sme.S",
    "native/backend.S",
    "native/attention.S",
    "native/softmax.S",
    "native/normalization.S",
    "native/precision.S",
    "native/model.S",
    "native/runtime.S",
    "native/Makefile",
    "transformer/spec.py",
    "transformer/reference.py",
    "transformer/prepare.py",
    "transformer/python_train.py",
    "transformer/validate.py",
    "transformer/backend.py",
    "transformer/audit.py",
    "transformer/runtime_replay.py",
    "transformer/precision.py",
    "transformer/precision_audit.py",
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


def artifact_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def arr(value, dtype=np.float32):
    return np.ascontiguousarray(value, dtype=dtype)


def tensor(value, *, grad=False, dtype=torch.float32):
    return torch.tensor(value, dtype=dtype, requires_grad=grad)


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def verify_library_model(library, *, allow_legacy=False):
    """Reject incompatible model buffers before any native numerical call.

    Historical diagnostics can explicitly admit pre-identity width-32 binaries.
    A library that exports identity symbols is always checked, even in that mode.
    """
    width_fn = getattr(library, "tf_model_width", None)
    count_fn = getattr(library, "tf_parameter_count", None)
    if width_fn is None or count_fn is None:
        if width_fn is None and count_fn is None and allow_legacy and spec.WIDTH == 32:
            return {"width": 32, "parameter_count": spec.PARAM_COUNT, "verified": False}
        raise RuntimeError("Native library lacks required model identity symbols")
    for function in (width_fn, count_fn):
        function.argtypes = []
        function.restype = ct.c_uint64
    width, count = int(width_fn()), int(count_fn())
    if (width, count) != (spec.WIDTH, spec.PARAM_COUNT):
        raise RuntimeError(
            f"Native model mismatch: library width={width}, parameters={count}; "
            f"Python width={spec.WIDTH}, parameters={spec.PARAM_COUNT}"
        )
    return {"width": width, "parameter_count": count, "verified": True}


class Native:
    """Typed FFI into the independently built handwritten assembly library."""

    def __init__(self, path: Path | str = DEFAULT_LIBRARY, *, verify_model=True):
        self.path = Path(path).resolve()
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Native validation requires macOS ARM64")
        if not self.path.is_file():
            raise FileNotFoundError(f"Native library missing: {self.path}; build it explicitly")
        self.lib = ct.CDLL(str(self.path))
        self.model_identity = verify_library_model(self.lib, allow_legacy=not verify_model)
        self._allow_legacy_backend = not verify_model
        library_backend(self.lib, allow_legacy=self._allow_legacy_backend)
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

    @property
    def backend_identity(self):
        return library_backend(self.lib, allow_legacy=self._allow_legacy_backend)

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
        self.diagnostics = []
        self.precision_diagnostics = []

    def require(self, name, condition, detail=None):
        self.records.append({"name": name, "passed": bool(condition), "detail": detail})

    def diagnostic_require(self, name, condition, detail=None):
        self.diagnostics.append(
            {
                "name": name,
                "passed": bool(condition),
                "detail": detail,
                "admission_role": "diagnostic",
            }
        )

    def close(self, name, actual, expected, tolerance="forward", *, diagnostic=False):
        actual = as_numpy(actual)
        expected = as_numpy(expected)
        thresholds = TOLERANCES[tolerance]
        same_shape = actual.shape == expected.shape
        passed = (
            same_shape
            and np.isfinite(actual).all()
            and np.isfinite(expected).all()
            and np.allclose(actual, expected, equal_nan=False, **thresholds)
        )
        finite = np.isfinite(actual) & np.isfinite(expected) if same_shape else np.array([])
        if finite.any():
            errors = np.abs(actual[finite].astype(np.float64) - expected[finite])
            max_error = float(errors.max())
            rmse = float(np.sqrt(np.mean(errors**2)))
        else:
            max_error = rmse = None
        record = {
            "name": name,
            "passed": bool(passed),
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "tolerance": tolerance,
            "max_absolute_error": max_error,
            "rmse": rmse,
        }
        if diagnostic:
            self.diagnostics.append(record | {"admission_role": "diagnostic"})
            self.require(
                name + "/finite_shape",
                same_shape and bool(np.all(finite)),
                {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape)},
            )
        else:
            self.records.append(record)

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


def check_common_state(native, checks, params, moment_m, moment_v, tokens, targets, step, origin):
    """Compare both engines at identical evolving weights, then replay identical gradients."""
    tag = f"common_state/{step}/{origin}"
    params, moment_m, moment_v = (arr(value).copy() for value in (params, moment_m, moment_v))
    checks.require(
        tag + "/input_state_shape_finite",
        all(
            value.shape == (spec.PARAM_COUNT,) and np.isfinite(value).all()
            for value in (params, moment_m, moment_v)
        ),
    )
    point = tensor(params, grad=True)
    expected, traces = reference_forward(point, tensor(tokens, dtype=torch.long), trace=True)
    expected_loss = F.cross_entropy(expected, tensor(targets, dtype=torch.long))
    expected_loss.backward()
    workspace = native.forward(params, tokens)
    native_loss, gradients = native.backward(params, tokens, targets, workspace)
    for name, value in traces.items():
        checks.close(tag + "/trace/" + name, workspace_tensor(workspace, name), value)
    checks.close(tag + "/loss", native_loss, expected_loss)
    for name, (start, count) in spec.PARAMS.items():
        checks.close(
            tag + "/gradient/" + name,
            gradients[start : start + count],
            point.grad[start : start + count],
            "gradient",
        )
    for gradient_origin, gradient in (("native", gradients), ("python", point.grad.numpy())):
        replay = tag + "/adam/" + gradient_origin
        candidate = tensor(params, grad=True)
        optimizer = torch.optim.Adam([candidate], lr=spec.LR, foreach=False)
        optimizer.state[candidate] = {
            "step": torch.tensor(float(step - 1)),
            "exp_avg": tensor(moment_m),
            "exp_avg_sq": tensor(moment_v),
        }
        candidate.grad = tensor(gradient)
        native_params, native_m, native_v = params.copy(), moment_m.copy(), moment_v.copy()
        optimizer.step()
        native.call(
            "adam", native_params, arr(gradient), native_m, native_v, step, spec.PARAM_COUNT
        )
        state = optimizer.state[candidate]
        checks.require(
            replay + "/step",
            int(state["step"].item()) == step,
            {"requested_native_step": step, "reference_step": float(state["step"])},
        )
        checks.close(replay + "/parameters", native_params, candidate, "optimizer")
        checks.close(replay + "/m", native_m, state["exp_avg"], "optimizer")
        checks.close(replay + "/v", native_v, state["exp_avg_sq"], "optimizer")
        checks.require(
            replay + "/all_state_finite",
            all(
                np.isfinite(as_numpy(value)).all()
                for value in (
                    native_params,
                    native_m,
                    native_v,
                    candidate,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                )
            ),
        )


def legacy_diagnostic_names(runtime):
    names = []
    for step in range(1, 4):
        names.append(f"trajectory/{step}/loss")
        names.extend(f"trajectory/{step}/gradient/{name}" for name in spec.PARAMS)
        names.extend(f"trajectory/{step}/{name}" for name in ("parameters", "m", "v"))
    if runtime:
        names.extend(
            "runtime/matched/" + name
            for name in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32", "test_logits.f32")
        )
        names.extend(("runtime/history", "runtime/python/best_epoch"))
    return names


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
        if spec.WIDTH >= 512:
            state = optimizer.state.get(tp, {})
            check_common_state(native, checks, params, m, v, tokens, targets, step, "native")
            check_common_state(
                native,
                checks,
                tp.detach().numpy(),
                as_numpy(state["exp_avg"]) if state else np.zeros_like(params),
                as_numpy(state["exp_avg_sq"]) if state else np.zeros_like(params),
                tokens,
                targets,
                step,
                "python",
            )
        workspace = native.forward(params, tokens)
        native_loss, gradients = native.backward(params, tokens, targets, workspace)
        optimizer.zero_grad(set_to_none=True)
        reference_loss = F.cross_entropy(
            reference_forward(tp, tensor(tokens, dtype=torch.long)),
            tensor(targets, dtype=torch.long),
        )
        reference_loss.backward()
        checks.close(
            f"trajectory/{step}/loss",
            native_loss,
            reference_loss,
            "trajectory",
            diagnostic=spec.WIDTH >= 512,
        )
        for name, (start, count) in spec.PARAMS.items():
            checks.close(
                f"trajectory/{step}/gradient/{name}",
                gradients[start : start + count],
                tp.grad[start : start + count],
                "trajectory",
                diagnostic=spec.WIDTH >= 512,
            )
        optimizer.step()
        native.call("adam", params, gradients, m, v, step, spec.PARAM_COUNT)
        checks.close(
            f"trajectory/{step}/parameters", params, tp, "trajectory", diagnostic=spec.WIDTH >= 512
        )
        checks.close(
            f"trajectory/{step}/m",
            m,
            optimizer.state[tp]["exp_avg"],
            "trajectory",
            diagnostic=spec.WIDTH >= 512,
        )
        checks.close(
            f"trajectory/{step}/v",
            v,
            optimizer.state[tp]["exp_avg_sq"],
            "trajectory",
            diagnostic=spec.WIDTH >= 512,
        )


def guarded_array(shape, *, fill=np.nan):
    """Contiguous tensor surrounded by finite sentinels to detect tail overwrites."""
    count = int(np.prod(shape))
    storage = np.full(count + 16, np.float32(12345.75), dtype=np.float32)
    view = storage[8:-8].reshape(shape)
    view.fill(fill)
    return view, storage


def guard_intact(storage):
    return bool(
        np.all(storage[:8] == np.float32(12345.75)) and np.all(storage[-8:] == np.float32(12345.75))
    )


def check_tiled_linear_boundaries(native, checks, rng, *, shapes=None, prefix="v2"):
    # Fast forward: R/I/O divisible by 4; fast backward additionally I % 16 == 0.
    # Other positive shapes must remain correct through the generic fallback.
    shapes = shapes or (
        (1, 1, 1),
        (3, 5, 7),
        (4, 12, 20),
        (4, 32, 4),
        (8, 32, 96),
        (32, 32, 128),
        (4, 64, 32),
        (32, 64, 32),
        (5, 32, 96),
        (31, 32, 32),
        (33, 64, 12),
    )
    for rows, width, output in shapes:
        tag = f"{prefix}/linear/{rows}x{width}x{output}"
        x = rng.normal(0, 0.2, (rows, width)).astype(np.float32)
        w = rng.normal(0, 0.15, (output, width)).astype(np.float32)
        b = rng.normal(0, 0.1, output).astype(np.float32)
        dy = rng.normal(0, 0.2, (rows, output)).astype(np.float32)
        originals = [value.copy() for value in (x, w, b, dy)]
        shape = arr([rows, width, output], np.int32)
        y, y_storage = guarded_array((rows, output))
        native.call("linear_forward", x, w, b, y, shape)
        tx, tw, tb = [tensor(value, grad=True) for value in (x, w, b)]
        expected = F.linear(tx, tw, tb)
        (expected * tensor(dy)).sum().backward()
        checks.close(tag + "/forward", y, expected)
        checks.require(tag + "/output_guards", guard_intact(y_storage))
        dx, dx_storage = guarded_array(x.shape)
        dw, dw_storage = guarded_array(w.shape)
        db, db_storage = guarded_array(b.shape)
        initial_dw = rng.normal(0, 0.05, w.shape).astype(np.float32)
        initial_db = rng.normal(0, 0.05, b.shape).astype(np.float32)
        dw[...] = initial_dw
        db[...] = initial_db
        for accumulation in (1, 2):
            native.call("linear_backward", x, w, dy, dx, dw, db, shape)
            checks.close(f"{tag}/accumulation_{accumulation}/dx", dx, tx.grad, "gradient")
            checks.close(
                f"{tag}/accumulation_{accumulation}/dw",
                dw,
                initial_dw + tw.grad.numpy() * accumulation,
                "gradient",
            )
            checks.close(
                f"{tag}/accumulation_{accumulation}/db",
                db,
                initial_db + tb.grad.numpy() * accumulation,
                "gradient",
            )
        checks.require(
            tag + "/gradient_guards",
            all(guard_intact(storage) for storage in (dx_storage, dw_storage, db_storage)),
        )
        checks.require(
            tag + "/inputs_unchanged",
            all(
                np.array_equal(value, original)
                for value, original in zip((x, w, b, dy), originals, strict=True)
            ),
        )


def check_vector_norm_boundaries(native, checks, rng, *, shapes=None, prefix="v2"):
    shapes = shapes or (
        (1, 1, "constant"),
        (2, 3, "random"),
        (3, 12, "near_constant"),
        (4, 32, "random"),
        (5, 64, "zero"),
    )
    for rows, width, kind in shapes:
        tag = f"{prefix}/layernorm/{rows}x{width}/{kind}"
        x = rng.normal(0, 0.3, (rows, width)).astype(np.float32)
        if kind == "constant":
            x.fill(0.25)
        elif kind == "zero":
            x.fill(0)
        elif kind == "near_constant":
            x = (0.25 + x * 1e-4).astype(np.float32)
        gamma = rng.uniform(0.5, 1.5, width).astype(np.float32)
        beta = rng.normal(0, 0.1, width).astype(np.float32)
        dy = rng.normal(0, 0.1, x.shape).astype(np.float32)
        y, sy = guarded_array(x.shape)
        normalized, sn = guarded_array(x.shape)
        inv, si = guarded_array((rows,))
        native.call("ln_forward", x, gamma, beta, y, normalized, inv, rows, width)
        tx, tg, tb = [tensor(value, grad=True) for value in (x, gamma, beta)]
        expected = F.layer_norm(tx, (width,), tg, tb, spec.EPS)
        (expected * tensor(dy)).sum().backward()
        checks.close(tag + "/forward", y, expected)
        variance, mean = torch.var_mean(tx, dim=-1, correction=0, keepdim=True)
        checks.close(
            tag + "/normalized", normalized, (tx - mean) * torch.rsqrt(variance + spec.EPS)
        )
        checks.close(tag + "/inv", inv, torch.rsqrt(variance + spec.EPS).squeeze(-1))
        dx, sx = guarded_array(x.shape)
        dg, sg = guarded_array((width,), fill=0.125)
        db, sb = guarded_array((width,), fill=-0.125)
        native.call(
            "ln_backward", dy, normalized, inv, gamma, dx, dg, db, arr([rows, width], np.int32)
        )
        checks.close(tag + "/dx", dx, tx.grad, "gradient")
        checks.close(tag + "/dgamma_add", dg, tg.grad + 0.125, "gradient")
        checks.close(tag + "/dbeta_add", db, tb.grad - 0.125, "gradient")
        checks.require(
            tag + "/guards", all(guard_intact(storage) for storage in (sy, sn, si, sx, sg, sb))
        )


def check_vector_attention_boundaries(native, checks, rng, *, shapes=None, prefix="v2"):
    if shapes is None:
        shapes = [(tokens, 32, 4, "random") for tokens in (2, 3, 4, 5, 31)]
        shapes += [
            (32, 32, 4, "uniform"),
            (32, 32, 4, "large_scores"),
            (33, 32, 4, "random"),
            (5, 15, 3, "random"),
        ]
    for tokens, width, heads, kind in shapes:
        tag = f"{prefix}/attention/{tokens}x{width}x{heads}/{kind}"
        qkv = rng.normal(0, 0.3, (tokens, 3 * width)).astype(np.float32)
        if kind == "uniform":
            qkv[:, : 2 * width] = 0
        elif kind == "large_scores":
            qkv[:, :width] = 20
            qkv[:, width : 2 * width] = np.linspace(-20, 20, tokens)[:, None]
        original = qkv.copy()
        output, so = guarded_array((tokens, width))
        probs, sp = guarded_array((heads, tokens, tokens))
        shape = arr([tokens, width, heads], np.int32)
        native.call("attention_forward", qkv, output, probs, shape)
        tqkv = tensor(qkv, grad=True)
        q, k, v = [
            part.reshape(tokens, heads, width // heads).transpose(0, 1)
            for part in tqkv.chunk(3, dim=-1)
        ]
        mask = torch.ones(tokens, tokens, dtype=torch.bool).tril()
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // heads)
        expected_probs = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        expected = (expected_probs @ v).transpose(0, 1).reshape(tokens, width)
        checks.close(tag + "/forward", output, expected)
        checks.close(tag + "/probabilities", probs, expected_probs)
        checks.close(tag + "/mass", probs.sum(-1), np.ones((heads, tokens)))
        checks.require(tag + "/masked_exact_zero", np.all(probs[:, ~mask.numpy()] == 0))
        # Prefix outputs cannot depend on any future Q/K/V components.
        dy = rng.normal(0, 0.2, output.shape).astype(np.float32)
        prefix = (tokens + 1) // 2
        dy[prefix:] = 0
        (expected * tensor(dy)).sum().backward()
        dqkv, sd = guarded_array(qkv.shape)
        native.call("attention_backward", qkv, probs, dy, dqkv, shape)
        checks.close(tag + "/dqkv", dqkv, tqkv.grad, "gradient")
        checks.require(tag + "/future_gradient_exact_zero", np.all(dqkv[prefix:] == 0))
        checks.require(tag + "/guards", all(guard_intact(storage) for storage in (so, sp, sd)))
        checks.require(tag + "/qkv_unchanged", np.array_equal(qkv, original))


def check_vector_softmax_boundaries(native, checks, rng):
    for rows, vocab, kind in (
        (1, 1, "constant"),
        (3, 7, "random"),
        (2, 128, "ties"),
        (4, 128, "exp_boundaries"),
        (5, 128, "extreme"),
    ):
        tag = f"v2/cross_entropy/{rows}x{vocab}/{kind}"
        logits = rng.normal(0, 0.2, (rows, vocab)).astype(np.float32)
        targets = rng.integers(0, vocab, rows, dtype=np.int32)
        if kind == "constant":
            logits.fill(0.5)
        elif kind == "ties":
            logits.fill(-1000)
            logits[:, :2] = 1000
            targets[:] = np.arange(rows) % 2
        elif kind == "exp_boundaries":
            boundary = arr([0, -0.0001, -0.1, -1, -10, -80, -88, -103.9, -104, -104.1, -120])
            logits[...] = np.resize(boundary, vocab)
            targets[:] = [0, 1, 8, 10]
        elif kind == "extreme":
            logits.fill(-1000)
            logits[:, 0] = 1000
            logits[:, 1] = 999.75
            targets[:] = [0, 1, 127, 64, 2]
        original = logits.copy()
        gradients, storage = guarded_array(logits.shape)
        loss = native.call("cross_entropy", logits, targets, gradients, rows, vocab)
        reference = tensor(logits, grad=True)
        expected = F.cross_entropy(reference, tensor(targets, dtype=torch.long))
        expected.backward()
        checks.close(tag + "/loss", loss, expected)
        checks.close(tag + "/gradients", gradients, reference.grad, "gradient")
        checks.close(tag + "/gradient_mass", gradients.sum(-1), np.zeros(rows))
        checks.require(tag + "/guards", guard_intact(storage))
        checks.require(tag + "/logits_unchanged", np.array_equal(logits, original))


def check_attention_precision(native, checks, *, width=None, split="admission"):
    """V6 attention accuracy against independent FP64 equations.

    Existing FP32 compatibility checks and thresholds remain unchanged. New
    adversarial cases have their own FP32 compatibility diagnostics, with domain
    failures blocking even when a magnitude disagreement is diagnostic.
    """
    from transformer.precision import (
        ORACLE_TOLERANCES,
        attention_reference,
        attention_stress_cases,
    )

    width = spec.WIDTH if width is None else width
    for case in attention_stress_cases(width, split=split):
        tag = "v6/precision/" + case["name"]
        tokens, heads = case["tokens"], case["heads"]
        qkv, sq = guarded_array(case["qkv"].shape)
        qkv[...] = case["qkv"]
        dy, sy = guarded_array(case["dy"].shape)
        dy[...] = case["dy"]
        reference = attention_reference(qkv, heads, dy)
        shape = arr([tokens, width, heads], np.int32)
        output, so = guarded_array((tokens, width))
        probabilities, sp = guarded_array((heads, tokens, tokens))
        dqkv, sd = guarded_array(qkv.shape)
        native.call("attention_forward", qkv, output, probabilities, shape)
        before_probabilities = probabilities.copy()
        native.call("attention_backward", qkv, probabilities, dy, dqkv, shape)

        torch_references = {}
        mask = torch.ones(tokens, tokens, dtype=torch.bool).tril()
        for dtype in (torch.float64, torch.float32):
            tqkv = tensor(qkv, grad=True, dtype=dtype)
            q, k, v = [
                part.reshape(tokens, heads, width // heads).transpose(0, 1)
                for part in tqkv.chunk(3, dim=-1)
            ]
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // heads)
            probs = scores.masked_fill(~mask, -torch.inf).softmax(-1)
            out = (probs @ v).transpose(0, 1).reshape(tokens, width)
            (out * tensor(dy, dtype=dtype)).sum().backward()
            torch_references[dtype] = {
                "output": as_numpy(out),
                "probabilities": as_numpy(probs),
                "dqkv": as_numpy(tqkv.grad),
            }
        checks.require(
            tag + "/fp64_oracles_agree",
            all(
                np.allclose(
                    torch_references[torch.float64][key], reference[key], **ORACLE_TOLERANCES
                )
                for key in ("output", "probabilities", "dqkv")
            ),
            {
                "tolerances": ORACLE_TOLERANCES,
                "maximum_absolute_errors": {
                    key: float(np.abs(torch_references[torch.float64][key] - reference[key]).max())
                    for key in ("output", "probabilities", "dqkv")
                },
            },
        )
        checks.close(tag + "/output_fp64", output, reference["output"])
        checks.close(tag + "/probabilities_fp64", probabilities, reference["probabilities"], "math")
        checks.close(tag + "/dqkv_fp64", dqkv, reference["dqkv"], "gradient")
        same_probabilities = attention_reference(qkv, heads, dy, probabilities=probabilities)
        checks.close(
            tag + "/dqkv_same_probabilities_fp64", dqkv, same_probabilities["dqkv"], "gradient"
        )
        checks.close(
            tag + "/probability_mass", probabilities.sum(-1), np.ones((heads, tokens)), "math"
        )
        checks.require(tag + "/masked_exact_zero", np.all(probabilities[:, ~mask.numpy()] == 0))

        repeated_output, repeated_probs = np.empty_like(output), np.empty_like(probabilities)
        native.call("attention_forward", qkv, repeated_output, repeated_probs, shape)
        checks.require(
            tag + "/repeat_forward_exact",
            np.array_equal(output, repeated_output)
            and np.array_equal(probabilities, repeated_probs),
        )
        prefix = (tokens + 1) // 2
        intervened = qkv.copy()
        intervened[prefix:] = intervened[prefix:] * np.float32(-3.25) + np.float32(7.5)
        native.call("attention_forward", intervened, repeated_output, repeated_probs, shape)
        checks.require(
            tag + "/future_intervention_exact",
            np.array_equal(output[:prefix], repeated_output[:prefix])
            and np.array_equal(probabilities[:, :prefix], repeated_probs[:, :prefix]),
        )
        prefix_dy = dy.copy()
        prefix_dy[prefix:] = 0
        prefix_derivative, sg = guarded_array(qkv.shape)
        native.call("attention_backward", qkv, probabilities, prefix_dy, prefix_derivative, shape)
        checks.close(
            tag + "/prefix_dqkv_fp64",
            prefix_derivative,
            attention_reference(qkv, heads, prefix_dy)["dqkv"],
            "gradient",
        )
        checks.require(tag + "/future_gradient_exact_zero", np.all(prefix_derivative[prefix:] == 0))
        checks.require(tag + "/guards", all(guard_intact(s) for s in (sq, sy, so, sp, sd, sg)))
        checks.require(
            tag + "/inputs_unchanged",
            np.array_equal(qkv, case["qkv"])
            and np.array_equal(dy, case["dy"])
            and np.array_equal(probabilities, before_probabilities),
        )
        actuals = {"output": output, "probabilities": probabilities, "dqkv": dqkv}
        for group, actual, expected in (
            ("native_vs_torch32", actuals, torch_references[torch.float32]),
            ("torch32_vs_fp64", torch_references[torch.float32], reference),
        ):
            for key in ("output", "probabilities", "dqkv"):
                tolerance = (
                    "gradient" if key == "dqkv" else "math" if key == "probabilities" else "forward"
                )
                diagnostic = Checks()
                diagnostic.close(
                    tag + f"/{group}/{key}", actual[key], expected[key], tolerance, diagnostic=True
                )
                checks.precision_diagnostics.extend(diagnostic.diagnostics)
                checks.records.extend(diagnostic.records)


def check_vector_finite_boundaries(native, checks):
    pattern = arr(
        [
            0,
            -0.0,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            np.nextafter(np.float32(0), np.float32(1)),
            -1,
        ]
    )
    for count in (0, 1, 3, 4, 5, 17):
        tag = f"v2/finite/count_{count}"
        values, storage = guarded_array((count,))
        if count:
            values[...] = np.resize(pattern, count)
        before = storage.view(np.uint32).copy()
        checks.require(tag + "/all_finite", native.call("finite", values, count) == 1)
        checks.require(tag + "/input_unchanged", np.array_equal(storage.view(np.uint32), before))
        original = values.copy()
        for index in range(count):
            for label, invalid in (
                ("nan", np.nan),
                ("positive_inf", np.inf),
                ("negative_inf", -np.inf),
            ):
                values[...] = original
                values[index] = invalid
                before = storage.view(np.uint32).copy()
                checks.require(
                    f"{tag}/{index}/{label}/reject", native.call("finite", values, count) == 0
                )
                checks.require(
                    f"{tag}/{index}/{label}/input_unchanged",
                    np.array_equal(storage.view(np.uint32), before),
                )


def check_width_boundaries(native, checks, rng):
    """Exercise every selected-width projection and attention head boundary."""
    width = spec.WIDTH
    check_tiled_linear_boundaries(
        native,
        checks,
        rng,
        shapes=(
            (4, width, 3 * width),
            (32, width, 3 * width),
            (32, width, width),
            (32, width, 2 * width),
            (32, 2 * width, width),
            (32, width, spec.VOCAB),
            (5, width, 3 * width),
            (4, width + 1, 3 * width),
        ),
        prefix="v3",
    )
    check_vector_norm_boundaries(
        native,
        checks,
        rng,
        shapes=(
            (4, width, "random"),
            (3, width, "near_constant"),
            (2, width, "zero"),
            (1, width + 1, "random"),
        ),
        prefix="v3",
    )
    attention_shapes = [(tokens, width, spec.HEADS, "random") for tokens in (1, 2, 3, 4, 5, 31, 32)]
    attention_shapes += [
        (32, width, spec.HEADS, "uniform"),
        (32, width, spec.HEADS, "large_scores"),
        (33, width, spec.HEADS, "random"),
        (3, width + spec.HEADS, spec.HEADS, "random"),
        (5, width, 8, "random"),
    ]
    check_vector_attention_boundaries(native, checks, rng, shapes=attention_shapes, prefix="v3")

    # The public whole-model layout must remain within its declared buffers.
    params, sp = guarded_array((spec.PARAM_COUNT,))
    params[...] = initial_parameters(rng)
    original = params.copy()
    workspace, sw = guarded_array((spec.WORK_COUNT,))
    gradients, sg = guarded_array((spec.PARAM_COUNT,), fill=0.125)
    tokens = rng.integers(0, spec.VOCAB, spec.CONTEXT, dtype=np.int32)
    tokens[::3] = 32
    targets = rng.integers(0, spec.VOCAB, spec.CONTEXT, dtype=np.int32)
    original_tokens, original_targets = tokens.copy(), targets.copy()
    native.call("forward", params, tokens, workspace)
    checks.require("v3/model/forward_workspace_guards", guard_intact(sw))
    native.call("backward", params, tokens, targets, workspace, gradients)
    checks.require("v3/model/backward_guards", all(guard_intact(value) for value in (sp, sw, sg)))
    checks.require("v3/model/gradients_finite", np.isfinite(gradients).all())
    checks.require(
        "v3/model/inputs_unchanged",
        np.array_equal(params, original)
        and np.array_equal(tokens, original_tokens)
        and np.array_equal(targets, original_targets),
    )


def check_backend_boundaries(native, checks):
    """V4 independent stream: alignment, panel boundaries, stale-pack and ABI probes."""
    rng = np.random.default_rng(20261030)
    cases = [
        (16, 64, 64),
        (31, 64, 64),
        (32, 64, 64),
        (33, 64, 64),
        (32, 63, 64),
        (32, 65, 64),
        (32, 64, 63),
        (32, 64, 65),
        (32, spec.WIDTH, 3 * spec.WIDTH),
    ]
    for index, (rows, inputs, outputs) in enumerate(cases):
        tag = f"backend/linear/{rows}x{inputs}x{outputs}"
        stores = []

        def guarded(shape):
            count = math.prod(shape)
            start = 8 + 1 + index % 3  # Exercise 4/8/12-byte offsets from 16-byte alignment.
            storage = np.full(start + count + 8, -12345.25, dtype=np.float32)
            value = storage[start : start + count].reshape(shape)
            stores.append((storage, start, count))
            return value

        x, w, b, dout = [
            guarded(shape)
            for shape in ((rows, inputs), (outputs, inputs), (outputs,), (rows, outputs))
        ]
        for value in (x, w, b, dout):
            value[:] = rng.normal(0, 0.1, value.shape)
        out, dx, dw, db = [guarded(shape) for shape in ((rows, outputs), x.shape, w.shape, b.shape)]
        shape = arr([rows, inputs, outputs], np.int32)
        for iteration in range(2):
            # Change weights at the SAME address; a hidden stale packing cache must fail.
            w += np.float32(0.01 * iteration)
            tx, tw, tb = (tensor(value, grad=True) for value in (x, w, b))
            expected = F.linear(tx, tw, tb)
            expected.backward(tensor(dout))
            dw[:] = rng.normal(0, 0.03, dw.shape)
            db[:] = rng.normal(0, 0.03, db.shape)
            dw_before, db_before = dw.copy(), db.copy()
            originals = [value.copy() for value in (x, w, b, dout, shape)]
            out.fill(np.nan)
            dx.fill(np.nan)
            native.call("linear_forward", x, w, b, out, shape)
            native.call("linear_backward", x, w, dout, dx, dw, db, shape)
            checks.close(f"{tag}/{iteration}/forward", out, expected)
            checks.close(f"{tag}/{iteration}/dx", dx, tx.grad, "gradient")
            checks.close(f"{tag}/{iteration}/dw", dw, tw.grad.numpy() + dw_before, "gradient")
            checks.close(f"{tag}/{iteration}/db", db, tb.grad.numpy() + db_before, "gradient")
            checks.require(
                f"{tag}/{iteration}/inputs_immutable",
                all(
                    np.array_equal(value, before)
                    for value, before in zip((x, w, b, dout, shape), originals, strict=True)
                ),
            )
            checks.require(
                f"{tag}/{iteration}/guards",
                all(
                    np.all(storage[:start] == -12345.25)
                    and np.all(storage[start + count :] == -12345.25)
                    for storage, start, count in stores
                ),
            )
    abi_check = getattr(native.lib, "tf_backend_abi_check")
    abi_check.argtypes, abi_check.restype = [], ct.c_uint64
    checks.require("backend/callee_saved_FP_and_streaming_state", abi_check() == 1)
    before = native.backend_identity
    # Init is idempotent and captures the process policy exactly once.
    checks.require("backend/repeated_init_identity", before == native.backend_identity)


def check_wide_matrix_limits(native, checks):
    """V5 public SME limit/fallback checks, separate from historical random streams."""
    if spec.WIDTH < 512:
        return
    rng = np.random.default_rng(20261031)
    for shape in (
        (32, 2048, 64),
        (32, 2049, 64),
        (32, 1536, 64),
        (32, 64, 3072),
        (32, 64, 3073),
        (31, 2048, 64),
    ):
        before = native.backend_identity
        check_tiled_linear_boundaries(native, checks, rng, shapes=(shape,), prefix="v5")
        after = native.backend_identity
        rows, inputs, outputs = shape
        eligible = (
            rows == 32
            and 64 <= inputs <= 2048
            and inputs & (inputs - 1) == 0
            and 64 <= outputs <= 3072
            and outputs % 32 == 0
        )
        selected = "sme" if before["selected_kind"] and eligible else "neon"
        other = "neon" if selected == "sme" else "sme"
        expected = {
            f"{selected}_forward_calls": 1,
            f"{selected}_backward_calls": 2,
            f"{other}_forward_calls": 0,
            f"{other}_backward_calls": 0,
        }
        checks.require(
            f"v5/linear/{rows}x{inputs}x{outputs}/observed_dispatch",
            all(after[key] - before[key] == value for key, value in expected.items()),
        )


def numerical_checks(native, checks):
    torch.set_num_threads(1)
    checks.require("model/library_identity_verified", native.model_identity["verified"])
    rng = np.random.default_rng(1729)
    check_linear(native, checks, rng)
    check_norm(native, checks, rng)
    check_attention(native, checks, rng)
    check_elementwise(native, checks, rng)
    check_loss(native, checks, rng)
    check_adam(native, checks, rng)
    check_complete_model(native, checks, rng)
    # Preserve the complete v1 fixture stream and thresholds. Additional v2
    # boundary probes use a separate seed after every original check.
    boundary_rng = np.random.default_rng(20261028)
    check_tiled_linear_boundaries(native, checks, boundary_rng)
    check_vector_norm_boundaries(native, checks, boundary_rng)
    check_vector_attention_boundaries(native, checks, boundary_rng)
    check_vector_softmax_boundaries(native, checks, boundary_rng)
    check_vector_finite_boundaries(native, checks)
    # V3 fixtures do not perturb either historical random stream.
    check_width_boundaries(native, checks, np.random.default_rng(20261029))
    check_backend_boundaries(native, checks)
    check_wide_matrix_limits(native, checks)
    # V6 uses a separate frozen admission stream and independent FP64 equations.
    check_attention_precision(native, checks)


def run_child(command, *, timeout=180, env=None):
    started = time.perf_counter()
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, cwd=ROOT, env=env
    )
    return {
        "command": [str(item) for item in command],
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "wall_seconds": time.perf_counter() - started,
    }


def check_runtime_shared_checkpoints(native, checks, fixture, directory):
    """Replay each engine's own saved weights through both complete implementations."""
    from transformer.audit import audit_run, prediction_metrics, read_exact
    from transformer.reference import Transformer

    model = Transformer(seed=1729)
    splits = {
        name: read_exact(fixture / f"{name}.i32", "<i4", (2, spec.CONTEXT + 1))
        for name in ("test", "validation")
    }
    for owner in ("native", "python"):
        run = directory / owner
        # Independent functional PyTorch recomputation also enforces exact file
        # sizes, finite states, second moments, selection and optimizer step count.
        audit_run(fixture, run, checks, recompute=True)
        history = read_exact(run / "history.f64", "<f8", (2, 3))
        saved_logits = read_exact(run / "test_logits.f32", "<f4", (2, spec.CONTEXT, spec.VOCAB))
        for checkpoint in ("best", "last"):
            params = read_exact(run / f"{checkpoint}.f32", "<f4", (spec.PARAM_COUNT,))
            model.load_flat(params)
            for split, sequences in splits.items():
                tag = f"runtime/shared_checkpoint/{owner}/{checkpoint}/{split}"
                native_logits = np.stack(
                    [
                        workspace_tensor(
                            native.forward(params, arr(row[:-1], np.int32)), "logits"
                        ).copy()
                        for row in sequences
                    ]
                )
                with torch.no_grad():
                    python_logits = model(tensor(sequences[:, :-1], dtype=torch.long)).numpy()
                checks.close(tag + "/native_vs_python", native_logits, python_logits)
                checks.require(
                    tag + "/all_logits_finite",
                    np.isfinite(native_logits).all() and np.isfinite(python_logits).all(),
                )
                if checkpoint == "best" and split == "test":
                    checks.close(tag + "/native_vs_own_saved", native_logits, saved_logits)
                    checks.close(tag + "/python_vs_own_saved", python_logits, saved_logits)
                    own = native_logits if owner == "native" else python_logits
                    checks.require(
                        tag + "/own_engine_saved_logits_exact", np.array_equal(own, saved_logits)
                    )
                if split == "validation":
                    expected_loss = history[:, 1].min() if checkpoint == "best" else history[-1, 1]
                    for engine, logits in (("native", native_logits), ("python", python_logits)):
                        checks.close(
                            tag + f"/{engine}_vs_own_history",
                            prediction_metrics(logits, sequences[:, 1:])["nll"],
                            expected_loss,
                            "trajectory",
                        )

    checks.require("runtime/shared_checkpoints_complete", True)


def runtime_checks(
    checks, directory, executable=DEFAULT_EXECUTABLE, *, native=None, precision_policy=0
):
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
    config = struct.unpack("<9I", (fixture / "config.bin").read_bytes())
    checks.require("runtime/fixture_version_two", config[1] == 2)
    checks.require("runtime/fixture_model_width", config[8] == spec.WIDTH)
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
    backend_config = {
        "epochs": 2,
        "train_sequences": 5,
        "validation_sequences": 2,
        "test_sequences": 2,
        "width": spec.WIDTH,
    }
    policy = os.environ.get("ARM64_TRANSFORMER_BACKEND", "auto")
    for name in ("native", "native-repeat"):
        metadata = read_backend(directory / name / "backend.u64", policy=policy)
        validate_runtime_backend(
            metadata, backend_config, policy=policy, precision_policy=precision_policy
        )
        checks.require(f"runtime/{name}/backend_dispatch_counts", True, metadata)
    checks.require(
        "runtime/backend_repeat_exact",
        (directory / "native/backend.u64").read_bytes()
        == (directory / "native-repeat/backend.u64").read_bytes(),
    )
    rejected_modes = ["invalid"] + ([] if metadata["sme_supported"] else ["sme"])
    for mode in rejected_modes:
        destination = directory / f"rejected-backend-{mode}"
        result = run_child(
            [str(executable), str(fixture), str(destination)],
            env=os.environ | {"ARM64_TRANSFORMER_BACKEND": mode},
        )
        records.append(result)
        checks.require(
            f"runtime/reject_backend_{mode}",
            result["returncode"] > 0 and not destination.exists(),
            result,
        )
    for filename in ("best.f32", "last.f32", "adam_m.f32", "adam_v.f32", "test_logits.f32"):
        python_values = np.fromfile(directory / "python" / filename, dtype="<f4")
        native_values = np.fromfile(directory / "native" / filename, dtype="<f4")
        checks.close(
            "runtime/matched/" + filename,
            native_values,
            python_values,
            "trajectory",
            diagnostic=spec.WIDTH >= 512,
        )
        checks.require(
            "runtime/repeat/" + filename,
            (directory / "native" / filename).read_bytes()
            == (directory / "native-repeat" / filename).read_bytes(),
        )
    python_history = np.fromfile(directory / "python/history.f64", "<f8").reshape(2, 3)
    native_history = np.fromfile(directory / "native/history.f64", "<f8").reshape(2, 3)
    checks.close(
        "runtime/history",
        native_history[:, :2],
        python_history[:, :2],
        "trajectory",
        diagnostic=spec.WIDTH >= 512,
    )
    native_repeat = np.fromfile(directory / "native-repeat/history.f64", "<f8").reshape(2, 3)
    checks.require(
        "runtime/repeat/history", np.array_equal(native_history[:, :2], native_repeat[:, :2])
    )
    for name in ("python", "native"):
        performance = np.fromfile(directory / name / "performance.f64", "<f8")
        checks.require(f"runtime/{name}/performance_shape", performance.shape == (8,))
        checks.require(f"runtime/{name}/four_optimizer_steps", performance[6] == 4)
        checks.require(f"runtime/{name}/positive_timers", np.all(performance[[0, 1, 7]] > 0))
        if spec.WIDTH >= 512 and name == "python":
            checks.diagnostic_require(
                f"runtime/{name}/best_epoch", performance[5] == 1 + native_history[:, 1].argmin()
            )
            checks.require(
                f"runtime/{name}/own_best_epoch",
                performance[5] == 1 + python_history[:, 1].argmin(),
            )
        else:
            checks.require(
                f"runtime/{name}/best_epoch", performance[5] == 1 + native_history[:, 1].argmin()
            )
        timings = np.fromfile(directory / name / "timings.f64", "<f8")
        checks.require(
            f"runtime/{name}/timings",
            timings.shape == (100,) and bool(np.all(np.isfinite(timings) & (timings > 0))),
        )

    if spec.WIDTH >= 512:
        if native is None:
            raise ValueError("Wide runtime checkpoint replay requires the audited native library")
        check_runtime_shared_checkpoints(native, checks, fixture, directory)
        from transformer.runtime_replay import check_runtime_updates

        check_runtime_updates(native, checks, fixture, directory)

    # Preserve the output bytes across a rejected duplicate invocation.
    before = {
        path.name: sha256(path) for path in (directory / "native").iterdir() if path.is_file()
    }
    result = run_child([str(executable), str(fixture), str(directory / "native")])
    records.append(result)
    after = {path.name: sha256(path) for path in (directory / "native").iterdir() if path.is_file()}
    checks.require("runtime/reject_existing_output", result["returncode"] != 0 and before == after)

    fixture_hashes = {path.name: sha256(path) for path in fixture.iterdir() if path.is_file()}
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
        "old_version": ("config.bin", lambda data: data[:4] + struct.pack("<I", 1) + data[8:]),
        "truncated_config": ("config.bin", lambda data: data[:-4]),
        "extra_config": ("config.bin", lambda data: data + b"\x00\x00\x00\x00"),
        "unsupported_width": ("config.bin", lambda data: data[:32] + struct.pack("<I", 33)),
        "mismatched_width": (
            "config.bin",
            lambda data: data[:32] + struct.pack("<I", 64 if spec.WIDTH == 32 else 32),
        ),
    }
    for name, (filename, corrupt) in corruptions.items():
        bad = directory / ("bad-" + name)
        # Shared immutable payloads avoid duplicating wide initial weights in every bad case.
        shutil.copytree(fixture, bad, copy_function=os.link)
        path = bad / filename
        corrupted = corrupt(path.read_bytes())
        path.unlink()
        path.write_bytes(corrupted)
        result = run_child([str(executable), str(bad), str(directory / ("rejected-" + name))])
        records.append(result)
        checks.require(
            "runtime/reject/" + name,
            result["returncode"] > 0,
            {"returncode": result["returncode"], "stderr": result["stderr"]},
        )
        checks.require(
            "runtime/rejected_output_absent/" + name,
            not (directory / ("rejected-" + name)).exists(),
        )
    checks.require(
        "runtime/rejection_fixtures_preserve_original_inputs",
        fixture_hashes == {path.name: sha256(path) for path in fixture.iterdir() if path.is_file()},
    )
    (directory / "child-processes.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    from transformer.precision import (
        ORACLE_TOLERANCES,
        PRECISION_CONTRACT,
        PRECISION_SEEDS,
        expected_precision_check_names,
        expected_precision_diagnostic_names,
        precision_case_specs,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--development-precision-policy", type=int, choices=range(8))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Audit output already exists; choose a fresh evidence path")
    if args.runtime and args.work_dir is None:
        parser.error("--runtime requires a fresh --work-dir")
    checks = Checks()
    error = None
    identity = None
    backend = None
    compiled_precision = None
    expected_precision = args.development_precision_policy or 0
    source_before = {
        name: sha256(ROOT / name) if (ROOT / name).is_file() else None for name in SOURCE_FILES
    }
    layout_before = sha256(GENERATED_LAYOUT) if GENERATED_LAYOUT.is_file() else None
    binary_before = {
        "library": sha256(args.library) if args.library.is_file() else None,
        "executable": sha256(args.executable) if args.executable.is_file() else None,
    }
    try:
        native = Native(args.library)
        identity = native.model_identity
        precision_function = native.lib.tf_precision_policy
        precision_function.argtypes, precision_function.restype = [], ct.c_uint64
        compiled_precision = int(precision_function())
        if compiled_precision != expected_precision:
            raise ValueError(
                "Compiled precision policy differs from the explicit validation policy"
            )
        numerical_checks(native, checks)
        backend = native.backend_identity
        checks.require("backend/identity", backend is not None)
        if backend["selected_kind"] and spec.WIDTH >= 64:
            checks.require("backend/SME_forward_executed", backend["sme_forward_calls"] > 0)
            checks.require("backend/SME_backward_executed", backend["sme_backward_calls"] > 0)
        checks.require(
            "backend/NEON_fallback_executed",
            backend["neon_forward_calls"] > 0 and backend["neon_backward_calls"] > 0,
        )
        if args.runtime:
            runtime_checks(
                checks,
                args.work_dir,
                args.executable,
                native=native,
                precision_policy=expected_precision,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        checks.require("audit/exception", False, error)
    if spec.WIDTH >= 512:
        checks.require(
            "audit/legacy_diagnostics_complete",
            [record["name"] for record in checks.diagnostics]
            == legacy_diagnostic_names(args.runtime),
        )
    checks.require(
        "audit/precision_checks_complete",
        [record["name"] for record in checks.records if record["name"].startswith("v6/precision/")]
        == expected_precision_check_names(spec.WIDTH),
    )
    checks.require(
        "audit/precision_diagnostics_complete",
        [record["name"] for record in checks.precision_diagnostics]
        == expected_precision_diagnostic_names(spec.WIDTH),
    )
    source_after = {
        name: sha256(ROOT / name) if (ROOT / name).is_file() else None for name in SOURCE_FILES
    }
    checks.require("audit/source_files_present", all(source_after.values()))
    checks.require("audit/source_unchanged_during_checks", source_before == source_after)
    layout_after = sha256(GENERATED_LAYOUT) if GENERATED_LAYOUT.is_file() else None
    checks.require("audit/generated_layout_present", layout_after is not None)
    checks.require("audit/generated_layout_unchanged", layout_before == layout_after)
    checks.require(
        "audit/generated_layout_matches_spec",
        GENERATED_LAYOUT.is_file() and GENERATED_LAYOUT.read_text() == spec.assembly_constants(),
    )
    binary_after = {
        "library": sha256(args.library) if args.library.is_file() else None,
        "executable": sha256(args.executable) if args.executable.is_file() else None,
    }
    checks.require("audit/binaries_unchanged_during_checks", binary_before == binary_after)
    result = {
        "schema_version": 1,
        "validation_contract": (
            validation_contract(spec.WIDTH)
            if args.development_precision_policy is None
            else "trained-precision-development-v1"
        ),
        "native_precision_policy": compiled_precision,
        "legacy_diagnostics": checks.diagnostics,
        "legacy_diagnostic_count": len(checks.diagnostics),
        "legacy_failed_checks": [
            record["name"] for record in checks.diagnostics if not record["passed"]
        ],
        "legacy_contract_passed": all(record["passed"] for record in checks.diagnostics),
        "precision": {
            "contract": PRECISION_CONTRACT,
            "fixture_stream": "admission",
            "fixture_seed": PRECISION_SEEDS["admission"],
            "case_count": len(precision_case_specs(spec.WIDTH)),
            "oracle_tolerances": ORACLE_TOLERANCES,
            "diagnostics": checks.precision_diagnostics,
            "diagnostic_count": len(checks.precision_diagnostics),
            "failed_diagnostics": [
                record["name"] for record in checks.precision_diagnostics if not record["passed"]
            ],
        },
        "model_width": spec.WIDTH,
        "parameter_count": spec.PARAM_COUNT,
        "model_identity": identity,
        "backend": backend,
        "backend_environment": os.environ.get("ARM64_TRANSFORMER_BACKEND", "auto"),
        "library_path": artifact_path(args.library),
        "executable_path": artifact_path(args.executable),
        "passed": checks.passed,
        "check_count": len(checks.records),
        "failed_checks": [record["name"] for record in checks.records if not record["passed"]],
        "tolerances": TOLERANCES,
        "library_sha256": binary_after["library"],
        "executable_sha256": binary_after["executable"],
        "runtime_included": args.runtime,
        "source_sha256": source_after,
        "generated_layout_path": str(GENERATED_LAYOUT.relative_to(ROOT)),
        "generated_layout_sha256": layout_after,
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
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("passed", "check_count", "failed_checks", "error")}
        )
    )
    return 0 if checks.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
