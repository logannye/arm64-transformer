"""Measure synthetic native kernels and PyTorch forward operations on one CPU thread.

This is a diagnostic microbenchmark, not a replacement for the matched study.
Native inputs, output buffers and ctypes pointers are prepared before timing.
Every observation includes Python-call/FFI or PyTorch-dispatch overhead; no control
measurement is subtracted. Component times cannot be interpreted as additive
fractions of an internally profiled model execution.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 777
WARMUPS = 30
BLOCKS = 10
CALLS = 200
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def create_cases(libraries, np, torch):
    from torch.nn import functional as F

    from transformer import spec
    from transformer.reference import Transformer
    from transformer.validate import TOLERANCES, Native, workspace_tensor

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = Transformer(seed=SEED).eval()
    params = model.flat_parameters()
    tokens = rng.integers(0, spec.VOCAB, spec.CONTEXT, dtype=np.int32)
    targets = rng.integers(0, spec.VOCAB, spec.CONTEXT, dtype=np.int32)
    ttokens = torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)
    ttargets = torch.from_numpy(targets.astype(np.int64))
    with torch.no_grad():
        logits, trace = model(ttokens, return_trace=True)

    cases = {}
    keepalive = []
    sanity = []

    def add(name, call, scope):
        cases[name] = {"call": call, "scope": scope}

    def bind(native, name, *args):
        keepalive.extend(args)
        converted = []
        for value in args:
            if isinstance(value, np.ndarray):
                if not value.flags.c_contiguous:
                    raise ValueError("Native benchmark arguments must be contiguous")
                converted.append(ct.c_void_p(value.ctypes.data))
            else:
                converted.append(value)
        fn = getattr(native.lib, "tf_" + name)
        return lambda: fn(*converted)

    def parameter(name):
        offset, size = spec.PARAMS[name]
        return params[offset : offset + size].reshape(spec.PARAM_SHAPES[name])

    def feature(name):
        return np.ascontiguousarray(trace[name].squeeze(0).numpy(), dtype=np.float32)

    def check(name, actual, expected):
        expected = expected.detach().numpy() if isinstance(expected, torch.Tensor) else expected
        np.testing.assert_allclose(actual, expected, **TOLERANCES["forward"])
        sanity.append(
            {"name": name, "max_absolute_error": float(np.max(np.abs(actual - expected)))}
        )

    linear_inputs = {}
    for prefix, source in (
        ("qkv", "n1"),
        ("attention", "attention"),
        ("ff1", "n2"),
        ("ff2", "relu"),
        ("head", "n3"),
    ):
        x = feature(source)
        w, b = parameter(prefix + "_weight"), parameter(prefix + "_bias")
        shape = np.array([x.shape[0], x.shape[1], w.shape[0]], dtype=np.int32)
        dy = rng.normal(0, 0.01, (x.shape[0], w.shape[0])).astype(np.float32)
        tx, tw, tb = map(torch.from_numpy, (x, w, b))
        with torch.no_grad():
            expected = F.linear(tx, tw, tb)
        linear_inputs[prefix] = (x, w, b, shape, dy, expected)
        add(
            f"pytorch/linear_forward_{prefix}",
            lambda x=tx, w=tw, b=tb: F.linear(x, w, b),
            "PyTorch linear forward; framework output allocation and dispatch included",
        )

    qkv = feature("qkv")
    attention_shape = np.array([spec.CONTEXT, spec.WIDTH, spec.HEADS], dtype=np.int32)
    attention_dy = rng.normal(0, 0.01, (spec.CONTEXT, spec.WIDTH)).astype(np.float32)
    tqkv = torch.from_numpy(qkv).unsqueeze(0)
    q, k, v = [
        part.reshape(1, spec.CONTEXT, spec.HEADS, spec.WIDTH // spec.HEADS).transpose(1, 2)
        for part in tqkv.chunk(3, dim=-1)
    ]

    def torch_attention():
        return (
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            .transpose(1, 2)
            .contiguous()
            .reshape(spec.CONTEXT, spec.WIDTH)
        )

    with torch.no_grad():
        expected_attention = torch_attention()
    add(
        "pytorch/attention_forward",
        torch_attention,
        "PyTorch SDPA plus output layout; framework output allocation and dispatch included",
    )
    add(
        "pytorch/sdpa_only",
        lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True),
        "PyTorch SDPA only; excludes output-layout conversion",
    )

    norm_inputs = {}
    for index, source in ((1, "x"), (2, "r1"), (3, "r2")):
        x = feature(source)
        gamma, beta = parameter(f"ln{index}_weight"), parameter(f"ln{index}_bias")
        dy = rng.normal(0, 0.01, x.shape).astype(np.float32)
        tx, tg, tb = map(torch.from_numpy, (x, gamma, beta))
        with torch.no_grad():
            expected = F.layer_norm(tx, (spec.WIDTH,), tg, tb, spec.EPS)
        norm_inputs[index] = (x, gamma, beta, dy, expected)
        add(
            f"pytorch/ln{index}_forward",
            lambda x=tx, g=tg, b=tb: F.layer_norm(x, (spec.WIDTH,), g, b, spec.EPS),
            "PyTorch LayerNorm forward; framework output allocation and dispatch included",
        )

    loss_logits = np.ascontiguousarray(logits.squeeze(0).numpy())
    tloss_logits = torch.from_numpy(loss_logits)
    add(
        "pytorch/full_forward",
        lambda: model(ttokens),
        "Complete PyTorch batch-one, full-context forward; no autograd graph",
    )
    add(
        "pytorch/cross_entropy_loss_only",
        lambda: F.cross_entropy(tloss_logits, ttargets),
        "Forward loss only; does NOT include the logit gradient produced by native loss",
    )

    for label, path in libraries.items():
        native = Native(path, verify_model=label != "baseline")
        keepalive.append(native)
        workspace = native.forward(params, tokens)
        check(f"{label}/full_forward", workspace_tensor(workspace, "logits"), logits.squeeze(0))
        gradients = np.zeros_like(params)
        add(
            f"{label}/ffi_zero_length_relu",
            bind(native, "relu_forward", workspace, workspace, 0),
            "ctypes call control with zero numerical work; not subtracted from observations",
        )
        add(
            f"{label}/full_forward",
            bind(native, "forward", params, tokens, workspace),
            "Complete native batch-one, full-context forward; preallocated buffers",
        )
        add(
            f"{label}/full_backward_including_cross_entropy",
            bind(native, "backward", params, tokens, targets, workspace, gradients),
            "Native backward plus loss, reusing saved forward activations; gradients accumulate",
        )
        for prefix, (x, w, b, shape, dy, expected) in linear_inputs.items():
            y = np.empty(dy.shape, dtype=np.float32)
            call = bind(native, "linear_forward", x, w, b, y, shape)
            call()
            check(f"{label}/linear_forward_{prefix}", y, expected)
            add(
                f"{label}/linear_forward_{prefix}",
                call,
                "Native linear forward; preallocated output",
            )
            dx, dw, db = np.empty_like(x), np.zeros_like(w), np.zeros_like(b)
            add(
                f"{label}/linear_backward_{prefix}",
                bind(native, "linear_backward", x, w, dy, dx, dw, db, shape),
                "Native linear backward; dX overwritten, dW/db accumulate in preallocated buffers",
            )

        out = np.empty((spec.CONTEXT, spec.WIDTH), dtype=np.float32)
        probs = np.empty((spec.HEADS, spec.CONTEXT, spec.CONTEXT), dtype=np.float32)
        call = bind(native, "attention_forward", qkv, out, probs, attention_shape)
        call()
        check(f"{label}/attention_forward", out, expected_attention)
        add(f"{label}/attention_forward", call, "Native causal attention; preallocated buffers")
        dqkv = np.empty_like(qkv)
        add(
            f"{label}/attention_backward",
            bind(native, "attention_backward", qkv, probs, attention_dy, dqkv, attention_shape),
            "Native causal attention backward; Q/K/V gradients overwritten",
        )
        for index, (x, gamma, beta, dy, expected) in norm_inputs.items():
            y, norm = np.empty_like(x), np.empty_like(x)
            inv = np.empty(spec.CONTEXT, dtype=np.float32)
            call = bind(
                native, "ln_forward", x, gamma, beta, y, norm, inv, spec.CONTEXT, spec.WIDTH
            )
            call()
            check(f"{label}/ln{index}_forward", y, expected)
            add(f"{label}/ln{index}_forward", call, "Native LayerNorm; preallocated buffers")
            dx, dg, db = np.empty_like(x), np.zeros_like(gamma), np.zeros_like(beta)
            shape = np.array([spec.CONTEXT, spec.WIDTH], dtype=np.int32)
            add(
                f"{label}/ln{index}_backward",
                bind(native, "ln_backward", dy, norm, inv, gamma, dx, dg, db, shape),
                "Native LayerNorm backward; dX overwritten, dgamma/dbeta accumulate",
            )
        dlogits = np.empty_like(loss_logits)
        add(
            f"{label}/cross_entropy_with_gradient",
            bind(native, "cross_entropy", loss_logits, targets, dlogits, spec.CONTEXT, spec.VOCAB),
            "Native mean cross-entropy and logit gradient together; preallocated gradient",
        )
    return cases, keepalive, sanity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from transformer import spec

    if args.library is None:
        args.library = ROOT / f"native/build/width-{spec.WIDTH}/libtransformer.dylib"
    if args.output.exists():
        parser.error("Output exists; choose a fresh diagnostic JSON path")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Native kernel diagnostics require macOS ARM64")
    libraries = {"candidate": args.library.resolve()}
    if args.baseline_library is not None:
        libraries["baseline"] = args.baseline_library.resolve()
    for path in libraries.values():
        if not path.is_file():
            parser.error(f"Native library is missing; build explicitly first: {path}")
    library_hashes = {name: sha256(path) for name, path in libraries.items()}
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cases, keepalive, sanity = create_cases(libraries, np, torch)
    samples = {name: [] for name in cases}
    keys = list(cases)
    block_orders = []
    with torch.inference_mode():
        for value in cases.values():
            for _ in range(WARMUPS):
                value["call"]()
        for block in range(BLOCKS):
            rotation = block // 2
            order = keys[rotation:] + keys[:rotation]
            if block % 2:
                order = list(reversed(order))
            block_orders.append(order)
            for name in order:
                call = cases[name]["call"]
                start = time.perf_counter_ns()
                for _ in range(CALLS):
                    call()
                samples[name].append((time.perf_counter_ns() - start) / CALLS / 1000)
    if any(sha256(path) != library_hashes[name] for name, path in libraries.items()):
        raise RuntimeError("A native library changed during measurement; diagnostic rejected")
    result = {
        "scope": (
            "Synthetic seed-777, post-study kernel diagnostics; no study data or checkpoints. "
            "Native buffers and ctypes pointers are preallocated; one CPU compute thread. "
            "Every observation includes Python/FFI or PyTorch dispatch. No FFI subtraction. "
            "Native backward repeats reuse saved activations and accumulate parameter gradients. "
            "PyTorch timings run under inference_mode, with no autograd graph. Component times "
            "cannot be treated as additive runtime shares or as replacements for matched-study "
            "timings. The PyTorch loss-only case excludes the gradient produced by native loss."
        ),
        "seed": SEED,
        "warmups": WARMUPS,
        "blocks": BLOCKS,
        "calls_per_block": CALLS,
        "order_rule": "Paired forward/reverse blocks, rotated left by block index // 2",
        "block_orders": block_orders,
        "case_order": keys,
        "command": sys.argv,
        "revision": revision(),
        "script_sha256": sha256(__file__),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {key: os.environ[key] for key in THREAD_KEYS},
        "libraries": {
            name: {"path": str(path), "sha256": library_hashes[name]}
            for name, path in libraries.items()
        },
        "forward_sanity_checks": sanity,
        "cases": {
            name: {
                "scope": cases[name]["scope"],
                "median_us": float(np.median(values)),
                "min_us": min(values),
                "max_us": max(values),
                "block_us": values,
            }
            for name, values in samples.items()
        },
    }
    if "baseline" in libraries:
        result["baseline_over_candidate_median_ratios"] = {
            name.removeprefix("candidate/"): (
                result["cases"][name.replace("candidate/", "baseline/", 1)]["median_us"]
                / record["median_us"]
            )
            for name, record in result["cases"].items()
            if name.startswith("candidate/")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, record in result["cases"].items():
        print(f"{name:54s} {record['median_us']:10.3f} us")
    print(args.output.resolve())
    # Hold arrays, ctypes libraries, and function pointers through the last call.
    assert keepalive


if __name__ == "__main__":
    main()
