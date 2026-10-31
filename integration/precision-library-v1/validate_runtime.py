"""Fresh all-width primitive gates for explicitly captured maintained precision APIs.

One engine and width per process; no installation, build, fit or admission data.
Retain actual/repeated arrays, two independent FP64 oracles and every derivative.
"""

import argparse
import importlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCER = "integration/precision-library-v1/validate_runtime.py"
OPERATIONS = ("linear", "layer_norm", "scaled_dot_product_attention")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = load(
    "primitive_captured_runtime", ROOT / "integration/trained-precision/captured_replay.py"
)
SUITE = REPLAY.SUITE
require = SUITE.require


def operands(np, width, operation, index):
    rng = np.random.default_rng(20261103 + width * 17 + index)
    if operation == "linear":
        shapes = ((2, 3, width), (3 * width, width), (3 * width,))
        output_shape = (2, 3, 3 * width)
    elif operation == "layer_norm":
        shapes = ((2, 3, width), (width,), (width,))
        output_shape = (2, 3, width)
    else:
        require(operation == "scaled_dot_product_attention", "Unknown primitive")
        shapes = ((2, 4, 5, width // 4),) * 3
        output_shape = shapes[0]
    values = [rng.normal(0, 0.15, shape).astype("f4") for shape in shapes]
    if operation == "layer_norm":
        values[1] += 1
    if operation == "scaled_dot_product_attention" and index == 1:
        # With V equal to an identity padded to head width, the actual output's
        # first T columns are the attention probabilities. This observes the
        # public operation without replacing it or inspecting a private graph.
        values[2].fill(0)
        for position in range(values[2].shape[-2]):
            values[2][..., position, position] = 1
    return values, rng.normal(0, 0.15, output_shape).astype("f4")


def numpy_oracle(np, operation, values, upstream):
    a, b, c = [x.astype("f8") for x in values]
    d = upstream.astype("f8")
    if operation == "linear":
        return a @ b.T + c, (
            d @ b,
            d.reshape(-1, d.shape[-1]).T @ a.reshape(-1, a.shape[-1]),
            d.reshape(-1, d.shape[-1]).sum(0),
        )
    if operation == "layer_norm":
        centered = a - a.mean(-1, keepdims=True)
        inverse = 1 / np.sqrt((centered * centered).mean(-1, keepdims=True) + 1e-5)
        n = centered * inverse
        dn = d * b
        dx = inverse * (dn - dn.mean(-1, keepdims=True) - n * (dn * n).mean(-1, keepdims=True))
        return n * b + c, (dx, (d * n).sum((0, 1)), d.sum((0, 1)))
    scale = 1 / math.sqrt(a.shape[-1])
    scores = a @ np.swapaxes(b, -1, -2) * scale
    mask = np.triu(np.ones(scores.shape[-2:], bool), 1)
    scores = np.where(mask, -np.inf, scores)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    dp = d @ np.swapaxes(c, -1, -2)
    ds = p * (dp - (dp * p).sum(-1, keepdims=True)) * scale
    return p @ c, (ds @ b, np.swapaxes(ds, -1, -2) @ a, np.swapaxes(p, -1, -2) @ d)


def torch_oracle(torch, operation, values, upstream):
    inputs = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in values]
    a, b, c = inputs
    if operation == "linear":
        y = torch.nn.functional.linear(a, b, c)
    elif operation == "layer_norm":
        y = torch.nn.functional.layer_norm(a, (a.shape[-1],), b, c, 1e-5)
    else:
        scores = (a @ b.transpose(-1, -2)) / math.sqrt(a.shape[-1])
        mask = torch.ones(a.shape[-2], a.shape[-2], dtype=torch.bool).triu(1)
        y = scores.masked_fill(mask, -torch.inf).softmax(-1) @ c
    gradients = torch.autograd.grad(y, inputs, torch.tensor(upstream, dtype=torch.float64))
    return y.detach().numpy().copy(), tuple(g.detach().numpy().copy() for g in gradients)


def torch_execution(torch, np, api, operation, values, upstream, device, noncontiguous):
    inputs = []
    for value in values:
        tensor = torch.tensor(value, dtype=torch.float32, device=device)
        if noncontiguous:
            tensor = torch.stack((tensor, tensor), dim=-1)[..., 0]
            require(not tensor.is_contiguous(), "Missing noncontiguous primitive operand")
        inputs.append(tensor.detach().requires_grad_())
    saved = []

    def pack(tensor):
        saved.append(dict(dtype=str(tensor.dtype), device=tensor.device.type))
        require(
            not tensor.is_floating_point() or tensor.dtype == torch.float32,
            "A non-FP32 floating-point tensor was saved",
        )
        require(tensor.device.type == device, "A saved tensor left the selected device")
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        y = getattr(api, operation)(*inputs)
        gradients = torch.autograd.grad(y, inputs, torch.tensor(upstream, device=device))
    require(
        y.device.type == device and all(g.device.type == device for g in gradients),
        "Primitive output or derivative left the selected device",
    )
    for observed, expected in zip(inputs, values, strict=True):
        require(
            observed.detach().cpu().numpy().tobytes() == expected.tobytes(),
            "Primitive mutated an input",
        )
    require(bool(saved), "Missing saved-tensor observation")
    arrays = [value.detach().cpu().numpy().copy() for value in (y, *gradients)]
    require(all(value.dtype == np.float32 for value in arrays), "Non-FP32 saved/result contract")
    return arrays, dict(saved_tensors=saved, noncontiguous_operands=noncontiguous)


def checks(np, arrays, *, probability_basis=False):
    result = {}
    for name in ("output", "gradient0", "gradient1", "gradient2"):
        actual, repeat = arrays["actual_" + name], arrays["repeat_" + name]
        numpy64, torch64 = arrays["numpy64_" + name], arrays["torch64_" + name]
        atol, rtol = (3e-5, 3e-3) if name.startswith("gradient") else (2e-5, 2e-4)
        require(
            actual.shape == repeat.shape == numpy64.shape == torch64.shape,
            "Primitive shape differs",
        )
        require(
            actual.dtype == repeat.dtype == np.float32
            and numpy64.dtype == torch64.dtype == np.float64,
            "Primitive dtype differs",
        )
        require(
            all(np.isfinite(value).all() for value in (actual, repeat, numpy64, torch64)),
            "Nonfinite primitive result",
        )
        result[name] = dict(
            agreement=bool(np.allclose(actual, numpy64, atol=atol, rtol=rtol)),
            dual_oracle=bool(np.allclose(torch64, numpy64, atol=1e-10, rtol=1e-10)),
            exact_replay=actual.tobytes() == repeat.tobytes(),
            max_absolute_error=float(np.max(np.abs(actual - numpy64))),
            atol=atol,
            rtol=rtol,
        )
    if probability_basis:
        length = arrays["input2"].shape[-2]
        values = [
            arrays[label + "_output"][..., :length]
            for label in ("actual", "repeat", "numpy64", "torch64")
        ]
        actual, repeat, numpy64, torch64 = values
        result["probabilities"] = dict(
            agreement=bool(np.allclose(actual, numpy64, atol=2e-7, rtol=2e-6)),
            dual_oracle=bool(np.allclose(torch64, numpy64, atol=1e-10, rtol=1e-10)),
            exact_replay=actual.tobytes() == repeat.tobytes(),
            max_absolute_error=float(np.max(np.abs(actual - numpy64))),
            atol=2e-7,
            rtol=2e-6,
        )
    return result


def malformed_inputs(api, engine, torch):
    if engine == "mlx":
        import mlx.core as mx

        def make(shape):
            return mx.zeros(shape, dtype=mx.float32)

        def integer(x):
            return x.astype(mx.int32)
    else:
        device = "mps" if engine == "mps" else "cpu"

        def make(shape):
            return torch.zeros(shape, dtype=torch.float32, device=device)

        def integer(x):
            return x.to(torch.int32)

    x, weight, bias, q = make((2, 3)), make((4, 3)), make((4,)), make((1, 1, 2, 4))
    calls = {
        "linear_wrong_dtype": lambda: api.linear(integer(x), weight, bias),
        "linear_incompatible_shape": lambda: api.linear(x, weight[:, :-1], bias),
        "norm_negative_epsilon": lambda: api.layer_norm(x, epsilon=-1),
        "norm_incompatible_affine": lambda: api.layer_norm(x, make((4,))),
        "attention_noncausal": lambda: api.scaled_dot_product_attention(q, q, q, is_causal=False),
        "attention_dropout": lambda: api.scaled_dot_product_attention(q, q, q, dropout_p=0.1),
    }
    result = []
    for name, function in calls.items():
        try:
            function()
        except ValueError as error:
            result.append(dict(id=name, rejected=True, error_type="ValueError", message=str(error)))
        else:
            raise ValueError("Malformed primitive input accepted: " + name)
    return result


def run(args):
    require(
        os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(args.width),
        "Set width before process start",
    )
    require(not args.output.exists(), "Primitive output must be fresh")
    sys.pycache_prefix = str(args.output / "unused-bytecode-cache")
    SUITE.check_controls()
    capture, plan = REPLAY.captured_plan(args.capture, args.capture_sha256)
    authority, backend, inventory, _ = SUITE.bootstrap(plan)
    receipt, _ = authority.verify()
    SUITE.output_path(args.output, receipt["project"])
    SUITE.pinned(ROOT / PRODUCER, receipt["project"]["files"][PRODUCER])
    framework = "mlx" if args.engine == "mlx" else "pytorch"
    module = "mlx.nn.precise_fp32" if framework == "mlx" else "torch.nn.precise_fp32"
    selected = receipt["frameworks"][framework]["selection"].get("precision_api", {})
    require(selected.get("module") == module, "Primitive API must be explicitly captured")
    api = importlib.import_module(module)
    require(api.IDENTITY == "fp32-saved-precision-api-v1", "Unknown installed precision API")
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.engine == "mps":
        require(torch.backends.mps.is_available(), "Real MPS required")
    record = dict(
        schema="maintained-library-primitive-execution-v1",
        engine=args.engine,
        width=args.width,
        capture=SUITE.binding(args.capture),
        producer=SUITE.binding(ROOT / PRODUCER),
        selected_api=selected,
        module_path=api.__file__,
        formal_eligible=False,
        formal_fits=0,
        admission_stream_opened=False,
        complete=False,
        passed=False,
        cases=[],
    )
    args.output.mkdir()
    try:
        record["origins_before"] = REPLAY.origins(args.engine, authority, backend, inventory)
        record["malformed_inputs"] = malformed_inputs(api, args.engine, torch)
        storage = load("primitive_raw_storage", ROOT / "transformer/trained_precision.py")
        total = 0
        for operation in OPERATIONS:
            trace_count = 0
            graph = None
            if args.engine == "mlx":
                import mlx.core as mx

                def traced(*inputs):
                    nonlocal trace_count
                    trace_count += 1
                    outputs, gradients = mx.vjp(getattr(api, operation), inputs[:-1], [inputs[-1]])
                    return (*outputs, *gradients)

                function = mx.compile(traced)
            prior = None
            for index in (0, 1):
                require(
                    shutil.disk_usage(args.output).free > 6 * 1024**3,
                    "Primitive disk reserve reached",
                )
                values, upstream = operands(np, args.width, operation, index)
                numpy64 = numpy_oracle(np, operation, values, upstream)
                torch64 = torch_oracle(torch, operation, values, upstream)
                arrays = {"input" + str(i): value for i, value in enumerate(values)}
                arrays["upstream"] = upstream
                for label, observed in (("numpy64", numpy64), ("torch64", torch64)):
                    for name, value in zip(
                        ("output", "gradient0", "gradient1", "gradient2"),
                        (observed[0], *observed[1]),
                        strict=True,
                    ):
                        arrays[label + "_" + name] = value
                executions = []
                for label in ("actual", "repeat"):
                    if args.engine == "mlx":
                        require(
                            mx.default_device() == mx.gpu and mx.metal.is_available(),
                            "Natural MLX GPU required",
                        )
                        inputs = [mx.array(value) for value in values]
                        if index:
                            inputs = [mx.stack((value, value), axis=-1)[..., 0] for value in inputs]
                        outputs = function(*inputs, mx.array(upstream))
                        require(trace_count == 1, "Compiled primitive re-entered Python")
                        if graph is None:
                            stream = io.StringIO()
                            mx.export_to_dot(stream, *outputs)
                            graph = stream.getvalue()
                            require("Compiled" in graph, "Missing actual compiled primitive graph")
                        mx.eval(*outputs)
                        require(
                            all(value.dtype == mx.float32 for value in outputs),
                            "Non-FP32 MLX outputs/derivatives",
                        )
                        observed = [np.array(value, copy=True) for value in outputs]
                        executions.append(
                            dict(
                                default_device="gpu",
                                outputs_and_derivatives_dtype="float32",
                                saved_state_scope=(
                                    "source-bound FP32 operations; no MLX saved-tensor hook"
                                ),
                            )
                        )
                        for value, wanted in zip(inputs, values, strict=True):
                            require(
                                np.array(value, copy=True).tobytes() == wanted.tobytes(),
                                "MLX primitive mutated input",
                            )
                    else:
                        observed, proof = torch_execution(
                            torch,
                            np,
                            api,
                            operation,
                            values,
                            upstream,
                            "mps" if args.engine == "mps" else "cpu",
                            bool(index),
                        )
                        executions.append(proof)
                    for name, value in zip(
                        ("output", "gradient0", "gradient1", "gradient2"), observed, strict=True
                    ):
                        arrays[label + "_" + name] = value
                changed = prior is None or arrays["actual_output"].tobytes() != prior
                prior = arrays["actual_output"].tobytes()
                require(
                    sum(value.nbytes for value in arrays.values()) <= 512 * 1024**2,
                    "Primitive raw-memory budget",
                )
                raw = storage.save_arrays(
                    args.output / f"{operation}-{index}.npz", arrays, args.raw_store
                )
                total += raw["bytes"]
                require(total <= 512 * 1024**2, "Primitive evidence budget exceeded")
                row = dict(
                    operation=operation,
                    index=index,
                    executions=executions,
                    changed_output=changed,
                    raw=raw,
                    passed=False,
                    probability_basis=operation == "scaled_dot_product_attention" and index == 1,
                )
                record["cases"].append(row)
                require(changed, "Changed operands did not change primitive output")
                result = checks(np, arrays, probability_basis=row["probability_basis"])
                row.update(
                    checks=result,
                    passed=all(
                        all(row[k] for k in ("agreement", "dual_oracle", "exact_replay"))
                        for row in result.values()
                    ),
                )
            if args.engine == "mlx":
                graph_path = args.output / (operation + ".dot")
                graph_path.write_text(graph)
                record.setdefault("compiled_graphs", {})[operation] = dict(
                    file=SUITE.binding(graph_path), python_trace_count=trace_count, executions=4
                )
        record.update(complete=True, passed=all(row["passed"] for row in record["cases"]))
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        try:
            record["origins_after"] = REPLAY.origins(args.engine, authority, backend, inventory)
            authority.verify()
            record["integrity_after"] = True
        except BaseException:
            record.update(
                integrity_after=False,
                complete=False,
                passed=False,
                integrity_error=traceback.format_exc(),
            )
        SUITE.write_json(args.output / "result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "output", "raw-store"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--engine", choices=("cpu_torch", "mps", "mlx"), required=True)
    parser.add_argument("--width", type=int, choices=(32, 64, 128, 256, 512, 1024), required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("complete", "passed")}))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
