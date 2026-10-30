"""Localize a retained compiled-MLX gradient failure without changing its model.

An auxiliary trace is usable only if logits, loss and every gradient reproduce
the original compiled worker bytes. The gate substitution is a separately named
FP64 backward counterfactual, never a replacement model or relaxed tolerance.
"""

import argparse
import importlib.util
import io
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def traced_forward(p, tokens, ops, model):
    """The exact policy-7 expressions, with auxiliary intermediate outputs."""
    batch, length = tokens.shape
    width = p["token"].shape[-1]
    trace = {}

    def keep(name, value):
        trace[name] = value
        return value

    x = keep("x", ops.embedding(tokens, p["token"]) + p["position"][:length])
    n1 = keep("n1", model.normalization(x, p["ln1_weight"], p["ln1_bias"], ops))
    qkv = keep("qkv", ops.precise_linear(n1, p["qkv_weight"], p["qkv_bias"]))
    q, k, v = [
        ops.swapaxes(part.reshape(batch, length, 4, width // 4), 1, 2)
        for part in ops.split_qkv(qkv)
    ]
    attention = model.FF.attention(q, k, v, ops)[0]
    attention = keep("attention", ops.swapaxes(attention, 1, 2).reshape(batch, length, width))
    projected = keep("projected", ops.linear(attention, p["attention_weight"], p["attention_bias"]))
    r1 = keep("r1", x + projected)
    n2 = keep("n2", ops.norm(r1, p["ln2_weight"], p["ln2_bias"]))
    ff1 = keep("ff1", ops.linear(n2, p["ff1_weight"], p["ff1_bias"]))
    hidden = keep("relu", ops.relu(ff1))
    ff2 = keep("ff2", ops.linear(hidden, p["ff2_weight"], p["ff2_bias"]))
    r2 = keep("r2", r1 + ff2)
    n3 = keep("n3", ops.norm(r2, p["ln3_weight"], p["ln3_bias"]))
    keep("logits", ops.linear(n3, p["head_weight"], p["head_bias"]))
    return trace


def same_bytes(first, second):
    return (
        first.dtype == second.dtype
        and first.shape == second.shape
        and first.tobytes() == second.tobytes()
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema="trained-framework-relu-localization-v1",
        complete=False,
        formal_eligible=False,
        admission_stream_opened=False,
        case=args.case,
        width=args.width,
    )
    runtime = load("branch_runtime", ROOT / "integration/gpu-owned/runtime_probe.py")
    auth = None
    try:
        if os.environ.get("ARM64_TRANSFORMER_WIDTH") != str(args.width):
            raise ValueError("Width must be set before imports")
        auth = runtime.bootstrap(args.workspace)
        sys.path.insert(0, str(ROOT))
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        import torch

        from transformer import spec
        from transformer.gpu_mlx_compile import verify_compilation
        from transformer.precision_audit import comparison
        from transformer.trained_precision import (
            checked_file,
            digest,
            save_arrays,
            source_inventory,
        )

        sources = source_inventory()
        for name in (
            "framework_relu_localize.py",
            "localize_gpu.py",
            "relu_localize.py",
            "audit_replay.py",
        ):
            path = Path(__file__).with_name(name)
            sources[str(path.relative_to(ROOT))] = digest(path)
        record["source_sha256"] = sources
        if (
            digest(args.receipt) != args.receipt_sha256
            or digest(args.inventory) != args.inventory_sha256
        ):
            raise ValueError("External replay authority changed")
        replay = json.loads(args.receipt.read_text())
        plan = json.loads(args.inventory.read_text())
        row = next(c for c in replay["cases"] if c["case"]["id"] == args.case)
        case = next(c for c in plan["cases"] if c["id"] == args.case)
        if not (
            replay["complete"] is True
            and replay["engine"] == "mlx"
            and replay["framework_candidate"] == "compensated-framework-model-diagnostic-v2"
            and replay["framework_candidate_policy"] == 7
            and replay["framework_compiled"] is True
            and replay["gradients_checked"] is True
            and replay["formal_eligible"] is False
            and replay["inventory_sha256"] == args.inventory_sha256
            and replay["source_sha256"] == source_inventory()
            and replay["batch_size"] == 16
            and row["case"] == case
            and case["width"] == args.width
        ):
            raise ValueError("Expected the unchanged compiled policy-7 gradient replay")
        record.update(
            receipt_sha256=args.receipt_sha256,
            inventory_sha256=args.inventory_sha256,
            owned_runtime=runtime.public_auth(auth),
        )
        flat = np.fromfile(
            checked_file(args.evidence_root, case["checkpoint"], case["checkpoint_sha256"]), "f4"
        )
        rows = np.fromfile(
            checked_file(args.evidence_root, case["rows"], case["rows_sha256"]), "i4"
        ).reshape(-1, 33)
        first = case["sequence"] // 16 * 16
        tokens, targets = (
            np.ascontiguousarray(a)
            for a in (rows[first : first + 16, :-1], rows[first : first + 16, 1:])
        )
        audit = load("branch_audit", Path(__file__).with_name("audit_replay.py"))
        with (
            audit.raw_path(
                args.receipt.parent, row["raw"], row["raw_sha256"], args.raw_store
            ) as raw,
            np.load(raw, allow_pickle=False) as arrays,
        ):
            if not same_bytes(tokens, arrays["tokens"]) or not same_bytes(
                targets, arrays["targets"]
            ):
                raise ValueError("Retained raw inputs differ")
            retained = {n: arrays["actual_" + n] for n in ("logits", "loss", "gradient")}
            oracle = {n: arrays["numpy64_" + n] for n in retained}
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        record["torch_origins"] = runtime.verify_torch_origins(torch, auth)
        record["mlx_origins"] = runtime.mlx_probe(auth)
        mx.set_default_device(mx.gpu)
        record["compilation_configuration"] = verify_compilation(mx)
        evaluator = load("branch_candidate", Path(__file__).with_name("framework_candidate.py"))
        model = evaluator.MODEL
        candidate = evaluator.Evaluator("mlx", 7, gradients=True, compiled=True)
        ordinary = candidate.evaluate(flat, tokens, targets)
        record["ordinary_reproduces_retained_bytes"] = {
            n: same_bytes(v, retained[n]) for n, v in ordinary.items()
        }
        if not all(record["ordinary_reproduces_retained_bytes"].values()):
            raise ValueError("Ordinary compiled execution no longer reproduces retained bytes")
        parameters = {
            n: mx.array(flat[start : start + count].reshape(spec.PARAM_SHAPES[n]))
            for n, (start, count) in spec.PARAMS.items()
        }
        x, y = mx.array(tokens), mx.array(targets)
        ops = model.mlx_ops(mx)
        traces = 0

        def loss(p, tokens, targets):
            nonlocal traces
            traces += 1
            values = traced_forward(p, tokens, ops, model)
            return nn.losses.cross_entropy(values["logits"], targets, reduction="mean"), values

        function = mx.compile(mx.value_and_grad(loss))
        saved = dict(
            tokens=tokens,
            targets=targets,
            ordinary_gradient=ordinary["gradient"],
            fp64_gradient=oracle["gradient"],
        )
        repeated = []
        graph = load("branch_graph", Path(__file__).with_name("graph_evidence.py"))
        for index in range(2):
            (value, trace), gradient = function(parameters, x, y)
            gradient = mx.concatenate([gradient[n].reshape(-1) for n in spec.PARAMS])
            if index == 0:
                text = io.StringIO()
                mx.export_to_dot(text, gradient)
                record["gradient_graph"] = graph.write(args.output, "gradient", text.getvalue())
            mx.eval(value, gradient, *trace.values())
            arrays = {n: np.array(v, copy=True) for n, v in trace.items()}
            arrays.update(loss=np.array(value, copy=True), gradient=np.array(gradient, copy=True))
            repeated.append(arrays)
        record["python_trace_count"] = traces
        if traces != 1 or not all(same_bytes(v, repeated[1][n]) for n, v in repeated[0].items()):
            raise ValueError("Traced compiled execution is not exact across repeats")
        actual = repeated[0]
        saved.update({"trace_" + n: v for n, v in actual.items()})
        record["trace_reproduces_retained_bytes"] = {
            n: same_bytes(actual[n], retained[n]) for n in retained
        }
        record["trace_usable_for_original_failure"] = all(
            record["trace_reproduces_retained_bytes"].values()
        )
        # Extra outputs can change compiler transformations. Never infer an
        # original branch from an instrumentation-altered compiled model.
        if record["trace_usable_for_original_failure"]:
            localization = load("branch_pipeline", Path(__file__).with_name("localize_gpu.py"))
            reference = localization.pipeline(flat, tokens, {})
            expected_gate, actual_gate = reference["ff1"] > 0, actual["ff1"] > 0
            coordinates = np.argwhere(expected_gate != actual_gate)
            record["gate_disagreements"] = len(coordinates)
            record["gate_examples"] = [
                dict(
                    index=c.tolist(),
                    mlx=float(actual["ff1"][tuple(c)]),
                    fp64=float(reference["ff1"][tuple(c)]),
                )
                for c in coordinates[:32]
            ]
            probe = mx.grad(lambda z: mx.sum(mx.maximum(z, 0)))(mx.array([-1.0, -0.0, 0.0, 1.0]))
            mx.eval(probe)
            if not same_bytes(np.array(probe), np.array([0, 0, 0, 1], "f4")):
                raise ValueError(
                    "Maximum derivative differs from the observed greater-than-zero gate"
                )
            saved.update(
                fp64_ff1=reference["ff1"], actual_gate=actual_gate, expected_gate=expected_gate
            )
            overrides, prefixes = {}, {}
            for name in localization.STAGES[:7]:
                overrides[name] = actual[name]
                prefix = localization.pipeline(flat, tokens, overrides)["ff1"]
                prefixes[name] = dict(
                    gate_disagreements=int(np.count_nonzero((prefix > 0) != expected_gate)),
                    preactivation=comparison("ff1", prefix, reference["ff1"]),
                )
                saved["prefix_" + name + "_ff1"] = prefix
            record["prefixes"] = prefixes
            relu = load("branch_counterfactual", Path(__file__).with_name("relu_localize.py"))

            def double_gradient(mask=None):
                p = {
                    n: torch.tensor(
                        flat[start : start + count].reshape(spec.PARAM_SHAPES[n]),
                        dtype=torch.float64,
                        requires_grad=True,
                    )
                    for n, (start, count) in spec.PARAMS.items()
                }
                operations = model.torch_ops(torch, "cpu")
                if mask is not None:
                    operations.relu = relu.frozen_gate_relu(torch, torch.tensor(mask))
                logits = model.forward(p, torch.tensor(tokens.astype("i8")), operations, 0)
                value = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 128), torch.tensor(targets.astype("i8")).reshape(-1)
                )
                value.backward()
                return logits.detach().numpy(), torch.cat(
                    [p[n].grad.reshape(-1) for n in spec.PARAMS]
                ).numpy()

            logits, gradient = double_gradient()
            if not (
                np.allclose(logits, oracle["logits"], atol=1e-10, rtol=1e-10)
                and np.allclose(gradient, oracle["gradient"], atol=1e-10, rtol=1e-10)
            ):
                raise ValueError("Independent FP64 equations differ from the retained oracle")
            frozen_logits, frozen_gradient = double_gradient(actual_gate)
            if not same_bytes(frozen_logits, logits):
                raise ValueError("Counterfactual changed forward execution")
            saved["frozen_gate_gradient"] = frozen_gradient
            record["original_comparison"] = comparison(
                "gradient", actual["gradient"], oracle["gradient"], tolerance="gradient"
            )
            record["frozen_gate_comparison"] = comparison(
                "gradient", actual["gradient"], frozen_gradient, tolerance="gradient"
            )
        stored = save_arrays(args.output / "arrays.npz", saved, args.raw_store)
        record.update(raw_sha256=stored["sha256"], raw_bytes=stored["bytes"], complete=True)
        if (
            any(digest(ROOT / name) != value for name, value in sources.items())
            or digest(args.receipt) != args.receipt_sha256
            or digest(args.inventory) != args.inventory_sha256
        ):
            raise ValueError("Diagnostic inputs or source changed")
    except BaseException:
        record.update(complete=False, error=traceback.format_exc())
    finally:
        if auth is not None:
            try:
                record["postflight"] = runtime.postflight(auth)
            except BaseException:
                record.update(complete=False, postflight_error=traceback.format_exc())
        with (args.output / "result.json").open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return 0 if record["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "evidence-root", "receipt", "inventory", "raw-store", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("receipt-sha256", "inventory-sha256", "case"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--width", type=int, choices=(512, 1024), required=True)
    raise SystemExit(run(parser.parse_args()))
