"""Explicit GPU model diagnostics, kept separate from all comparison arms."""

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args):
    runtime = load("model_probe_runtime", ROOT / "integration/gpu-owned/runtime_probe.py")
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema="trained-precision-development-v1",
        engine=args.engine,
        width=args.width,
        batch_size=16,
        gradients_checked=not args.forward_only,
        framework_candidate="compensated-framework-model-diagnostic-v2",
        framework_candidate_policy=args.policy,
        framework_compiled=args.compiled,
        framework_operation="forward" if args.forward_only else "gradient",
        complete=False,
        passed=False,
        formal_eligible=False,
        admission_stream_opened=False,
        cases=[],
        pid=os.getpid(),
    )
    auth = None
    sources = {}
    try:
        if os.environ.get("ARM64_TRANSFORMER_WIDTH") != str(args.width):
            raise ValueError("Width must be set before import")
        if args.compiled and args.engine != "mlx":
            raise ValueError("Only the already-installed MLX graph compiler is in scope")
        if sha(args.inventory) != args.inventory_sha256:
            raise ValueError("Inventory authority changed")
        plan = json.loads(args.inventory.read_text())
        case = next(c for c in plan["cases"] if c["id"] == args.case)
        if case["width"] != args.width or plan["formal_eligible"] is not False:
            raise ValueError("Wrong development inventory")
        auth = runtime.bootstrap(args.workspace)
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch

        from transformer import spec
        from transformer.gpu_admission_checks import numpy_model
        from transformer.precision_audit import comparison
        from transformer.trained_precision import (
            checked_file,
            save_arrays,
            source_inventory,
            torch_evaluate,
        )

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        sources = source_inventory()
        for filename in (
            "framework_probe.py",
            "framework_reference.py",
            "compensated_attention.py",
        ):
            path = Path(__file__).with_name(filename)
            sources[str(path.relative_to(ROOT))] = sha(path)
        record.update(
            source_sha256=sources,
            inventory_sha256=args.inventory_sha256,
            expected_cases=[args.case],
            full_width_inventory=False,
            owned_runtime=runtime.public_auth(auth),
            runtime={"numpy": np.__version__, "torch": torch.__version__},
            runtime_origins_before=runtime.verify_torch_origins(torch, auth),
        )
        flat = np.fromfile(
            checked_file(args.evidence_root, case["checkpoint"], case["checkpoint_sha256"]), "f4"
        )
        rows = np.fromfile(
            checked_file(args.evidence_root, case["rows"], case["rows_sha256"]), "i4"
        ).reshape(-1, 33)
        first = case["sequence"] // 16 * 16
        batch = rows[first : first + 16]
        tokens, targets = np.ascontiguousarray(batch[:, :-1]), np.ascontiguousarray(batch[:, 1:])
        oracle = numpy_model(flat, tokens, targets, backward=not args.forward_only)
        double = torch_evaluate(
            flat, tokens, targets, dtype=torch.float64, gradients=not args.forward_only
        )
        candidate = load("framework_reference", Path(__file__).with_name("framework_reference.py"))
        parameters = {
            name: flat[start : start + count].reshape(spec.PARAM_SHAPES[name])
            for name, (start, count) in spec.PARAMS.items()
        }
        if args.engine == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("A real MPS device is required")
            ops = candidate.torch_ops(torch, "mps")
            x, y = (torch.tensor(a.astype("i8"), device="mps") for a in (tokens, targets))

            def evaluate():
                p = {
                    name: torch.tensor(a, device="mps", requires_grad=not args.forward_only)
                    for name, a in parameters.items()
                }
                with torch.set_grad_enabled(not args.forward_only):
                    logits = candidate.forward(p, x, ops, args.policy)
                    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), y.reshape(-1))
                    values = dict(logits=logits, loss=loss)
                    if not args.forward_only:
                        loss.backward()
                        values["gradient"] = torch.cat(
                            [p[name].grad.reshape(-1) for name in spec.PARAMS]
                        )
                return {name: value.detach().cpu().numpy().copy() for name, value in values.items()}

        else:
            import mlx.core as mx
            import mlx.nn as nn

            record["mlx_origins"] = runtime.mlx_probe(auth)
            mx.set_default_device(mx.gpu)
            ops = candidate.mlx_ops(mx)
            x, y = mx.array(tokens), mx.array(targets)
            p = {name: mx.array(a) for name, a in parameters.items()}

            def loss(p, x, y):
                logits = candidate.forward(p, x, ops, args.policy)
                return nn.losses.cross_entropy(logits, y, reduction="mean"), logits

            def forward_only(p, x, y):
                return candidate.forward(p, x, ops, args.policy)

            base_function = forward_only if args.forward_only else mx.value_and_grad(loss)
            trace_count = 0

            def traced(p, x, y):
                nonlocal trace_count
                trace_count += 1
                return base_function(p, x, y)

            function = traced
            if args.compiled:
                from transformer.gpu_mlx_compile import verify_compilation

                record["compilation_configuration"] = verify_compilation(mx)
                function = mx.compile(function)

            def evaluate():
                if args.forward_only:
                    logits = function(p, x, y)
                    values = dict(logits=logits)
                else:
                    (loss, logits), grad = function(p, x, y)
                    gradient = mx.concatenate([grad[name].reshape(-1) for name in spec.PARAMS])
                    values = dict(logits=logits, loss=loss, gradient=gradient)
                if args.compiled:
                    if trace_count != 1:
                        raise ValueError("Compiled model re-entered Python")
                    if "model_compilation" not in record:
                        graphs = {}
                        for name in ("logits",) if args.forward_only else ("logits", "gradient"):
                            graph = io.StringIO()
                            mx.export_to_dot(graph, values[name])
                            data = graph.getvalue()
                            graph_storage = load(
                                "graph_storage", Path(__file__).with_name("graph_evidence.py")
                            )
                            graphs[name] = graph_storage.write(args.output, name, data)
                        record["model_compilation"] = dict(graphs=graphs)
                    record["model_compilation"]["python_trace_count"] = trace_count
                mx.eval(*values.values())
                if args.forward_only:
                    # Materialize the inference graph before evaluating loss.
                    # Requesting a VJP may dispatch a different attention kernel.
                    values["loss"] = nn.losses.cross_entropy(logits, y, reduction="mean")
                    mx.eval(values["loss"])
                return {name: np.array(value, copy=True) for name, value in values.items()}

        actual, repeat = evaluate(), evaluate()
        if args.policy == 0:
            if args.engine == "mps":
                ordinary = torch_evaluate(
                    flat,
                    tokens,
                    targets,
                    dtype=torch.float32,
                    device="mps",
                    gradients=not args.forward_only,
                )
            else:
                from transformer.gpu_mlx import Backend

                backend = Backend(flat, max_batch=16, max_length=32)
                try:
                    device = backend.upload_inputs(tokens, targets)
                    if args.forward_only:
                        ordinary = dict(
                            logits=backend.to_numpy(backend.forward(device)).copy(),
                            loss=backend.to_numpy(backend.loss(device)).copy(),
                        )
                    else:
                        loss, gradient = backend.loss_and_grad(device)
                        ordinary = dict(
                            logits=backend.to_numpy(backend._last_logits).copy(),
                            loss=backend.to_numpy(loss).copy(),
                            gradient=backend.to_numpy(gradient).copy(),
                        )
                finally:
                    backend.close()
            record["ordinary_control_exact"] = {
                name: actual[name].dtype == value.dtype
                and actual[name].shape == value.shape
                and actual[name].tobytes() == value.tobytes()
                for name, value in ordinary.items()
            }
            control = save_arrays(args.output / "ordinary-control.npz", ordinary, args.raw_store)
            record["ordinary_control_raw"] = dict(
                path="ordinary-control.npz", sha256=control["sha256"], bytes=control["bytes"]
            )
        checks = {}
        for name in oracle:
            checks[name] = comparison(
                name,
                actual[name],
                oracle[name],
                tolerance="gradient" if name == "gradient" else "forward",
            )
            checks[name + "_oracle"] = dict(
                passed=bool(np.allclose(double[name], oracle[name], atol=1e-10, rtol=1e-10))
            )
            checks[name + "_repeat"] = dict(
                passed=actual[name].dtype == repeat[name].dtype
                and actual[name].tobytes() == repeat[name].tobytes()
            )
        if not args.forward_only:
            for name, (start, count) in spec.PARAMS.items():
                checks["gradient/" + name] = comparison(
                    name,
                    actual["gradient"][start : start + count],
                    oracle["gradient"][start : start + count],
                    tolerance="gradient",
                )
        arrays = dict(tokens=tokens, targets=targets)
        for label, values in (
            ("actual", actual),
            ("repeat", repeat),
            ("numpy64", oracle),
            ("torch64", double),
        ):
            arrays.update({label + "_" + name: value for name, value in values.items()})
        raw = args.output / "arrays.npz"
        stored = save_arrays(raw, arrays, args.raw_store)
        passed = all(c["passed"] for c in checks.values())
        record.update(
            complete=True,
            passed=passed,
            cases=[
                dict(
                    case=case,
                    first_sequence=first,
                    raw=raw.name,
                    raw_sha256=stored["sha256"],
                    raw_bytes=stored["bytes"],
                    checks=checks,
                    passed=passed,
                )
            ],
            runtime_origins_after=runtime.verify_torch_origins(torch, auth),
        )
    except BaseException:
        record.update(complete=False, passed=False, error=traceback.format_exc())
    finally:
        if auth is not None:
            try:
                record["runtime_postflight"] = runtime.postflight(auth)
            except BaseException:
                record.update(complete=False, passed=False, postflight_error=traceback.format_exc())
        if any(sha(ROOT / path) != digest for path, digest in sources.items()):
            record.update(complete=False, passed=False, producer_error="Candidate source changed")
        with (args.output / "result.json").open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return 0 if record["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "evidence-root", "inventory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--width", type=int, choices=(512, 1024), required=True)
    parser.add_argument("--engine", choices=("mps", "mlx"), required=True)
    parser.add_argument("--policy", type=int, choices=range(8), required=True)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--raw-store", type=Path)
    parser.add_argument("--forward-only", action="store_true")
    raise SystemExit(run(parser.parse_args()))
