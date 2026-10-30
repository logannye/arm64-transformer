"""Explicit GPU model diagnostics, kept separate from all comparison arms."""

import argparse
import hashlib
import importlib.util
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
        gradients_checked=True,
        framework_candidate="compensated-framework-model-diagnostic-v1",
        framework_candidate_policy=args.policy,
        framework_compiled=args.compiled,
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
        from transformer.trained_precision import checked_file, source_inventory, torch_evaluate

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
        oracle = numpy_model(flat, tokens, targets, backward=True)
        double = torch_evaluate(flat, tokens, targets, dtype=torch.float64, gradients=True)
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
                    name: torch.tensor(a, device="mps", requires_grad=True)
                    for name, a in parameters.items()
                }
                logits = candidate.forward(p, x, ops, args.policy)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), y.reshape(-1))
                loss.backward()
                gradient = torch.cat([p[name].grad.reshape(-1) for name in spec.PARAMS])
                return {
                    name: value.detach().cpu().numpy().copy()
                    for name, value in dict(logits=logits, loss=loss, gradient=gradient).items()
                }

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

            function = mx.value_and_grad(loss)
            if args.compiled:
                function = mx.compile(function)

            def evaluate():
                (loss, logits), grad = function(p, x, y)
                gradient = mx.concatenate([grad[name].reshape(-1) for name in spec.PARAMS])
                mx.eval(loss, logits, gradient)
                return {
                    name: np.array(value, copy=True)
                    for name, value in dict(logits=logits, loss=loss, gradient=gradient).items()
                }

        actual, repeat = evaluate(), evaluate()
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
        np.savez_compressed(raw, **arrays)
        passed = all(c["passed"] for c in checks.values())
        record.update(
            complete=True,
            passed=passed,
            cases=[
                dict(
                    case=case,
                    first_sequence=first,
                    raw=raw.name,
                    raw_sha256=sha(raw),
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
    parser.add_argument("--policy", type=int, choices=range(4), required=True)
    parser.add_argument("--compiled", action="store_true")
    raise SystemExit(run(parser.parse_args()))
