"""Development replay of exposed trained states; never admission or timing.

Inventory authenticates retained v6 inputs. Workers compare complete execution
with two FP64 equations, retaining raw arrays and every failure. No native build
or package installation is performed. A worker has one width and engine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import sys
import traceback
from pathlib import Path

SCHEMA = "trained-precision-development-v1"
ROOT = Path(__file__).resolve().parents[1]
PYTHON_CANDIDATES = {
    1: "python-fp32-saved-qkv-attention-v1",
    2: "python-fp32-saved-ln1-qkv-attention-v2",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inventory(evidence_root):
    root = Path(evidence_root).resolve(strict=True)
    cases, reports = {}, {}
    for width in (512, 1024):
        relative = f"experiments/width-sweep-v6/width-{width}/summary.json"
        reports[relative] = digest(root / relative)
        summary = json.loads((root / relative).read_text())
        for run in summary["runs"]:
            for category in ("native_fp64", "python_fp64"):
                failures = run["numerical_audit"]["categories"][category]["failed_checks"]
                for name in failures:
                    match = re.search(
                        r"/v6/(test|validation)/sequence_(\d+)/" + category + "$", name
                    )
                    if match is None:
                        raise ValueError("Unexpected non-sequence failure: " + name)
                    split, sequence = match[1], int(match[2])
                    checkpoint_hash = run["artifact_sha256"]["best.f32"]
                    key = (width, checkpoint_hash, split, sequence)
                    if key not in cases:
                        checkpoint = f"results/width-sweep-v6/width-{width}/{run['run']}/best.f32"
                        rows = f"results/inputs-v6/width-{width}/{run['fixture']}/{split}.i32"
                        cases[key] = {
                            "id": f"w{width}-{checkpoint_hash[:16]}-{split}-{sequence}",
                            "width": width,
                            "checkpoint": checkpoint,
                            "checkpoint_sha256": checkpoint_hash,
                            "rows": rows,
                            "rows_sha256": run["fixture_sha256"][split + ".i32"],
                            "split": split,
                            "sequence": sequence,
                            "checkpoint_origin": run["implementation"],
                            "historical_failures": [],
                        }
                    cases[key]["historical_failures"].append(name)
    # Check every artifact once, before serializing an executable inventory.
    files = {**reports}
    for case in cases.values():
        files[case["checkpoint"]] = case["checkpoint_sha256"]
        files[case["rows"]] = case["rows_sha256"]
    for name, expected in files.items():
        if digest(root / name) != expected:
            raise ValueError("Retained artifact changed: " + name)
    return {
        "schema": SCHEMA,
        "scope": "exposed v6 union of native/PyTorch failures; development only",
        "formal_eligible": False,
        "files": files,
        "cases": sorted(cases.values(), key=lambda c: c["id"]),
    }


def checked_file(root, name, expected):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or digest(path) != expected:
        raise ValueError("Changed or escaped evidence: " + name)
    return path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_inventory():
    paths = [*ROOT.glob("transformer/*.py"), *ROOT.glob("native/*.S")]
    paths += [*ROOT.glob("native/metal/*.mm"), *ROOT.glob("native/metal/*.metal")]
    paths += [ROOT / "integration/gpu-owned/runtime_probe.py", ROOT / "native/Makefile"]
    paths += [
        ROOT / "integration/trained-precision/precision_reference.py",
        ROOT / "integration/trained-precision/precision_reference_v2.py",
        ROOT / "integration/trained-precision/zip_evidence.py",
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def save_arrays(path, arrays, store=None):
    """Write original NPZ bytes once, or retain their exact shared members."""
    import numpy as np

    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Raw evidence already exists")
    if store is None:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        return {"sha256": digest(path), "bytes": path.stat().st_size}
    if sum(value.nbytes for value in arrays.values()) > 512 * 1024**2:
        raise ValueError("In-memory raw evidence exceeds 512 MiB")
    storage = load_module(
        "precision_zip_store", ROOT / "integration/trained-precision/zip_evidence.py"
    )
    with io.BytesIO() as stream:
        np.savez_compressed(stream, **arrays)
        store = Path(store)
        store.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(store).free < 2 * 1024**3 + stream.tell():
            raise RuntimeError("Preserve 2 GiB free space plus the complete next archive")
        stream.seek(0)
        identity = hashlib.file_digest(stream, "sha256").hexdigest()
        return storage.pack_stream(stream, identity, store)


def torch_evaluate(
    flat,
    tokens,
    targets,
    *,
    dtype,
    device="cpu",
    gradients=False,
    policy=0,
    candidate=False,
    candidate_version=1,
):
    import numpy as np
    import torch

    from transformer import spec
    from transformer.reference import Transformer

    if candidate:
        if (
            policy
            or device != "cpu"
            or dtype != torch.float32
            or candidate_version not in PYTHON_CANDIDATES
        ):
            raise ValueError("The separate Python candidate requires CPU FP32 and no probe policy")
        filename = (
            "precision_reference_v2.py" if candidate_version == 2 else "precision_reference.py"
        )
        module = load_module(
            "precision_reference", ROOT / "integration/trained-precision" / filename
        )
        if module.IDENTITY != PYTHON_CANDIDATES[candidate_version]:
            raise ValueError("Unexpected Python candidate identity")
        model = module.PrecisionReference()
    elif policy:
        from transformer.torch_precision_probe import PrecisionProbe

        model = PrecisionProbe(policy)
    else:
        model = Transformer()
    model = model.load_flat(flat).to(device=device, dtype=dtype)
    x, y = (torch.tensor(a.astype(np.int64), device=device) for a in (tokens, targets))
    with torch.set_grad_enabled(gradients):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), y.reshape(-1))
        result = {
            "logits": logits.detach().cpu().numpy().copy(),
            "loss": loss.detach().cpu().numpy().copy(),
        }
        if gradients:
            loss.backward()
            result["gradient"] = (
                torch.cat([getattr(model, n).grad.reshape(-1) for n in spec.PARAM_SHAPES])
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
    return result


def native_evaluate(native, flat, tokens, targets, gradients):
    import numpy as np

    from transformer import spec
    from transformer.validate import workspace_tensor

    logits, losses = [], []
    grad = np.zeros(spec.PARAM_COUNT, "f4")
    for x, y in zip(tokens, targets, strict=True):
        work = native.forward(flat, x.copy())
        logits.append(workspace_tensor(work, "logits").copy())
        if gradients:
            loss, grad = native.backward(flat, x.copy(), y.copy(), work, grad)
        else:
            dl = np.empty_like(logits[-1])
            loss = native.call("cross_entropy", logits[-1], y.copy(), dl, spec.CONTEXT, spec.VOCAB)
        losses.append(loss)
    result = {"logits": np.stack(logits), "loss": np.asarray(np.mean(losses))}
    if gradients:
        native.call("scale", grad, len(grad), 1.0 / len(tokens))
        result["gradient"] = grad
    return result


def worker(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": SCHEMA,
        "engine": args.engine,
        "width": args.width,
        "batch_size": args.batch_size,
        "gradients_checked": args.gradients,
        "torch_diagnostic_policy": args.torch_policy,
        "python_candidate": (
            PYTHON_CANDIDATES[args.python_candidate_version] if args.python_candidate else None
        ),
        "formal_eligible": False,
        "complete": False,
        "passed": False,
        "admission_stream_opened": False,
        "cases": [],
        "pid": os.getpid(),
    }
    runtime = auth = None
    try:
        if os.environ.get("ARM64_TRANSFORMER_WIDTH") != str(args.width):
            raise ValueError("Worker width must be set before imports")
        if args.torch_policy and args.engine != "cpu_torch":
            raise ValueError("Counterfactual precision probes are CPU-only")
        if args.python_candidate and (args.engine != "cpu_torch" or args.torch_policy):
            raise ValueError("Separate Python candidate cannot change another engine/probe")
        if not args.python_candidate and args.python_candidate_version != 1:
            raise ValueError("A candidate version requires the separate Python candidate")
        if args.metal_shaders:
            if args.engine != "metal" or digest(args.metal_shaders) != args.metal_shaders_sha256:
                raise ValueError("Alternate Metal source requires an explicit external binding")
            if args.kernel_policy or os.environ.get("ARM64_TRANSFORMER_METAL_POLICY"):
                raise ValueError("Alternate Metal source is restricted to the reference policy")
            receipt["metal_candidate_shader_sha256"] = args.metal_shaders_sha256
        if digest(args.inventory) != args.inventory_sha256:
            raise ValueError("Inventory identity changed")
        plan = json.loads(Path(args.inventory).read_text())
        if plan["schema"] != SCHEMA or plan["formal_eligible"] is not False:
            raise ValueError("Unexpected replay contract")
        before = source_inventory()
        receipt["source_sha256"] = before
        if args.owned_workspace:
            if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
                raise ValueError(
                    "Owned development requires the retained bootstrap's Python -I -S -B"
                )
            runtime = load_module(
                "precision_runtime", ROOT / "integration/gpu-owned/runtime_probe.py"
            )
            auth = runtime.bootstrap(Path(args.owned_workspace))
            receipt["owned_runtime"] = runtime.public_auth(auth)
        # The retained runtime bootstrap exposes its historical project path.
        # These development equations must come from this hash-bound checkout.
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch

        from transformer import spec
        from transformer.gpu_admission_checks import numpy_model
        from transformer.precision_audit import comparison
        from transformer.validate import Native

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        receipt["runtime"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_file": torch.__file__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        }
        if auth is not None:
            receipt["runtime_origins_before"] = {
                "numpy": runtime.verify_numpy_origin(np, auth),
                "torch": runtime.verify_torch_origins(torch, auth),
            }
            if args.engine == "mlx":
                receipt["runtime_origins_before"]["mlx"] = runtime.mlx_probe(auth)
        if Path(sys.modules["transformer"].__file__).resolve().parent != ROOT / "transformer":
            raise ValueError("Wrong project package origin")
        native = None
        if args.engine == "native":
            if not args.library or digest(args.library) != args.library_sha256:
                raise ValueError("Native binary must have an external SHA256 binding")
            native = Native(args.library)
            import ctypes

            policy = getattr(native.lib, "tf_precision_policy", None)
            if policy:
                policy.argtypes, policy.restype = [], ctypes.c_uint64
            receipt["native"] = {
                "library": str(Path(args.library).resolve()),
                "sha256": args.library_sha256,
                "precision_policy": int(policy()) if policy else 0,
                "backend": native.backend_identity,
            }
        root = Path(args.evidence_root).resolve(strict=True)
        cases = [case for case in plan["cases"] if case["width"] == args.width]
        if args.case_offset < 0 or (args.case_limit is not None and args.case_limit <= 0):
            raise ValueError("Invalid replay slice")
        cases = cases[args.case_offset :]
        if args.case_limit is not None:
            cases = cases[: args.case_limit]
        if not cases:
            raise ValueError("Empty replay inventory")
        receipt["inventory_sha256"] = args.inventory_sha256
        receipt["expected_cases"] = [c["id"] for c in cases]
        receipt["full_width_inventory"] = args.case_limit is None and args.case_offset == 0
        raw_total = 0
        for case in cases:
            checkpoint = checked_file(root, case["checkpoint"], case["checkpoint_sha256"])
            rows_path = checked_file(root, case["rows"], case["rows_sha256"])
            flat = np.fromfile(checkpoint, "f4")
            if flat.shape != (spec.PARAM_COUNT,) or not np.isfinite(flat).all():
                raise ValueError("Invalid checkpoint")
            rows = np.fromfile(rows_path, "i4").reshape(-1, 33)
            first = case["sequence"] // args.batch_size * args.batch_size
            batch = rows[first : first + args.batch_size]
            if len(batch) != args.batch_size:
                raise ValueError("Partial replay batch")
            tokens, targets = (
                np.ascontiguousarray(batch[:, :-1]),
                np.ascontiguousarray(batch[:, 1:]),
            )
            oracle = numpy_model(flat, tokens, targets, backward=args.gradients)
            double = torch_evaluate(
                flat, tokens, targets, dtype=torch.float64, gradients=args.gradients
            )
            backend = None
            try:
                if args.engine in ("metal", "mlx"):
                    import importlib

                    module = importlib.import_module("transformer.gpu_" + args.engine)
                    if args.engine == "metal":
                        if not args.library or digest(args.library) != args.library_sha256:
                            raise ValueError("Metal bridge needs an external binary binding")
                        module.LIBRARY = Path(args.library)
                        if args.metal_shaders:
                            module.SHADERS = Path(args.metal_shaders)
                    policy = (
                        json.loads(Path(args.kernel_policy).read_text())
                        if args.kernel_policy
                        else None
                    )
                    backend = module.Backend(
                        flat,
                        max_batch=args.batch_size,
                        max_length=32,
                        **({"kernel_policy": policy} if args.engine == "metal" else {}),
                    )
                    if args.engine == "metal":
                        receipt["metal"] = backend.capabilities()
                    device_batch = backend.upload_inputs(tokens, targets)

                def evaluate():
                    if native is not None:
                        return native_evaluate(native, flat, tokens, targets, args.gradients)
                    if backend is not None:
                        result = {"logits": backend.to_numpy(backend.forward(device_batch)).copy()}
                        if args.gradients:
                            loss, gradient = backend.loss_and_grad(device_batch)
                            result.update(
                                loss=backend.to_numpy(loss).copy(),
                                gradient=backend.to_numpy(gradient).copy(),
                            )
                        else:
                            result["loss"] = backend.to_numpy(backend.loss(device_batch)).copy()
                        return result
                    return torch_evaluate(
                        flat,
                        tokens,
                        targets,
                        dtype=torch.float32,
                        device="mps" if args.engine == "mps" else "cpu",
                        gradients=args.gradients,
                        policy=args.torch_policy,
                        candidate=args.python_candidate,
                        candidate_version=args.python_candidate_version,
                    )

                actual, repeat = evaluate(), evaluate()
            finally:
                if backend is not None:
                    backend.close()
            checks = {}
            for name in oracle:
                checks[name] = comparison(
                    name,
                    actual[name],
                    oracle[name],
                    tolerance="gradient" if name == "gradient" else "forward",
                )
                checks[name + "_oracle"] = {
                    "passed": bool(np.allclose(double[name], oracle[name], atol=1e-10, rtol=1e-10)),
                    "max_absolute_error": float(np.max(np.abs(double[name] - oracle[name]))),
                }
                checks[name + "_repeat"] = {
                    "passed": actual[name].dtype == repeat[name].dtype
                    and actual[name].shape == repeat[name].shape
                    and actual[name].tobytes() == repeat[name].tobytes()
                }
            if args.gradients:
                for name, (start, size) in spec.PARAMS.items():
                    checks["gradient/" + name] = comparison(
                        name,
                        actual["gradient"][start : start + size],
                        oracle["gradient"][start : start + size],
                        tolerance="gradient",
                    )
            raw = output / (case["id"] + ".npz")
            arrays = {"tokens": tokens, "targets": targets}
            for label, values in (
                ("actual", actual),
                ("repeat", repeat),
                ("numpy64", oracle),
                ("torch64", double),
            ):
                arrays.update({label + "_" + name: value for name, value in values.items()})
            if raw_total + sum(a.nbytes for a in arrays.values()) > 4 * 1024**3:
                raise RuntimeError(
                    "Worker raw-evidence budget exceeded; remaining cases unexecuted"
                )
            stored = save_arrays(raw, arrays, args.raw_store)
            raw_total += stored["bytes"]
            record = {
                "case": case,
                "first_sequence": first,
                "checks": checks,
                "raw": raw.name,
                "raw_sha256": stored["sha256"],
                "raw_bytes": stored["bytes"],
                "passed": all(c["passed"] for c in checks.values()),
            }
            receipt["cases"].append(record)
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "passed": record["passed"],
                        "failed": [k for k, v in checks.items() if not v["passed"]],
                    }
                ),
                flush=True,
            )
        if before != source_inventory():
            raise ValueError("Source changed during execution")
        if digest(args.inventory) != args.inventory_sha256:
            raise ValueError("Inventory changed during execution")
        if args.metal_shaders and digest(args.metal_shaders) != args.metal_shaders_sha256:
            raise ValueError("Alternate Metal source changed during execution")
        if native is not None:
            before_backend = receipt["native"]["backend"]
            after_backend = native.backend_identity
            receipt["native"]["backend_after"] = after_backend
            selected = "sme" if before_backend["selected_kind"] == 1 else "neon"
            count = (5 - bool(receipt["native"]["precision_policy"] & 2)) * 2
            forward_calls = count * len(cases) * args.batch_size
            expected_dispatch = {
                f"{backend}_{direction}_calls": (
                    forward_calls
                    if backend == selected and (direction == "forward" or args.gradients)
                    else 0
                )
                for backend in ("sme", "neon")
                for direction in ("forward", "backward")
            }
            if any(after_backend[k] - before_backend[k] != n for k, n in expected_dispatch.items()):
                raise ValueError("Actual native dispatch counts differ from the declared policy")
            receipt["native"]["expected_dispatch_delta"] = expected_dispatch
        receipt["complete"] = True
        receipt["passed"] = all(c["passed"] for c in receipt["cases"])
        if auth is not None:
            receipt["runtime_origins_after"] = {
                "numpy": runtime.verify_numpy_origin(np, auth),
                "torch": runtime.verify_torch_origins(torch, auth),
            }
            if args.engine == "mlx":
                receipt["runtime_origins_after"]["mlx"] = runtime.mlx_probe(auth)
    except BaseException:
        receipt["error"] = traceback.format_exc()
        receipt["passed"] = False
        receipt["complete"] = False
    finally:
        if auth is not None:
            try:
                receipt["runtime_postflight"] = runtime.postflight(auth)
            except BaseException:
                receipt["postflight_error"] = traceback.format_exc()
                receipt["passed"] = False
        write_new(output / "result.json", receipt)
    return 0 if receipt["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("inventory")
    plan.add_argument("--evidence-root", required=True)
    plan.add_argument("--output", required=True)
    run = sub.add_parser("worker")
    run.add_argument("--evidence-root", required=True)
    run.add_argument("--inventory", required=True)
    run.add_argument("--inventory-sha256", required=True)
    run.add_argument(
        "--engine", choices=("native", "cpu_torch", "mps", "metal", "mlx"), required=True
    )
    run.add_argument("--width", type=int, choices=(512, 1024), required=True)
    run.add_argument("--batch-size", type=int, choices=(1, 16), default=1)
    run.add_argument("--library")
    run.add_argument("--library-sha256")
    run.add_argument("--kernel-policy")
    run.add_argument("--owned-workspace")
    run.add_argument("--gradients", action="store_true")
    run.add_argument("--torch-policy", type=int, choices=range(8), default=0)
    run.add_argument("--python-candidate", action="store_true")
    run.add_argument("--python-candidate-version", type=int, choices=(1, 2), default=1)
    run.add_argument("--metal-shaders")
    run.add_argument("--metal-shaders-sha256")
    run.add_argument("--case-limit", type=int)
    run.add_argument("--case-offset", type=int, default=0)
    run.add_argument("--output", required=True)
    run.add_argument("--raw-store", help="Retain exact NPZ members without a duplicate archive")
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(args.evidence_root)
        write_new(args.output, result)
        print(json.dumps({"cases": len(result["cases"]), "sha256": digest(args.output)}))
        return 0
    return worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
