"""Bounded, development-only GPU repeatability diagnostics.

Importing this module does not import a GPU framework. Framework numerical
implementations are unchanged. Exact repetition and FP64 accuracy are distinct
results; this diagnostic never opens formal admission or times an experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

CONTRACT = "gpu-v2-standard-baseline-repeatability-v1"
WIDTHS = (32, 64, 128, 256, 512, 1024)
BATCHES = (1, 3, 16)
LENGTHS = (3, 32)
PATTERNS = ("identical", "alternating", "skewed", "unique")
COTANGENTS = ("zero", "random", "cancellation")
STAGES = ("embedding", "full_model", "training")
REPETITIONS = 10
PROCESSES = 2
SEED = 20261103
PINS = {
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    "PYTORCH_MPS_FAST_MATH": "0",
    "PYTORCH_MPS_PREFER_METAL": "0",
    "MLX_ENABLE_TF32": "0",
    "MLX_DISABLE_COMPILE": "0",
    "MLX_SDPA_VJP_FALLBACK": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
TOLERANCES = {"forward": (2e-5, 2e-4), "gradient": (3e-5, 3e-3), "oracle": (1e-10, 1e-10)}


def array_identity(value):
    value = np.asarray(value)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    }


def exact_comparison(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return {
            "passed": False,
            "reason": "dtype_or_shape",
            "actual": array_identity(actual),
            "expected": array_identity(expected),
        }
    a = np.ascontiguousarray(actual).reshape(-1)
    b = np.ascontiguousarray(expected).reshape(-1)
    different = np.any(
        a.view(np.uint8).reshape(a.size, a.itemsize)
        != b.view(np.uint8).reshape(b.size, b.itemsize),
        axis=1,
    )
    return {
        "passed": not bool(different.any()),
        "different_coordinates": int(different.sum()),
        "actual": array_identity(actual),
        "expected": array_identity(expected),
    }


def accuracy(actual, expected, kind):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {"passed": False, "reason": "shape", "tolerance": kind}
    atol, rtol = TOLERANCES[kind]
    finite = np.isfinite(actual) & np.isfinite(expected)
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    failed = ~finite | (error > atol + rtol * np.abs(expected))
    finite_errors = error[np.isfinite(error)]
    return {
        "passed": not bool(failed.any()),
        "tolerance": kind,
        "atol": atol,
        "rtol": rtol,
        "failed_coordinates": int(failed.sum()),
        "max_absolute_error": float(finite_errors.max(initial=0)),
        "nonfinite_coordinates": int((~finite).sum()),
    }


def tokens_for(batch, length, pattern):
    if batch not in BATCHES or length not in LENGTHS or pattern not in PATTERNS:
        raise ValueError("Unknown declared shape or token pattern")
    n = batch * length
    if pattern == "unique":
        if n > 128:
            raise ValueError(
                "Globally unique tokens are impossible when B*T exceeds vocabulary 128"
            )
        values = np.arange(n, dtype=np.int32)
    elif pattern == "identical":
        values = np.full(n, 65, dtype=np.int32)
    elif pattern == "alternating":
        values = np.resize(np.array([65, 66], dtype=np.int32), n)
    else:
        values = np.resize(np.array([65, 65, 65, 65, 65, 66, 67, 68], dtype=np.int32), n)
    return values.reshape(batch, length)


def embedding_cases(width):
    if width not in WIDTHS:
        raise ValueError("Unsupported width")
    for batch in BATCHES:
        for length in LENGTHS:
            for pattern in PATTERNS:
                for cotangent in COTANGENTS:
                    name = f"embedding/b{batch}-t{length}/{pattern}/{cotangent}"
                    yield {
                        "name": name,
                        "batch": batch,
                        "length": length,
                        "pattern": pattern,
                        "cotangent": cotangent,
                        "supported": pattern != "unique" or batch * length <= 128,
                    }


def expected_case_names(width, stages):
    names = (
        [case["name"] for case in embedding_cases(width) if case["supported"]]
        if "embedding" in stages
        else []
    )
    names += [
        f"{stage}/b{batch}-t{length}"
        for stage in ("full_model", "training")
        if stage in stages
        for batch in BATCHES
        for length in LENGTHS
    ]
    return names


def embedding_operands(width, case):
    tokens = tokens_for(case["batch"], case["length"], case["pattern"])
    key = int.from_bytes(hashlib.sha256(case["name"].encode()).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, width, key]))
    table = rng.normal(0, 0.02, (128, width)).astype(np.float32)
    shape = (*tokens.shape, width)
    if case["cotangent"] == "zero":
        dy = np.zeros(shape, np.float32)
    elif case["cotangent"] == "random":
        dy = rng.normal(0, 0.2, shape).astype(np.float32)
    else:
        # Exact stored FP32 cancellation operands, including small residuals.
        dy = np.resize(np.array([1024, 0.125, -1024, -0.0625], np.float32), tokens.size)[
            :, None
        ] * np.ones((1, width), np.float32)
        dy = dy.reshape(shape)
    return tokens, table, dy


def embedding_oracle(tokens, table, dy):
    result = np.zeros(table.shape, np.float64)
    for token, gradient in zip(tokens.reshape(-1), dy.reshape(-1, table.shape[1])):
        result[token] += gradient.astype(np.float64)
    return {"output": table[tokens].astype(np.float64), "gradient": result}


def make_embedding_runner(engine, tokens, table, dy):
    """Use each baseline's actual lookup expression and ordinary differentiation."""
    if engine == "mps":
        import torch
        from torch.nn import functional as functional

        indices = torch.tensor(tokens, dtype=torch.int64, device="mps")
        cotangent = torch.tensor(dy, device="mps")
        parameter = torch.tensor(table, device="mps", requires_grad=True)

        def run():
            parameter.grad = None
            output = functional.embedding(indices, parameter)
            (output * cotangent).sum().backward()
            torch.mps.synchronize()
            return {
                "output": output.detach().cpu().numpy().copy(),
                "gradient": parameter.grad.detach().cpu().numpy().copy(),
            }

        return run
    if engine == "mlx":
        import mlx.core as mx

        indices, cotangent, parameter = mx.array(tokens), mx.array(dy), mx.array(table)

        def objective(weight):
            output = weight[indices]
            return mx.sum(output * cotangent), output

        operation = mx.compile(mx.value_and_grad(objective))

        def run():
            (_, output), gradient = operation(parameter)
            mx.eval(output, gradient)
            mx.synchronize()
            return {"output": np.array(output), "gradient": np.array(gradient)}

        return run
    raise ValueError("Isolated embedding diagnostics support unchanged MPS/MLX baselines only")


def _save_arrays(path, arrays):
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def repeat_case(name, operation, destination, oracle=None):
    """Retain every fingerprint; first outputs and first divergence in full.

    The first divergent output is retained per array; later discrepancies retain
    exact hashes and complete mismatch counts, bounding duplicated wide states.
    """
    directory = destination / name
    directory.mkdir(parents=True, exist_ok=False)
    result = {"name": name, "executions": [], "complete": True}
    first = None
    saved_failure_names = set()
    for index in range(REPETITIONS):
        try:
            actual = {key: np.asarray(value).copy() for key, value in operation().items()}
            if first is None:
                first = actual
                _save_arrays(directory / "first.npz", first)
            comparisons = {
                key: exact_comparison(value, first[key]) for key, value in actual.items()
            }
            if set(actual) != set(first):
                raise ValueError("Output inventory changed")
            failed = {
                key: value
                for key, value in actual.items()
                if not comparisons[key]["passed"] and key not in saved_failure_names
            }
            if failed:
                _save_arrays(directory / f"first-divergence-{index:02}.npz", failed)
                saved_failure_names.update(failed)
            entry = {
                "execution": index,
                "exact": comparisons,
                "finite": all(bool(np.isfinite(value).all()) for value in actual.values()),
            }
            if oracle is not None:
                entry["accuracy"] = oracle(actual)
            result["executions"].append(entry)
        except Exception as exc:
            result["executions"].append(
                {
                    "execution": index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            result["complete"] = False
            # Repeating the same rejected shape ten times adds no evidence.
            result["unexecuted_repetitions"] = list(range(index + 1, REPETITIONS))
            break
        finally:
            _write_json(directory / "progress.json", result)
    result["exact_passed"] = result["complete"] and all(
        entry["finite"] and all(value["passed"] for value in entry["exact"].values())
        for entry in result["executions"]
    )
    result["accuracy_passed"] = result["complete"] and all(
        all(value["passed"] for value in entry.get("accuracy", {}).values())
        for entry in result["executions"]
    )
    result["first_arrays"] = (
        str((directory / "first.npz").relative_to(destination)) if first else None
    )
    _write_json(directory / "result.json", result)
    return result


def _torch_reference(initial, tokens, targets, *, backward):
    import torch
    from torch.nn import functional as functional

    from transformer.reference import Transformer

    model = Transformer().load_flat(initial).double()
    x = torch.from_numpy(tokens.astype(np.int64))
    y = torch.from_numpy(targets.astype(np.int64))
    logits = model(x)
    loss = functional.cross_entropy(logits.reshape(-1, 128), y.reshape(-1))
    result = {"logits": logits.detach().numpy(), "loss": loss.detach().numpy()}
    if backward:
        loss.backward()
        result["gradient"] = torch.cat([p.grad.reshape(-1) for p in model.parameters()]).numpy()
    return result


def _numpy_forward(initial, tokens):
    """Independent FP64 equations, including partial sequence lengths."""
    from transformer import spec

    p = {
        name: initial[start : start + count].reshape(spec.PARAM_SHAPES[name]).astype(np.float64)
        for name, (start, count) in spec.PARAMS.items()
    }
    batch, length = tokens.shape

    def norm(x, i):
        centered = x - x.mean(-1, keepdims=True)
        return (
            centered
            / np.sqrt((centered * centered).mean(-1, keepdims=True) + spec.EPS)
            * p[f"ln{i}_weight"]
            + p[f"ln{i}_bias"]
        )

    def linear(x, name):
        return x @ p[name + "weight"].T + p[name + "bias"]

    x = p["token"][tokens] + p["position"][:length]
    qkv = linear(norm(x, 1), "qkv_")
    q, k, v = [
        part.reshape(batch, length, 4, spec.WIDTH // 4).transpose(0, 2, 1, 3)
        for part in np.split(qkv, 3, axis=-1)
    ]
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(spec.WIDTH // 4)
    scores = np.where(np.tril(np.ones((length, length), bool)), scores, -np.inf)
    probabilities = np.exp(scores - scores.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    attention = (probabilities @ v).transpose(0, 2, 1, 3).reshape(x.shape)
    r1 = x + linear(attention, "attention_")
    r2 = r1 + linear(np.maximum(linear(norm(r1, 2), "ff1_"), 0), "ff2_")
    return linear(norm(r2, 3), "head_")


def source_identity(engine=None):
    root = Path(__file__).resolve().parents[1]
    names = [
        "transformer/gpu_repeatability.py",
        "scripts/check_gpu_repeatability.py",
        "transformer/reference.py",
        "transformer/spec.py",
    ]
    engines = ("mps", "mlx", "metal") if engine is None else (engine,)
    names.extend(f"transformer/gpu_{name}.py" for name in engines)
    if "metal" in engines:
        names.extend(
            str(path.relative_to(root)) for path in (root / "native/metal").glob("*.metal")
        )
        names.append("native/metal/backend.mm")
        if (root / "transformer/gpu_policy.py").exists():
            names.append("transformer/gpu_policy.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def run_worker(engine, width, destination, stages, runtime_identity_sha256=None):
    from transformer import spec

    if width != spec.WIDTH:
        raise ValueError("Width must be selected before Python starts")
    runtime = importlib.import_module("transformer.gpu_" + engine)
    import torch

    torch.set_num_threads(1)
    from transformer.reference import Transformer

    initial = Transformer(seed=SEED).flat_parameters()
    backend = runtime.Backend(initial)
    report = {
        "contract": CONTRACT,
        "scope": "development_only",
        "engine": engine,
        "width": width,
        "stages": stages,
        "repetitions": REPETITIONS,
        "initial": array_identity(initial),
        "source_sha256": source_identity(engine),
        "runtime_identity_sha256": runtime_identity_sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {key: os.environ.get(key) for key in PINS},
        "capabilities": backend.capabilities(),
        "cases": [],
        "unsupported_controls": [],
        "cleanup_errors": [],
        "expected_cases": expected_case_names(width, stages),
    }
    initial_state = backend.export_state()
    _save_arrays(destination / "initial-state.npz", initial_state)

    def retain(result):
        nonlocal backend
        report["cases"].append(result)
        if not result["complete"]:
            recovery = {"attempted": True, "fresh_backend": False}
            try:
                backend.close()
            except Exception as error:
                recovery["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
                report["cleanup_errors"].append(
                    {"case": result["name"], **recovery["cleanup_error"]}
                )
            result["recovery"] = recovery
            # A rejected lazy graph can poison its backend's pending queue. A
            # fresh ordinary backend isolates the next declared case; its failed
            # predecessor and skipped repetitions remain explicit failures.
            backend = None
            backend = runtime.Backend(initial)
            backend.load_state(initial_state)
            recovery["fresh_backend"] = True
            _write_json(destination / result["name"] / "result.json", result)
        _write_json(destination / "worker.json", report)
        print(
            json.dumps(
                {
                    "case": result["name"],
                    "exact": result["exact_passed"],
                    "accuracy": result["accuracy_passed"],
                    "complete": result["complete"],
                }
            ),
            flush=True,
        )

    try:
        if "embedding" in stages:
            if engine == "metal":
                raise ValueError(
                    "Metal has no isolated embedding probe; select full_model/training explicitly"
                )
            for case in embedding_cases(width):
                if not case["supported"]:
                    report["unsupported_controls"].append(
                        {**case, "reason": "B*T exceeds vocabulary 128"}
                    )
                    continue
                tokens, table, dy = embedding_operands(width, case)
                gold = embedding_oracle(tokens, table, dy)
                weight = torch.tensor(table, dtype=torch.float64, requires_grad=True)
                output = torch.nn.functional.embedding(
                    torch.from_numpy(tokens.astype(np.int64)), weight
                )
                (output * torch.tensor(dy, dtype=torch.float64)).sum().backward()
                independent = accuracy(weight.grad.numpy(), gold["gradient"], "oracle")

                def oracle(actual, expected=gold, independent=independent):
                    return {
                        "output": accuracy(actual["output"], expected["output"], "forward"),
                        "gradient": accuracy(actual["gradient"], expected["gradient"], "gradient"),
                        "independent_embedding_oracle": independent,
                    }

                # Construction itself may reject a supported framework shape.
                operation = None

                def embedding_run():
                    nonlocal operation
                    if operation is None:
                        operation = make_embedding_runner(engine, tokens, table, dy)
                    return operation()

                result = repeat_case(case["name"], embedding_run, destination, oracle)
                result["fixture"] = {
                    "tokens": array_identity(tokens),
                    "table": array_identity(table),
                    "cotangent": array_identity(dy),
                }
                _save_arrays(
                    destination / case["name"] / "fixture.npz",
                    {"tokens": tokens, "table": table, "cotangent": dy},
                )
                retain(result)
        for batch in BATCHES:
            for length in LENGTHS:
                tokens = tokens_for(batch, length, "skewed")
                targets = np.roll(tokens, -1, axis=1).copy()

                def restore():
                    backend.load_state(initial_state)
                    restored = backend.export_state()
                    if not all(
                        exact_comparison(restored[key], initial_state[key])["passed"]
                        for key in initial_state
                    ):
                        raise RuntimeError("Exact initial-state restoration failed")

                if "full_model" in stages:
                    gold = _torch_reference(initial, tokens, targets, backward=True)
                    oracle_check = accuracy(
                        gold["logits"], _numpy_forward(initial, tokens), "oracle"
                    )

                    def full_model():
                        restore()
                        device_batch = backend.upload_inputs(tokens, targets)
                        logits = backend.to_numpy(backend.forward(device_batch)).copy()
                        loss, gradient = backend.loss_and_grad(device_batch)
                        result = {
                            "logits": logits,
                            "loss": backend.to_numpy(loss),
                            "gradient": backend.to_numpy(gradient),
                        }
                        restored = backend.export_state()
                        if not all(
                            exact_comparison(restored[key], initial_state[key])["passed"]
                            for key in initial_state
                        ):
                            raise RuntimeError(
                                "Forward/backward changed parameters or optimizer state"
                            )
                        return result

                    def full_oracle(actual):
                        return {
                            **{
                                key: accuracy(
                                    actual[key],
                                    gold[key],
                                    "gradient" if key == "gradient" else "forward",
                                )
                                for key in gold
                            },
                            "independent_forward_oracle": oracle_check,
                        }

                    result = repeat_case(
                        f"full_model/b{batch}-t{length}", full_model, destination, full_oracle
                    )
                    result["fixture"] = {
                        "tokens": array_identity(tokens),
                        "targets": array_identity(targets),
                    }
                    retain(result)
                if "training" in stages:

                    def training():
                        restore()
                        device_batch = backend.upload_inputs(tokens, targets)
                        losses = []
                        for _ in range(6):
                            losses.append(float(backend.to_numpy(backend.train_step(device_batch))))
                            if not backend.check_finite():
                                raise RuntimeError("Nonfinite state during six-step training")
                        state = backend.export_state()
                        logits = backend.to_numpy(backend.forward(device_batch)).copy()
                        # Exercise exact final-checkpoint restoration and replay in each execution.
                        backend.load_state(state)
                        replay = backend.to_numpy(backend.forward(device_batch)).copy()
                        if not exact_comparison(replay, logits)["passed"]:
                            raise RuntimeError("Final checkpoint forward replay changed")
                        return {**state, "logits": logits, "losses": np.asarray(losses, np.float32)}

                    def checkpoint_oracle(actual):
                        expected = _torch_reference(
                            actual["parameters"], tokens, targets, backward=False
                        )
                        numpy_logits = _numpy_forward(actual["parameters"], tokens)
                        return {
                            "checkpoint_logits": accuracy(
                                actual["logits"], numpy_logits, "forward"
                            ),
                            "independent_forward_oracle": accuracy(
                                expected["logits"], numpy_logits, "oracle"
                            ),
                        }

                    result = repeat_case(
                        f"training/b{batch}-t{length}", training, destination, checkpoint_oracle
                    )
                    result["fixture"] = {
                        "tokens": array_identity(tokens),
                        "targets": array_identity(targets),
                    }
                    retain(result)
    except Exception as exc:
        report["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                report["cleanup_errors"].append(
                    {
                        "case": "worker_finalization",
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
        report["source_sha256_after"] = source_identity(engine)
        report["source_unchanged"] = report["source_sha256"] == report["source_sha256_after"]
        report["inventory_complete"] = {case["name"] for case in report["cases"]} == set(
            report["expected_cases"]
        )
        report["complete"] = (
            report["inventory_complete"]
            and "fatal_error" not in report
            and not report["cleanup_errors"]
            and all(case["complete"] for case in report["cases"])
        )
        report["exact_passed"] = (
            report["complete"]
            and report["source_unchanged"]
            and all(case["exact_passed"] for case in report["cases"])
        )
        report["accuracy_passed"] = report["complete"] and all(
            case["accuracy_passed"] for case in report["cases"]
        )
        _write_json(destination / "worker.json", report)
    return report


def compare_processes(first, second, destination):
    a = json.loads((first / "worker.json").read_text())
    b = json.loads((second / "worker.json").read_text())
    result = {"source_equal": a["source_sha256"] == b["source_sha256"], "cases": []}
    by_name = {case["name"]: case for case in b["cases"]}
    result["inventory_equal"] = {case["name"] for case in a["cases"]} == set(by_name)
    for case in a["cases"]:
        other = by_name.get(case["name"])
        entry = {"name": case["name"], "passed": False}
        if other and case["first_arrays"] and other["first_arrays"]:
            with (
                np.load(first / case["first_arrays"], allow_pickle=False) as left,
                np.load(second / other["first_arrays"], allow_pickle=False) as right,
            ):
                entry["inventory_equal"] = set(left.files) == set(right.files)
                entry["exact"] = {
                    key: exact_comparison(right[key], left[key])
                    for key in left.files
                    if key in right.files
                }
                entry["passed"] = entry["inventory_equal"] and all(
                    check["passed"] for check in entry["exact"].values()
                )
        else:
            entry["reason"] = "missing_first_execution"
        result["cases"].append(entry)
    result["passed"] = (
        result["source_equal"]
        and result["inventory_equal"]
        and bool(result["cases"])
        and all(case["passed"] for case in result["cases"])
    )
    _write_json(destination, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines", nargs="+", choices=("mps", "mlx", "metal"), default=["mps", "mlx"]
    )
    parser.add_argument("--widths", nargs="+", type=int, choices=WIDTHS, default=list(WIDTHS))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument(
        "--runtime-identity-sha256", help="Prospectively supplied runtime binding hash"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.runtime_identity_sha256 is not None and (
        len(args.runtime_identity_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.runtime_identity_sha256)
    ):
        parser.error("Runtime identity must be a lowercase SHA-256 hex digest")
    if args.inventory:
        print(
            json.dumps(
                {
                    "contract": CONTRACT,
                    "widths": args.widths,
                    "engines": args.engines,
                    "shapes": [[b, t] for b in BATCHES for t in LENGTHS],
                    "embedding_cases_per_width": list(embedding_cases(args.widths[0])),
                    "repetitions_per_process": REPETITIONS,
                    "processes": PROCESSES,
                    "stages": args.stages,
                },
                indent=2,
            )
        )
        return 0
    if args.output is None:
        parser.error("--output is required")
    if "metal" in args.engines and "embedding" in args.stages:
        parser.error("Metal has no isolated embedding probe; select full_model training explicitly")
    if (
        len(args.engines) != len(set(args.engines))
        or len(args.widths) != len(set(args.widths))
        or len(args.stages) != len(set(args.stages))
    ):
        parser.error("Duplicate selections are forbidden")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.worker:
        if len(args.engines) != 1 or len(args.widths) != 1:
            parser.error("A worker owns exactly one engine and width")
        result = run_worker(
            args.engines[0], args.widths[0], args.output, args.stages, args.runtime_identity_sha256
        )
        return 0 if result["exact_passed"] and result["accuracy_passed"] else 1
    root = Path(__file__).resolve().parents[1]
    summary = {
        "contract": CONTRACT,
        "scope": "development_only",
        "engines": args.engines,
        "widths": args.widths,
        "stages": args.stages,
        "workers": [],
        "cross_process": [],
        "source_sha256": source_identity(),
        "runtime_identity_sha256": args.runtime_identity_sha256,
        "full_declared_inventory": args.engines == ["mps", "mlx"]
        and args.widths == list(WIDTHS)
        and args.stages == list(STAGES),
        "retention": (
            "All execution hashes/counts; first output and first divergent output per array "
            "in full; both process first outputs retained"
        ),
    }
    for engine in args.engines:
        for width in args.widths:
            paths = []
            for process in range(PROCESSES):
                destination = args.output / f"{engine}-w{width}-process{process}"
                env = {**os.environ, **PINS, "ARM64_TRANSFORMER_WIDTH": str(width)}
                command = [
                    sys.executable,
                    str(root / "scripts/check_gpu_repeatability.py"),
                    "--worker",
                    "--engines",
                    engine,
                    "--widths",
                    str(width),
                    "--stages",
                    *args.stages,
                    "--output",
                    str(destination.resolve()),
                ]
                if args.runtime_identity_sha256 is not None:
                    command.extend(["--runtime-identity-sha256", args.runtime_identity_sha256])
                with (args.output / f"{engine}-w{width}-process{process}.log").open("x") as log:
                    print(json.dumps({"worker_started": str(destination)}), flush=True)
                    completed = subprocess.run(
                        command,
                        cwd=root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                paths.append(destination)
                record = {
                    "engine": engine,
                    "width": width,
                    "process": process,
                    "returncode": completed.returncode,
                    "result": str((destination / "worker.json").relative_to(args.output)),
                    "complete": False,
                    "exact_passed": False,
                    "accuracy_passed": False,
                }
                if (destination / "worker.json").is_file():
                    try:
                        worker = json.loads((destination / "worker.json").read_text())
                        flags = ("complete", "exact_passed", "accuracy_passed")
                        record.update({key: worker.get(key) is True for key in flags})
                        if any(key not in worker for key in flags):
                            record["incomplete_worker_receipt"] = True
                    except (OSError, ValueError, TypeError) as error:
                        record["receipt_error"] = f"{type(error).__name__}: {error}"
                summary["workers"].append(record)
                print(json.dumps(record), flush=True)
                _write_json(args.output / "summary.json", summary)
            if all((path / "worker.json").is_file() for path in paths):
                cross_path = args.output / f"{engine}-w{width}-cross-process.json"
                try:
                    compared = compare_processes(*paths, cross_path)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    compared = {"passed": False, "error": f"{type(error).__name__}: {error}"}
                    _write_json(cross_path, compared)
                summary["cross_process"].append(
                    {"engine": engine, "width": width, "passed": compared["passed"]}
                )
    summary["complete"] = all(
        worker.get("complete", False) for worker in summary["workers"]
    ) and len(summary["cross_process"]) == len(args.engines) * len(args.widths)
    summary["exact_passed"] = (
        summary["complete"]
        and all(worker["exact_passed"] for worker in summary["workers"])
        and all(case["passed"] for case in summary["cross_process"])
    )
    summary["accuracy_passed"] = summary["complete"] and all(
        worker["accuracy_passed"] for worker in summary["workers"]
    )
    summary["formal_admission_opened"] = False
    _write_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "complete",
                    "exact_passed",
                    "accuracy_passed",
                    "formal_admission_opened",
                )
            }
        ),
        flush=True,
    )
    return 0 if summary["exact_passed"] and summary["accuracy_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
