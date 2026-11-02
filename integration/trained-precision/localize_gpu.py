"""Same-input local errors and propagated prefix errors on retained failing cases."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    "x",
    "n1",
    "qkv",
    "attention",
    "projected",
    "r1",
    "n2",
    "ff1",
    "relu",
    "ff2",
    "r2",
    "n3",
    "logits",
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pipeline(flat, tokens, overrides):
    import numpy as np

    from transformer import spec

    p = {
        name: flat[start : start + size].reshape(spec.PARAM_SHAPES[name]).astype("f8")
        for name, (start, size) in spec.PARAMS.items()
    }
    trace = {}

    def keep(name, value):
        trace[name] = np.asarray(overrides.get(name, value), "f8")
        return trace[name]

    def norm(x, index):
        centered = x - x.mean(-1, keepdims=True)
        normalized = centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + 1e-5)
        return normalized * p[f"ln{index}_weight"] + p[f"ln{index}_bias"]

    def linear(x, name):
        return x @ p[name + "_weight"].T + p[name + "_bias"]

    b, t = tokens.shape
    x = keep("x", p["token"][tokens] + p["position"][:t])
    n1 = keep("n1", norm(x, 1))
    qkv = keep("qkv", linear(n1, "qkv"))
    q, k, v = [
        a.reshape(b, t, 4, spec.WIDTH // 4).transpose(0, 2, 1, 3) for a in np.split(qkv, 3, -1)
    ]
    score = q @ k.swapaxes(-1, -2) / np.sqrt(spec.WIDTH // 4)
    score = np.where(np.tril(np.ones((t, t), bool)), score, -np.inf)
    prob = np.exp(score - score.max(-1, keepdims=True))
    prob /= prob.sum(-1, keepdims=True)
    attention = keep("attention", (prob @ v).transpose(0, 2, 1, 3).reshape(b, t, spec.WIDTH))
    projected = keep("projected", linear(attention, "attention"))
    r1 = keep("r1", x + projected)
    n2 = keep("n2", norm(r1, 2))
    ff1 = keep("ff1", linear(n2, "ff1"))
    relu = keep("relu", np.maximum(ff1, 0))
    ff2 = keep("ff2", linear(relu, "ff2"))
    r2 = keep("r2", r1 + ff2)
    n3 = keep("n3", norm(r2, 3))
    keep("logits", linear(n3, "head"))
    return trace


def worker(args):
    runtime = load("localization_runtime", ROOT / "integration/gpu-owned/runtime_probe.py")
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(
        engine=args.engine,
        width=args.width,
        case=args.case,
        complete=False,
        formal_eligible=False,
        admission_stream_opened=False,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    auth = None
    try:
        if os.environ.get("ARM64_TRANSFORMER_WIDTH") != str(args.width):
            raise ValueError("Width must be set before imports")
        auth = runtime.bootstrap(args.workspace)
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch

        from transformer.precision_audit import comparison
        from transformer.reference import Transformer
        from transformer.trained_precision import (
            checked_file,
            digest,
            source_inventory,
            torch_evaluate,
        )

        before = source_inventory()
        if digest(args.inventory) != args.inventory_sha256:
            raise ValueError("Inventory changed")
        plan = json.loads(args.inventory.read_text())
        case = next(c for c in plan["cases"] if c["id"] == args.case)
        if case["width"] != args.width:
            raise ValueError("Case width differs")
        flat = np.fromfile(
            checked_file(args.evidence_root, case["checkpoint"], case["checkpoint_sha256"]), "f4"
        )
        rows = np.fromfile(
            checked_file(args.evidence_root, case["rows"], case["rows_sha256"]), "i4"
        ).reshape(-1, 33)
        first = case["sequence"] // 16 * 16
        batch = rows[first : first + 16]
        tokens, targets = np.ascontiguousarray(batch[:, :-1]), np.ascontiguousarray(batch[:, 1:])
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        record.update(
            runtime=runtime.public_auth(auth),
            source_sha256=before,
            runtime_origins=runtime.verify_torch_origins(torch, auth),
        )
        backend = None
        try:
            if args.engine == "mps":
                model = Transformer().load_flat(flat).to("mps")
                x = torch.tensor(tokens.astype("i8"), device="mps")
                with torch.no_grad():
                    ordinary = model(x).cpu().numpy().copy()
                    traced, values = model(x, return_trace=True)
                    actual = {k: values[k].cpu().numpy().copy() for k in STAGES}
            else:
                module = __import__("transformer.gpu_" + args.engine, fromlist=["Backend"])
                if args.engine == "metal":
                    if digest(args.library) != args.library_sha256:
                        raise ValueError("Metal binary changed")
                    module.LIBRARY = args.library
                else:
                    record["mlx_origins"] = runtime.mlx_probe(auth)
                backend = module.Backend(flat, max_batch=16, max_length=32)
                device = backend.upload_inputs(tokens, targets)
                ordinary = backend.to_numpy(backend.forward(device)).copy()
                if args.engine == "metal":
                    values = backend.trace(device)
                else:
                    _, values = backend.forward(device, return_trace=True)
                actual = {k: backend.to_numpy(values[k]).copy() for k in STAGES}
            record["trace_matches_ordinary_bytes"] = (
                actual["logits"].tobytes() == ordinary.tobytes()
            )
        finally:
            if backend is not None:
                backend.close()
        expected = pipeline(flat, tokens, {})
        double = torch_evaluate(flat, tokens, targets, dtype=torch.float64)["logits"]
        record["independent_fp64_max_error"] = float(np.max(np.abs(double - expected["logits"])))
        if not np.allclose(double, expected["logits"], atol=1e-10, rtol=1e-10):
            raise ValueError("Independent FP64 localization equations disagree")
        record["final"] = comparison("logits", ordinary, expected["logits"])
        record["stages"] = {}
        arrays = {
            "tokens": tokens,
            "targets": targets,
            "ordinary_logits": ordinary,
            "torch64_logits": double,
        }
        for label, values in (("actual", actual), ("fp64", expected)):
            arrays.update({label + "_" + name: value for name, value in values.items()})
        overrides = {}
        for stage in STAGES:
            same_input = pipeline(flat, tokens, overrides)[stage]
            overrides[stage] = actual[stage]
            suffix = pipeline(flat, tokens, overrides)["logits"]
            record["stages"][stage] = {
                "local_same_input": comparison(stage, actual[stage], same_input),
                "propagated_prefix": comparison(stage, suffix, expected["logits"]),
            }
            arrays["same_input_" + stage] = same_input
            arrays["suffix_" + stage] = suffix
        raw = args.output / "trace.npz"
        np.savez_compressed(raw, **arrays)
        record.update(raw_sha256=digest(raw), complete=True)
        if before != source_inventory():
            raise ValueError("Source changed during localization")
    except BaseException:
        record.update(complete=False, error=traceback.format_exc())
    finally:
        if auth is not None:
            try:
                record["postflight"] = runtime.postflight(auth)
            except BaseException:
                record.update(complete=False, postflight_error=traceback.format_exc())
        if record["producer_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            record.update(complete=False, producer_error="Localization producer changed")
        with (args.output / "result.json").open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return 0 if record["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "evidence-root", "inventory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--engine", choices=("mps", "mlx", "metal"), required=True)
    parser.add_argument("--width", type=int, choices=(512, 1024), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--library-sha256")
    raise SystemExit(worker(parser.parse_args()))
