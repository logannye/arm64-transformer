"""Locate fresh-state branch sensitivity without changing the model or tolerances."""

import argparse
import ctypes
import importlib.util
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


def frozen_gate_relu(torch, mask):
    """Explicit counterfactual: original forward, observed foreign backward gate."""

    class FrozenGate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            if value.shape != mask.shape or mask.dtype != torch.bool:
                raise ValueError("Gate must match the complete activation tensor")
            ctx.save_for_backward(mask)
            return value.clamp_min(0)

        @staticmethod
        def backward(ctx, upstream):
            (gate,) = ctx.saved_tensors
            return upstream * gate

    return FrozenGate.apply


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema="trained-relu-localization-v1",
        complete=False,
        formal_eligible=False,
        admission_stream_opened=False,
        case=args.case,
    )
    try:
        if os.environ.get("ARM64_TRANSFORMER_WIDTH") != str(args.width):
            raise ValueError("Width must be set before imports")
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch

        from transformer import spec
        from transformer.precision_audit import comparison
        from transformer.trained_precision import (
            checked_file,
            digest,
            native_evaluate,
            save_arrays,
            source_inventory,
        )
        from transformer.validate import Native, workspace_tensor

        sources = source_inventory()
        for name in (
            "relu_localize.py",
            "localize_gpu.py",
            "framework_reference.py",
            "compensated_attention.py",
        ):
            path = Path(__file__).with_name(name)
            sources[str(path.relative_to(ROOT))] = digest(path)
        result["source_sha256"] = sources
        if (
            digest(args.receipt) != args.receipt_sha256
            or digest(args.inventory) != args.inventory_sha256
        ):
            raise ValueError("External input authority changed")
        replay = json.loads(args.receipt.read_text())
        plan = json.loads(args.inventory.read_text())
        row = next(c for c in replay["cases"] if c["case"]["id"] == args.case)
        case = next(c for c in plan["cases"] if c["id"] == args.case)
        if not (
            replay["complete"] is True
            and replay["engine"] == "native"
            and replay["gradients_checked"] is True
            and replay["formal_eligible"] is False
            and replay["inventory_sha256"] == args.inventory_sha256
            and row["case"] == case
            and case["width"] == args.width
            and replay["batch_size"] == 16
            and replay["native"]["sha256"] == args.library_sha256 == digest(args.library)
        ):
            raise ValueError("Not the bound native development replay")
        result.update(
            receipt_sha256=args.receipt_sha256,
            inventory_sha256=args.inventory_sha256,
            library_sha256=args.library_sha256,
            width=args.width,
            runtime=dict(numpy=np.__version__, torch=torch.__version__, python=sys.version),
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
        audit = load("relu_raw_audit", Path(__file__).with_name("audit_replay.py"))
        with (
            audit.raw_path(
                args.receipt.parent, row["raw"], row["raw_sha256"], args.raw_store
            ) as raw,
            np.load(raw, allow_pickle=False) as arrays,
        ):
            if not (
                np.array_equal(tokens, arrays["tokens"])
                and np.array_equal(targets, arrays["targets"])
            ):
                raise ValueError("Raw inputs differ")
            retained = {name: arrays["actual_" + name] for name in ("logits", "loss", "gradient")}
            oracle = {name: arrays["numpy64_" + name] for name in ("logits", "loss", "gradient")}
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        native = Native(args.library)
        fn = native.lib.tf_precision_policy
        fn.argtypes, fn.restype = [], ctypes.c_uint64
        if fn() != replay["native"]["precision_policy"]:
            raise ValueError("Native policy differs")
        result["native_policy"] = int(fn())
        result["backend_before"] = native.backend_identity
        actual = native_evaluate(native, flat, tokens, targets, True)
        result["reproduces_retained_bytes"] = all(
            actual[n].dtype == retained[n].dtype and actual[n].tobytes() == retained[n].tobytes()
            for n in actual
        )
        if not result["reproduces_retained_bytes"]:
            raise ValueError("Native replay no longer reproduces retained bytes")
        localization = load("relu_pipeline", Path(__file__).with_name("localize_gpu.py"))
        trace = {name: [] for name in localization.STAGES}
        for token_row in tokens:
            workspace = native.forward(flat, token_row.copy())
            for name in trace:
                trace[name].append(workspace_tensor(workspace, name).copy())
        trace = {name: np.stack(values) for name, values in trace.items()}
        if trace["logits"].tobytes() != actual["logits"].tobytes():
            raise ValueError("Trace differs from ordinary execution")
        reference = localization.pipeline(flat, tokens, {})
        expected_gate, actual_gate = reference["ff1"] > 0, trace["ff1"] > 0
        coordinates = np.argwhere(expected_gate != actual_gate)
        result["gate_disagreements"] = len(coordinates)
        result["gate_examples"] = [
            dict(
                index=c.tolist(),
                native=float(trace["ff1"][tuple(c)]),
                fp64=float(reference["ff1"][tuple(c)]),
            )
            for c in coordinates[:16]
        ]
        overrides, prefixes = {}, {}
        saved = dict(
            tokens=tokens,
            targets=targets,
            native_ff1=trace["ff1"],
            fp64_ff1=reference["ff1"],
            native_gradient=actual["gradient"],
            fp64_gradient=oracle["gradient"],
        )
        for name in localization.STAGES[:7]:
            overrides[name] = trace[name]
            preactivation = localization.pipeline(flat, tokens, overrides)["ff1"]
            prefixes[name] = dict(
                gate_disagreements=int(np.count_nonzero((preactivation > 0) != expected_gate)),
                preactivation=comparison("ff1", preactivation, reference["ff1"]),
            )
            saved["prefix_" + name + "_ff1"] = preactivation
        result["prefixes"] = prefixes
        model = load("relu_model", Path(__file__).with_name("framework_reference.py"))
        operations = model.torch_ops(torch, "cpu")

        def gradient(mask=None):
            parameters = {
                name: torch.tensor(
                    flat[start : start + count].reshape(spec.PARAM_SHAPES[name]),
                    dtype=torch.float64,
                    requires_grad=True,
                )
                for name, (start, count) in spec.PARAMS.items()
            }
            operations.relu = (
                torch.nn.functional.relu
                if mask is None
                else frozen_gate_relu(torch, torch.tensor(mask))
            )
            logits = model.forward(parameters, torch.tensor(tokens.astype("i8")), operations, 0)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 128), torch.tensor(targets.astype("i8")).reshape(-1)
            )
            loss.backward()
            return logits.detach().numpy(), torch.cat(
                [parameters[name].grad.reshape(-1) for name in spec.PARAMS]
            ).numpy()

        logits, double_gradient = gradient()
        if not (
            np.allclose(logits, oracle["logits"], atol=1e-10, rtol=1e-10)
            and np.allclose(double_gradient, oracle["gradient"], atol=1e-10, rtol=1e-10)
        ):
            raise ValueError("Independent FP64 diagnostic differs from retained oracle")
        frozen_logits, frozen_gradient = gradient(actual_gate)
        if not np.array_equal(frozen_logits, logits):
            raise ValueError("Counterfactual changed the forward equation")
        saved["frozen_gate_gradient"] = frozen_gradient
        result["native_vs_original_fp64"] = comparison(
            "gradient", actual["gradient"], oracle["gradient"], tolerance="gradient"
        )
        result["native_vs_fp64_with_native_gate"] = comparison(
            "gradient", actual["gradient"], frozen_gradient, tolerance="gradient"
        )
        result["backend_after"] = native.backend_identity
        stored = save_arrays(args.output / "arrays.npz", saved, args.raw_store)
        result.update(raw_sha256=stored["sha256"], raw_bytes=stored["bytes"], complete=True)
        if (
            any(digest(ROOT / name) != value for name, value in sources.items())
            or digest(args.library) != args.library_sha256
        ):
            raise ValueError("Source or binary changed")
    except BaseException:
        result.update(complete=False, error=traceback.format_exc())
    with (args.output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "inventory", "evidence-root", "library", "raw-store", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("receipt-sha256", "inventory-sha256", "library-sha256", "case"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--width", type=int, choices=(512, 1024), required=True)
    raise SystemExit(run(parser.parse_args()))
