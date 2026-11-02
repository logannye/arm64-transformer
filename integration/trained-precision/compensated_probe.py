"""Run a bounded same-QKV GPU arithmetic diagnostic in the owned runtime."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
import types
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
    runtime = load("compensated_runtime", ROOT / "integration/gpu-owned/runtime_probe.py")
    candidate_path = Path(__file__).with_name("compensated_attention.py")
    producer_before, candidate_before = sha(__file__), sha(candidate_path)
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema="trained-precision-compensated-attention-diagnostic-v1",
        engine=args.engine,
        complete=False,
        passed=False,
        formal_eligible=False,
        admission_stream_opened=False,
        candidate_sha256=candidate_before,
        producer_sha256=producer_before,
        trace_receipt_sha256=args.trace_receipt_sha256,
        pid=os.getpid(),
    )
    auth = None
    try:
        if sha(args.trace / "result.json") != args.trace_receipt_sha256:
            raise ValueError("Unbound same-input trace")
        trace = json.loads((args.trace / "result.json").read_text())
        if not trace["complete"] or not trace["trace_matches_ordinary_bytes"]:
            raise ValueError("Invalid ordinary-operation trace")
        if trace["engine"] != args.engine or sha(args.trace / "trace.npz") != trace["raw_sha256"]:
            raise ValueError("Trace input changed")
        auth = runtime.bootstrap(args.workspace)
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        record.update(
            runtime=runtime.public_auth(auth),
            origins_before=runtime.verify_torch_origins(torch, auth),
        )
        candidate = load("compensated_attention", candidate_path)
        raw_audit = load("compensated_raw_audit", Path(__file__).with_name("audit_replay.py"))
        with np.load(args.trace / "trace.npz", allow_pickle=False) as saved:
            qkv = saved["actual_qkv"].copy()
        if qkv.dtype != np.dtype("f4") or not np.isfinite(qkv).all() or np.max(np.abs(qkv)) > 1e4:
            raise ValueError("Operands are outside this bounded diagnostic")
        batch, length, projected = qkv.shape
        width = projected // 3
        operands = [
            np.ascontiguousarray(a.reshape(batch, length, 4, width // 4).transpose(0, 2, 1, 3))
            for a in np.split(qkv, 3, axis=-1)
        ]
        rng = np.random.default_rng(np.random.SeedSequence([20261103, width, 91]))
        upstream = rng.standard_normal(operands[0].shape).astype("f4") * np.float32(0.01)
        doubles = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in operands]
        expected = torch.nn.functional.scaled_dot_product_attention(*doubles, is_causal=True)
        (expected * torch.tensor(upstream, dtype=torch.float64)).sum().backward()
        arrays = {"qkv": qkv, "upstream": upstream, "torch64_output": expected.detach().numpy()}
        arrays.update(
            {"torch64_d" + name: value.grad.numpy() for name, value in zip("qkv", doubles)}
        )
        q, k, v = (a.astype("f8") for a in operands)
        score = q @ k.swapaxes(-1, -2) / np.sqrt(width // 4)
        score = np.where(np.tril(np.ones((length, length), bool)), score, -np.inf)
        probability = np.exp(score - score.max(-1, keepdims=True))
        probability /= probability.sum(-1, keepdims=True)
        arrays["numpy64_output"] = probability @ v
        dp = upstream @ v.swapaxes(-1, -2)
        ds = probability * (dp - (dp * probability).sum(-1, keepdims=True)) / np.sqrt(width // 4)
        arrays.update(
            numpy64_dq=ds @ k,
            numpy64_dk=ds.swapaxes(-1, -2) @ q,
            numpy64_dv=probability.swapaxes(-1, -2) @ upstream,
        )
        if args.engine == "mps":
            ops = types.SimpleNamespace(
                concatenate=lambda a, axis: torch.cat(a, dim=axis),
                zeros_like=torch.zeros_like,
                arange=lambda n: torch.arange(n, device="mps"),
                max=lambda a, axis, keepdims: torch.amax(a, dim=axis, keepdim=keepdims),
                where=torch.where,
                exp=torch.exp,
                swapaxes=torch.swapaxes,
            )

            def evaluate():
                inputs = [torch.tensor(a, device="mps", requires_grad=True) for a in operands]
                output, _ = candidate.attention(*inputs, ops)
                (output * torch.tensor(upstream, device="mps")).sum().backward()
                return [
                    a.detach().cpu().numpy().copy() for a in [output, *(x.grad for x in inputs)]
                ]

        else:
            import mlx.core as mx

            record["mlx_origins"] = runtime.mlx_probe(auth)

            def evaluate():
                inputs = [mx.array(a) for a in operands]

                def loss(q, k, v):
                    output, _ = candidate.attention(q, k, v, mx)
                    return (output * mx.array(upstream)).sum(), output

                (_, output), grads = mx.value_and_grad(loss, argnums=(0, 1, 2))(*inputs)
                mx.eval(output, *grads)
                return [np.array(a, copy=True) for a in [output, *grads]]

        actual, repeat = evaluate(), evaluate()
        checks = {}
        for name, value, replay in zip(("output", "dq", "dk", "dv"), actual, repeat, strict=True):
            arrays["actual_" + name], arrays["repeat_" + name] = value, replay
            atol, rtol = (2e-5, 2e-4) if name == "output" else (3e-5, 3e-3)
            checks[name] = raw_audit.compare(value, arrays["numpy64_" + name], atol, rtol)
            checks[name + "_oracle"] = raw_audit.compare(
                arrays["torch64_" + name], arrays["numpy64_" + name], 1e-10, 1e-10
            )
            checks[name + "_repeat"] = dict(passed=value.tobytes() == replay.tobytes())
            if value.dtype != np.dtype("f4"):
                raise ValueError("Diagnostic stored a non-FP32 result")
        raw = args.output / "arrays.npz"
        np.savez_compressed(raw, **arrays)
        record.update(
            complete=True,
            passed=all(c["passed"] for c in checks.values()),
            checks=checks,
            raw_sha256=sha(raw),
            origins_after=runtime.verify_torch_origins(torch, auth),
        )
    except BaseException:
        record.update(complete=False, passed=False, error=traceback.format_exc())
    finally:
        if auth is not None:
            try:
                record["postflight"] = runtime.postflight(auth)
            except BaseException:
                record.update(complete=False, passed=False, postflight_error=traceback.format_exc())
        if producer_before != sha(__file__) or candidate_before != sha(candidate_path):
            record.update(complete=False, passed=False, producer_error="Diagnostic source changed")
        with (args.output / "result.json").open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return 0 if record["complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--trace-receipt-sha256", required=True)
    parser.add_argument("--engine", choices=("mps", "mlx"), required=True)
    raise SystemExit(run(parser.parse_args()))
