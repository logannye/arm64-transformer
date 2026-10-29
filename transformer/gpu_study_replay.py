"""Isolated checkpoint replay; also runs against the immutable CPU source tree.

Invoke this file by absolute path with PYTHONPATH/cwd set to the selected source
tree. No study-module imports are used, so frozen CPU numerical modules stay
authoritative. This command performs no training and is never inside timing.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replay(engine, fixture_path, checkpoint, output):
    import numpy as np
    import torch

    from transformer import spec
    from transformer.prepare import load_fixture

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fixture = load_fixture(fixture_path)
    parameters = np.fromfile(checkpoint, dtype="<f4")
    if parameters.shape != (spec.PARAM_COUNT,) or not np.isfinite(parameters).all():
        raise ValueError("Malformed checkpoint")
    if Path(output).exists():
        raise ValueError("Replay output must be fresh")
    result = {}
    backend = None
    if engine == "cpu_arm64":
        from transformer.validate import Native, workspace_tensor

        native = Native()
        dlogits = np.empty((32, 128), np.float32)
    elif engine in ("cpu_torch", "oracle"):
        from torch.nn import functional as F

        from transformer.reference import Transformer

        model = Transformer().load_flat(parameters)
        if engine == "oracle":
            from transformer.precision_audit import FP64Transformer

            oracle = FP64Transformer(parameters)
            model = model.double()
    else:
        import importlib

        backend = importlib.import_module(f"transformer.gpu_{engine}").Backend(parameters)
    try:
        if engine == "metal":
            from transformer.gpu_policy import bundle, identity

            if backend.capabilities().get("kernel_policy_sha256") != identity(
                bundle(width=spec.WIDTH)
            ):
                raise ValueError("Replay executed an unexpected Metal kernel policy")
        for split in ("validation", "test"):
            rows = fixture[split]
            values = np.empty(
                (len(rows), 32, 128), np.float64 if engine == "oracle" else np.float32
            )
            total = 0.0
            if engine == "cpu_arm64":
                for i, row in enumerate(rows):
                    workspace = native.forward(parameters, np.ascontiguousarray(row[:-1]))
                    values[i] = workspace_tensor(workspace, "logits")
                    total += native.call(
                        "cross_entropy",
                        workspace_tensor(workspace, "logits"),
                        np.ascontiguousarray(row[1:]),
                        dlogits,
                        32,
                        128,
                    )
            else:
                for start in range(0, len(rows), 16):
                    selected = rows[start : start + 16]
                    if engine == "oracle":
                        logits = oracle(selected[:, :-1])
                        loss = 0.0  # Quality is independently evaluated by the auditor.
                    elif engine == "cpu_torch":
                        with torch.no_grad():
                            tensor_logits = model(torch.tensor(selected[:, :-1].astype(np.int64)))
                            loss = F.cross_entropy(
                                tensor_logits.reshape(-1, 128),
                                torch.tensor(selected[:, 1:].astype(np.int64)).reshape(-1),
                            ).item()
                            logits = tensor_logits.numpy()
                    else:
                        batch = backend.upload_inputs(selected[:, :-1], selected[:, 1:])
                        loss_value, device_logits = backend.evaluate(batch)
                        logits = backend.to_numpy(device_logits)
                        loss = float(backend.to_numpy(loss_value).reshape(-1)[0])
                    values[start : start + len(selected)] = logits
                    total += loss * len(selected)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite checkpoint replay")
            result[split] = values
            if engine != "oracle":
                result[split + "_nll"] = np.array(total / len(rows), dtype=np.float64)
            else:
                with torch.no_grad():
                    result[split + "_torch_fp64"] = model(
                        torch.tensor(rows[[0, -1], :-1].astype(np.int64))
                    ).numpy()
                result[split + "_rows"] = rows.copy()
        tokens = fixture["test"][:1, :32]
        if engine == "cpu_arm64":
            result["b1"] = workspace_tensor(
                native.forward(parameters, np.ascontiguousarray(tokens[0])), "logits"
            )[None].copy()
        elif engine in ("cpu_torch", "oracle"):
            with torch.no_grad():
                result["b1"] = model(torch.tensor(tokens.astype(np.int64))).numpy()
        else:
            result["b1"] = backend.to_numpy(backend.forward(backend.upload_inputs(tokens))).copy()
        with Path(output).open("xb") as handle:
            np.savez(handle, **result)
    finally:
        if backend is not None:
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=("cpu_arm64", "cpu_torch", "mps", "mlx", "metal", "oracle"),
        required=True,
    )
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay(args.engine, args.fixture, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
