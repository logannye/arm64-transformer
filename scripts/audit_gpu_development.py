"""Audit saved tiny development-fit logits against independent FP64 equations.

This performs no GPU execution and cannot qualify a formal GPU campaign. The
fixture must identify every split as coming from the CPU training partition.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from transformer import spec
from transformer.precision_audit import FP64Transformer, comparison, prediction_differences
from transformer.prepare import load_fixture
from transformer.reference import Transformer


def audit(fixture_directory, runs_directory, output):
    fixture = load_fixture(fixture_directory)
    if fixture["manifest"].get("gpu_development_fixture") is not True:
        raise ValueError("Only training-partition GPU development fixtures may be audited")
    if output.exists() or output.is_symlink():
        raise ValueError("Audit output must be fresh")
    torch.set_num_threads(1)
    records, engines, identities = [], {}, {}
    for engine in ("mps", "mlx", "metal"):
        directory = runs_directory / f"{engine}-{spec.WIDTH}"
        runtime = json.loads((directory / "runtime.json").read_text())
        if runtime["scope"] != "development_only" or runtime["status"] != "completed":
            raise ValueError(f"{engine}: missing completed development receipt")
        for name in ("best.f32", "test_logits.f32", "inference_logits.f32"):
            actual_hash = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if runtime["artifact_sha256"][name] != actual_hash:
                raise ValueError(f"{engine}/{name}: artifact authentication failed")
            identities[f"{engine}/{name}"] = actual_hash
        parameters = np.fromfile(directory / "best.f32", dtype="<f4")
        oracle = FP64Transformer(parameters)
        tokens, targets = fixture["test"][:, :-1], fixture["test"][:, 1:]
        expected = oracle(tokens)
        model = Transformer().load_flat(parameters).double()
        with torch.no_grad():
            torch_logits = model(torch.tensor(tokens.astype(np.int64))).numpy()
        oracle_errors = np.abs(expected - torch_logits)
        oracle_limits = 1e-10 + 1e-10 * np.abs(torch_logits)
        records.append(
            {
                "name": f"{engine}/independent-oracles",
                "passed": bool(
                    np.all(np.isfinite(expected)) and np.all(oracle_errors <= oracle_limits)
                ),
                "atol": 1e-10,
                "rtol": 1e-10,
                "max_absolute_error": float(oracle_errors.max()),
                "max_tolerance_ratio": float((oracle_errors / oracle_limits).max()),
                "failed_coordinates": int(np.count_nonzero(oracle_errors > oracle_limits)),
            }
        )
        saved = np.fromfile(directory / "test_logits.f32", dtype="<f4").reshape(expected.shape)
        records.append(comparison(f"{engine}/selected-checkpoint-logits", saved, expected))
        inference = np.fromfile(directory / "inference_logits.f32", dtype="<f4").reshape(
            1, spec.CONTEXT, spec.VOCAB
        )
        records.append(comparison(f"{engine}/batch-one-logits", inference, expected[:1]))
        engines[engine] = prediction_differences(saved, expected, targets)
    result = {
        "scope": "training-only GPU development checkpoints; not formal admission or measurement",
        "width": spec.WIDTH,
        "test_sequences_per_engine": len(fixture["test"]),
        "checks": records,
        "prediction_differences": engines,
        "artifact_sha256": identities,
        "passed": all(record["passed"] for record in records),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.fixture, args.runs, args.output)
    print(
        json.dumps(
            {"width": result["width"], "passed": result["passed"], "checks": len(result["checks"])}
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
