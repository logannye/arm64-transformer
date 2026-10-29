"""Expose every development attention case, retaining failures without tuning gates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from transformer import spec
from transformer.gpu_metal import Backend
from transformer.gpu_precision import attention_reference, attention_stress_cases
from transformer.precision_audit import comparison

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--output-root", type=Path, default=Path("work/gpu-v1/metal-development"))
args = parser.parse_args()
destination = args.output_root / args.label
destination.mkdir(parents=True, exist_ok=True)
out = destination / f"attention-width-{spec.WIDTH}.json"
if out.exists():
    raise FileExistsError(out)
initial = np.zeros(spec.PARAM_COUNT, np.float32)
backend = Backend(initial)
cases = []


def run(qkv, dy):
    return {key: backend.to_numpy(value) for key, value in backend.attention(qkv, dy).items()}


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


for case in attention_stress_cases(spec.WIDTH, split="development"):
    name, qkv, dy = case["name"], case["qkv"], case["dy"]
    before = (digest(qkv), digest(dy))
    values = run(qkv, dy)
    gold = attention_reference(qkv, 4, dy)
    saved = attention_reference(qkv, 4, dy, probabilities=values["probabilities"])
    checks = []
    for key, tolerance in [("output", "forward"), ("probabilities", "math"), ("dqkv", "gradient")]:
        checks.append(comparison(name + "/" + key, values[key], gold[key], tolerance=tolerance))
    checks.append(
        comparison(
            name + "/same_saved_probabilities_gradient",
            values["dqkv"],
            saved["dqkv"],
            tolerance="gradient",
        )
    )
    repeated = run(qkv, dy)
    exact_repeat = all(values[key].tobytes() == repeated[key].tobytes() for key in values)
    causal = True
    future_zero = True
    prefix_gradient = True
    if case["tokens"] > 1:
        dy_prefix = dy.copy()
        dy_prefix[-1] = 0
        prefix = run(qkv, dy_prefix)
        intervention = qkv.copy()
        intervention[-1] *= np.float32(-3)
        changed = run(intervention, dy_prefix)
        causal = np.array_equal(values["output"][:-1], changed["output"][:-1])
        prefix_gradient = np.array_equal(prefix["dqkv"][:-1], changed["dqkv"][:-1])
        future_zero = np.array_equal(prefix["dqkv"][-1], np.zeros_like(prefix["dqkv"][-1]))
    upper = values["probabilities"][
        :, np.triu_indices(case["tokens"], 1)[0], np.triu_indices(case["tokens"], 1)[1]
    ]
    mask_zero = bool(np.all(upper == 0))
    mass = comparison(
        name + "/probability_mass",
        values["probabilities"].sum(-1, dtype=np.float64),
        np.ones((4, case["tokens"])),
        tolerance="math",
    )
    checks.append(mass)
    guards = backend.check_guards()
    unchanged = before == (digest(qkv), digest(dy))
    finite = all(np.isfinite(values[key]).all() for key in ("output", "probabilities", "dqkv"))
    record = {
        "name": name,
        "tokens": case["tokens"],
        "fixture_sha256": {"qkv": before[0], "dy": before[1]},
        "checks": checks,
        "exact_repeat": exact_repeat,
        "causal_future_intervention": causal,
        "future_gradient_exact_zero": future_zero,
        "prefix_gradient_exact": prefix_gradient,
        "mask_exact_zero": mask_zero,
        "input_unchanged": unchanged,
        "finite": finite,
        "guards": guards,
    }
    record["passed"] = all(check["passed"] for check in checks) and all(
        record[k]
        for k in (
            "exact_repeat",
            "causal_future_intervention",
            "future_gradient_exact_zero",
            "prefix_gradient_exact",
            "mask_exact_zero",
            "input_unchanged",
            "finite",
            "guards",
        )
    )
    cases.append(record)
    print(
        json.dumps(
            {
                "case": name,
                "passed": record["passed"],
                "failed": {
                    check["name"]: check["failed_coordinates"]
                    for check in checks
                    if not check["passed"]
                },
            }
        ),
        flush=True,
    )
result = {
    "scope": "All 23 GPU development attention cases; no admission operands or thresholds changed",
    "label": args.label,
    "width": spec.WIDTH,
    "passed": all(case["passed"] for case in cases),
    "cases": cases,
    "capabilities": backend.capabilities(),
    "memory": backend.memory_stats(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
backend.close()
print("PASSED", result["passed"], flush=True)
raise SystemExit(0 if result["passed"] else 1)
