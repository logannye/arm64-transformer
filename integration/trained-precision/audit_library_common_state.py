"""Recompute captured common-state and fixed-gradient decisions from raw arrays."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "common_independent_provenance", ROOT / "integration/trained-precision/audit_captured_replay.py"
)
PROVENANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVENANCE)
SUITE, require = PROVENANCE.SUITE, PROVENANCE.require


def accuracy(actual, reference, kind):
    tolerances = {
        "forward": (2e-5, 2e-4),
        "gradient": (3e-5, 3e-3),
        "optimizer": (3e-6, 3e-4),
        "oracle": (1e-10, 1e-10),
        "exact": (0, 0),
    }
    atol, rtol = tolerances[kind]
    a, b = np.asarray(actual), np.asarray(reference)
    require(a.shape == b.shape, "Wrong raw result shape")
    if kind == "exact":
        passed = a.dtype == b.dtype and a.tobytes() == b.tobytes()
        return dict(passed=passed, atol=0, rtol=0)
    a, b = a.astype(np.float64), b.astype(np.float64)
    failed = ~np.isfinite(a) | ~np.isfinite(b) | (np.abs(a - b) > atol + rtol * np.abs(b))
    return dict(
        passed=not bool(failed.any()), atol=atol, rtol=rtol, failed_coordinates=int(failed.sum())
    )


def checkpoint(trajectory, step, count):
    rows = [r for r in trajectory["saved"] if r["step"] == step]
    require(
        len(rows) == 1 and set(rows[0]["files"]) == {"parameters", "m", "v"},
        "Incomplete common checkpoint",
    )
    require(
        len(trajectory["states"]) == 65 and trajectory["states"][step]["step"] == step,
        "Wrong trajectory state inventory",
    )
    state = dict(step=np.asarray(step, dtype=np.int64))
    for key, bound in rows[0]["files"].items():
        SUITE.binding(bound["path"], bound)
        require(
            bound["bytes"] == count * 4 and trajectory["states"][step][key] == bound["sha256"],
            "Wrong common checkpoint binding",
        )
        state[key] = np.fromfile(bound["path"], "<f4")
        require(np.isfinite(state[key]).all(), "Nonfinite common checkpoint")
    require((state["v"] >= 0).all(), "Negative second moment")
    return state


def adam(state, gradient, count):
    p, m, v = (state[k].copy() for k in ("parameters", "m", "v"))
    g = gradient.astype(np.float64)
    step = int(state["step"])
    for _ in range(count):
        step += 1
        m = (0.9 * m.astype(np.float64) + 0.1 * g).astype(np.float32)
        v = (0.999 * v.astype(np.float64) + 0.001 * g * g).astype(np.float32)
        corrected_m = m.astype(np.float64) / (1 - 0.9**step)
        corrected_v = v.astype(np.float64) / (1 - 0.999**step)
        p = (p.astype(np.float64) - 0.001 * corrected_m / (np.sqrt(corrected_v) + 1e-8)).astype(
            np.float32
        )
    return dict(parameters=p, m=m, v=v, step=np.asarray(step, dtype=np.int64))


def audit(path, expected, capture_path, capture_sha):
    SUITE.pinned(path, expected)
    value = SUITE.read_json(path)
    SUITE.pinned(capture_path, capture_sha)
    capture = SUITE.read_json(capture_path)
    require(
        value["schema"] == "maintained-library-common-state-development-v1"
        and value["complete"]
        and value["integrity_after"]
        and value["seed"] == 20261103
        and value["formal_eligible"] is False
        and value["formal_fits"] == 0
        and value["admission_stream_opened"] is False
        and value["capture"] == SUITE.binding(capture_path)
        and value["engine"] in ("cpu_torch", "mps", "mlx")
        and value["width"] in (32, 64, 128, 256, 512, 1024)
        and os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(value["width"]),
        "Wrong common-state worker contract or audit width",
    )
    for key in ("runtime_inventory", "runtime_receipt", "framework_selection"):
        SUITE.binding(capture[key]["path"], capture[key])
    runtime = SUITE.read_json(capture["runtime_inventory"]["path"])
    receipt = SUITE.read_json(capture["runtime_receipt"]["path"])
    project = receipt["project"]
    producer = Path(project["root"]) / "integration/trained-precision/library_common_state.py"
    require(
        value["producer"] == SUITE.binding(producer)
        and value["producer"]["sha256"]
        == project["files"][producer.relative_to(project["root"]).as_posix()],
        "Wrong common-state producer",
    )
    for key in ("origins_before", "origins_after"):
        PROVENANCE.audit_origins(value[key], value["engine"], capture, runtime, receipt)
    framework = "mlx" if value["engine"] == "mlx" else "pytorch"
    choice = receipt["frameworks"][framework]["selection"]["precision_api"]
    module = choice["module"]
    require(
        value["selected_api"]["module"] == module
        and value["selected_api"]["identity"] == "fp32-saved-precision-api-v1"
        and choice["stock_operators_unchanged"] is True
        and choice["derivation_receipt_sha256"]
        == capture["precision_api_derivations"][framework]["receipt"]["sha256"],
        "Wrong common-state model API",
    )
    loaded = value["origins_after"]["loaded_python_files"]
    require(loaded[module]["path"] == value["selected_api"]["path"], "Wrong loaded precision API")
    for name in (
        module,
        module.rsplit(".", 1)[0] + "._fp32_precise_dense",
        module.rsplit(".", 1)[0] + "._fp32_precise_math",
    ):
        PROVENANCE.known_file(loaded[name]["path"], runtime, capture["runtime_root"], project)
    bound = value["trajectory"]
    SUITE.binding(bound["path"], bound)
    trajectory = SUITE.read_json(bound["path"])
    require(
        trajectory["schema"] == "maintained-library-optimizer-development-v1"
        and trajectory["engine"] == "cpu_torch"
        and trajectory["width"] == value["width"]
        and trajectory["seed"] == 20261103
        and trajectory["start_step"] == 0
        and trajectory["complete"]
        and trajectory["passed"]
        and trajectory["integrity_after"],
        "Wrong exposed development trajectory",
    )
    SUITE.binding(trajectory["capture"]["path"], trajectory["capture"])
    previous = SUITE.read_json(trajectory["capture"]["path"])
    require(
        previous["runtime_identity_sha256"] == capture["runtime_identity_sha256"],
        "Common operands came from different runtime bytes",
    )
    for framework in ("pytorch", "mlx"):
        old, new = (c["precision_api_derivations"][framework] for c in (previous, capture))
        require(
            old["receipt"] == new["receipt"]
            and old["helper"]["sha256"] == new["helper"]["sha256"]
            and old["helper"]["bytes"] == new["helper"]["bytes"],
            "Common operands came from different package derivations",
        )
    SUITE.binding(trajectory["fixture"]["path"], trajectory["fixture"])
    fixture = SUITE.read_json(trajectory["fixture"]["path"])
    require(
        fixture["schema"] == "maintained-library-optimizer-inputs-v1"
        and fixture["width"] == value["width"]
        and fixture["seed"] == 20261103,
        "Wrong common-state input manifest",
    )
    for key in ("initial", "training"):
        SUITE.binding(fixture[key]["path"], fixture[key])
    count = 8 * value["width"] ** 2 + 301 * value["width"] + 128
    zero = checkpoint(trajectory, 0, count)
    require(
        hashlib.sha256(zero["parameters"].tobytes()).hexdigest() == fixture["initial"]["sha256"]
        and (zero["m"] == 0).all()
        and (zero["v"] == 0).all(),
        "Wrong initial common state",
    )
    rows = np.fromfile(fixture["training"]["path"], "<i4").reshape(2048, 33)
    rng = np.random.default_rng(np.random.SeedSequence([20261103, value["width"], 91]))
    gradient = rng.normal(0, 0.01, count).astype(np.float32)
    names = [f"model/step-{s}/b{b}-t32" for s in (0, 8, 64) for b in (1, 3, 16)] + [
        f"adam/step-{s}/count-{n}" for s in (0, 64) for n in (1, 2, 10)
    ]
    require(
        value["expected_cases"] == names and [r["name"] for r in value["cases"]] == names,
        "Missing or reordered common-state case",
    )
    import torch

    from transformer.gpu_admission_checks import numpy_model
    from transformer.gpu_repeatability import _torch_reference

    require(
        "torch.nn.precise_fp32" not in sys.modules and "mlx.core" not in sys.modules,
        "Independent oracle must not import selected candidate implementations",
    )
    checked = []
    for row in value["cases"]:
        family, step_text, shape = row["name"].split("/")
        state = checkpoint(trajectory, int(step_text.split("-")[1]), count)
        raw = row["actual"]
        raw_path = Path(path).parent / (row["name"].replace("/", "-") + ".npz")
        require(str(raw_path) == raw["path"], "Wrong raw result path")
        SUITE.binding(raw_path, raw)
        with np.load(raw_path, allow_pickle=False) as archive:
            actual = {k: archive[k].copy() for k in archive.files}
        require(all(np.isfinite(a).all() for a in actual.values()), "Nonfinite common result")
        oracle = {}
        if family == "model":
            batch = int(shape.split("-")[0][1:])
            tokens, targets = (
                np.ascontiguousarray(rows[:batch, :-1]),
                np.ascontiguousarray(rows[:batch, 1:]),
            )
            gold = numpy_model(state["parameters"], tokens, targets)
            double = _torch_reference(state["parameters"], tokens, targets, backward=True)
            require(
                set(actual) == set(gold) == set(row["repetitions"]), "Wrong model result inventory"
            )
            require(
                actual["logits"].dtype == actual["gradient"].dtype == np.float32
                and actual["loss"].dtype in (np.dtype("f4"), np.dtype("f8")),
                "Wrong model storage dtype",
            )
            for k, a in actual.items():
                fingerprint = dict(
                    dtype=a.dtype.str,
                    shape=list(a.shape),
                    sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                )
                require(
                    row["repetitions"][k] == [fingerprint, fingerprint],
                    "Raw model repetitions differ",
                )
            checks = {
                k: accuracy(actual[k], gold[k], "gradient" if k == "gradient" else "forward")
                for k in gold
            }
            oracle = {k: accuracy(double[k], gold[k], "oracle") for k in gold}
        else:
            gold = adam(state, gradient, int(shape.split("-")[1]))
            require(
                set(actual) == set(gold)
                and row["repetitions"] is None
                and all(
                    actual[k].shape == (count,) and actual[k].dtype == np.float32
                    for k in ("parameters", "m", "v")
                )
                and (actual["v"] >= 0).all(),
                "Wrong complete Adam state",
            )
            checks = {
                k: accuracy(actual[k], gold[k], "exact" if k == "step" else "optimizer")
                for k in gold
            }
        require(
            set(row["comparisons"]) == set(checks) and set(row["oracle"]) == set(oracle),
            "Comparison inventory differs",
        )
        for source, recomputed in ((row["comparisons"], checks), (row["oracle"], oracle)):
            require(
                all(
                    source[k]["passed"] == r["passed"]
                    and source[k]["atol"] == r["atol"]
                    and source[k]["rtol"] == r["rtol"]
                    for k, r in recomputed.items()
                ),
                "Worker raw comparison decision differs",
            )
        passed = all(r["passed"] for r in [*checks.values(), *oracle.values()])
        require(row["passed"] == passed, "Worker case outcome differs")
        checked.append(dict(name=row["name"], comparisons=checks, oracle=oracle, passed=passed))
    passed = all(r["passed"] for r in checked)
    require(value["passed"] == passed, "Worker numerical decision differs")
    SUITE.pinned(path, expected)
    return dict(
        schema="maintained-library-independent-common-state-v1",
        worker=SUITE.binding(path),
        capture=SUITE.binding(capture_path),
        engine=value["engine"],
        width=value["width"],
        cases=checked,
        evidence_valid=True,
        numerics_passed=passed,
        formal_eligible=False,
        audit_environment=dict(
            python=sys.version,
            numpy=np.__version__,
            torch=torch.__version__,
            torch_core=SUITE.binding(torch._C.__file__),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "capture"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    result = audit(args.receipt, args.receipt_sha256, args.capture, args.capture_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(int(not result["numerics_passed"]))
