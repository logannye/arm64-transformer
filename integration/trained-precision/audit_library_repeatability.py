"""Independent raw-array audit of the complete maintained-library diagnostic.

Recompute the retained embedding, FP64 full-model and checkpoint oracles, bind
every repetition to its raw first execution, and compare two fresh processes.
"""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "repeatability_independent_origins",
    ROOT / "integration/trained-precision/audit_captured_replay.py",
)
PROVENANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVENANCE)
SUITE, require = PROVENANCE.SUITE, PROVENANCE.require


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    require(all(np.isfinite(v).all() for v in result.values()), "Nonfinite retained arrays")
    return result


def check_repetitions(case, arrays):
    from transformer.gpu_repeatability import array_identity

    require(
        case["complete"]
        and case["exact_passed"]
        and [e["execution"] for e in case["executions"]] == list(range(10)),
        "Incomplete ten-execution case",
    )
    for execution in case["executions"]:
        require(
            execution["finite"] and set(execution["exact"]) == set(arrays), "Wrong output inventory"
        )
        for key, actual in arrays.items():
            comparison = execution["exact"][key]
            expected = array_identity(actual)
            require(
                comparison["passed"]
                and comparison["different_coordinates"] == 0
                and comparison["actual"] == expected
                and comparison["expected"] == expected,
                "Execution digest differs from retained raw array",
            )


def audit_worker(path, expected, capture_path, capture_sha):
    SUITE.pinned(path, expected)
    value = SUITE.read_json(path)
    SUITE.pinned(capture_path, capture_sha)
    capture = SUITE.read_json(capture_path)
    require(
        value["schema"] == "maintained-library-repeatability-development-v1"
        and value["protocol"] == "trained-precision-development-v1"
        and value["complete"]
        and value["integrity_after"]
        and value["formal_eligible"] is False
        and value["formal_fits"] == 0
        and value["admission_stream_opened"] is False
        and value["capture"] == SUITE.binding(capture_path)
        and value["engine"] in ("mps", "mlx")
        and value["width"] in (32, 64, 128, 256, 512, 1024)
        and os.environ.get("ARM64_TRANSFORMER_WIDTH") == str(value["width"]),
        "Wrong repeatability worker contract or audit width",
    )
    for key in ("runtime_inventory", "framework_selection", "runtime_receipt"):
        SUITE.binding(capture[key]["path"], capture[key])
    runtime = SUITE.read_json(capture["runtime_inventory"]["path"])
    receipt = SUITE.read_json(capture["runtime_receipt"]["path"])
    project = receipt["project"]
    producer = Path(project["root"]) / "integration/trained-precision/library_repeatability.py"
    require(value["producer"] == SUITE.binding(producer), "Wrong repeatability producer")
    SUITE.pinned(producer, project["files"][producer.relative_to(project["root"]).as_posix()])
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
        "Wrong maintained-library selection",
    )
    loaded = value["origins_after"]["loaded_python_files"]
    require(loaded[module]["path"] == value["selected_api"]["path"], "Loaded API differs")
    for name in (
        module,
        module.rsplit(".", 1)[0] + "._fp32_precise_dense",
        module.rsplit(".", 1)[0] + "._fp32_precise_math",
    ):
        PROVENANCE.known_file(loaded[name]["path"], runtime, capture["runtime_root"], project)
    directory = Path(path).parent
    actual_files = {
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    }
    require(actual_files == set(value["artifacts"]) | {"result.json"}, "Artifact inventory differs")
    for name, bound in value["artifacts"].items():
        p = directory / name
        require(p.resolve(strict=True) == p and str(p) == bound["path"], "Unsafe artifact path")
        SUITE.binding(p, bound)
    worker = SUITE.read_json(directory / "worker.json")
    from transformer import spec
    from transformer.gpu_admission_checks import numpy_model
    from transformer.gpu_repeatability import (
        TOLERANCES,
        _torch_reference,
        accuracy,
        array_identity,
        embedding_cases,
        embedding_operands,
        embedding_oracle,
        expected_case_names,
        tokens_for,
    )
    from transformer.reference import Transformer

    require(
        worker["contract"] == value["schema"]
        and worker["protocol"] == value["protocol"]
        and worker["engine"] == value["engine"]
        and worker["width"] == value["width"]
        and worker["stages"] == ["embedding", "full_model", "training"]
        and worker["repetitions"] == 10
        and worker["complete"]
        and worker["exact_passed"]
        and worker["source_unchanged"]
        and not worker["cleanup_errors"]
        and worker["inventory_complete"]
        and worker["runtime_identity_sha256"] == capture["runtime_identity_sha256"],
        "Incomplete worker or changed inventory",
    )
    for name, digest in worker["source_sha256"].items():
        require(project["files"][name] == digest, "Worker source differs from capture")
    initial = read_arrays(directory / "initial-state.npz")
    require(
        set(initial) == {"parameters", "m", "v", "step"}
        and all(
            initial[k].shape == (spec.PARAM_COUNT,) and initial[k].dtype == np.float32
            for k in ("parameters", "m", "v")
        )
        and int(initial["step"]) == 0
        and (initial["m"] == 0).all()
        and (initial["v"] == 0).all()
        and array_identity(initial["parameters"]) == worker["initial"],
        "Wrong initial state",
    )
    require(
        array_identity(initial["parameters"])
        == array_identity(Transformer(seed=20261103).flat_parameters()),
        "Development initialization differs",
    )
    names = expected_case_names(value["width"], worker["stages"])
    actual_names = [c["name"] for c in worker["cases"]]
    require(
        worker["expected_cases"] == names
        and len(actual_names) == len(names)
        and set(actual_names) == set(names),
        "Missing, duplicated or extra repeatability case",
    )
    for key in ("capabilities", "capabilities_after"):
        capabilities = worker[key]
        api = capabilities["library_api"]
        require(
            capabilities["implementation"] == "maintained-library-training-v1"
            and capabilities["standard_release_claim"] is False
            and capabilities["storage_dtype"] == "float32"
            and capabilities["cpu_tensor_fallback"] is False
            and api["module"] == module
            and api["module_path"] == loaded[module]["path"]
            and api["identity"] == "fp32-saved-precision-api-v1"
            and api["model_identity"] == "maintained-library-fp32-saved-model-v1"
            and api["precise_stages"]
            == ["ln1", "qkv", "attention"] + (["ff1"] if value["engine"] == "mlx" else []),
            "Wrong persistent model implementation",
        )
        if value["engine"] == "mlx":
            require(capabilities["compiled"] is True, "Compiled MLX required")
        else:
            require(capabilities["deterministic_algorithms"] is True, "Strict MPS required")
    if value["engine"] == "mlx":
        compiled = worker["compiled_training"]
        shapes = {f"b{b}-t{t}" for b in (1, 3, 16) for t in (3, 32)}
        require(
            compiled["python_trace_count"] == len(shapes)
            and compiled["python_trace_shapes"] == dict.fromkeys(shapes, 1)
            and set(compiled["graphs"]) == shapes,
            "Missing per-shape compiled training proof",
        )
        for shape, graph in compiled["graphs"].items():
            raw = (directory / graph["graph"]).read_bytes()
            require(
                graph["graph"] == f"training-loss-{shape}.dot"
                and SUITE.sha256(directory / graph["graph"]) == graph["sha256"]
                and len(raw) == graph["bytes"]
                and b"Compiled" in raw,
                "Missing compiled training graph",
            )
    embeddings = {c["name"]: c for c in embedding_cases(value["width"]) if c["supported"]}
    rows = []
    for case in worker["cases"]:
        require(case["first_arrays"] == case["name"] + "/first.npz", "Wrong first-array path")
        arrays = read_arrays(directory / case["first_arrays"])
        require(
            all(v.dtype == (np.int64 if k == "step" else np.float32) for k, v in arrays.items()),
            "Retained model state or output storage differs",
        )
        check_repetitions(case, arrays)
        checks = {}
        if case["name"] in embeddings:
            import torch

            tokens, table, dy = embedding_operands(value["width"], embeddings[case["name"]])
            fixture = read_arrays(directory / case["name"] / "fixture.npz")
            require(
                set(fixture) == {"tokens", "table", "cotangent"},
                "Embedding fixture inventory differs",
            )
            for key, array in (("tokens", tokens), ("table", table), ("cotangent", dy)):
                require(
                    array_identity(fixture[key]) == array_identity(array),
                    "Embedding operands differ",
                )
            gold = embedding_oracle(tokens, table, dy)
            require(set(arrays) == set(gold), "Embedding result inventory differs")
            checks = {
                k: accuracy(arrays[k], gold[k], "gradient" if k == "gradient" else "forward")
                for k in gold
            }
            weight = torch.tensor(table, dtype=torch.float64, requires_grad=True)
            output = torch.nn.functional.embedding(torch.tensor(tokens.astype(np.int64)), weight)
            (output * torch.tensor(dy, dtype=torch.float64)).sum().backward()
            checks["independent_embedding_oracle"] = accuracy(
                weight.grad.numpy(), gold["gradient"], "oracle"
            )
        else:
            shape = case["name"].split("/")[1]
            batch, length = (int(s[1:]) for s in shape.split("-"))
            tokens = tokens_for(batch, length, "skewed")
            targets = np.roll(tokens, -1, axis=1).copy()
            require(
                case["fixture"]
                == dict(tokens=array_identity(tokens), targets=array_identity(targets)),
                "Full-model operands differ",
            )
            training = case["name"].startswith("training/")
            p = arrays["parameters"] if training else initial["parameters"]
            gold = numpy_model(p, tokens, targets, backward=not training)
            independent = _torch_reference(p, tokens, targets, backward=not training)
            for key in gold:
                checks["independent_" + key] = accuracy(independent[key], gold[key], "oracle")
            for key in ("logits",) if training else gold:
                checks[key] = accuracy(
                    arrays[key], gold[key], "gradient" if key == "gradient" else "forward"
                )
            if training:
                require(
                    set(arrays) == {"parameters", "m", "v", "step", "logits", "losses"}
                    and int(arrays["step"]) == 6
                    and arrays["losses"].shape == (6,)
                    and all(
                        arrays[k].shape == (spec.PARAM_COUNT,) and arrays[k].dtype == np.float32
                        for k in ("parameters", "m", "v")
                    )
                    and (arrays["v"] >= 0).all(),
                    "Incomplete six-update state",
                )
            else:
                require(set(arrays) == set(gold), "Full-model output inventory differs")
        rows.append(
            dict(
                name=case["name"],
                comparisons=checks,
                passed=all(c["passed"] for c in checks.values()),
            )
        )
    numerics_passed = all(r["passed"] for r in rows)
    require(
        worker["accuracy_passed"] == numerics_passed and value["passed"] == numerics_passed,
        "Worker numerical decision differs from independent raw audit",
    )
    SUITE.pinned(path, expected)
    return value, dict(
        worker=SUITE.binding(path),
        cases=rows,
        tolerances=TOLERANCES,
        numerics_passed=numerics_passed,
    )


def audit_pair(first, second, capture, capture_sha):
    values, audits = [], []
    for path, digest in (first, second):
        value, audited = audit_worker(Path(path), digest, Path(capture), capture_sha)
        values.append(value)
        audits.append(audited)
    a, b = values
    require(
        a["pid"] != b["pid"] and all(a[k] == b[k] for k in ("engine", "width", "capture")),
        "Two matching fresh processes required",
    )
    from transformer.gpu_repeatability import exact_comparison

    left, right = Path(first[0]).parent, Path(second[0]).parent
    worker = SUITE.read_json(left / "worker.json")
    compared = []
    for case in worker["cases"]:
        x, y = (read_arrays(p / case["first_arrays"]) for p in (left, right))
        require(set(x) == set(y), "Cross-process array inventory differs")
        compared.append(
            dict(name=case["name"], passed=all(exact_comparison(x[k], y[k])["passed"] for k in x))
        )
    return dict(
        schema="maintained-library-independent-repeatability-pair-v1",
        evidence_valid=True,
        engine=a["engine"],
        width=a["width"],
        workers=audits,
        cross_process=compared,
        exact_passed=all(r["passed"] for r in compared),
        numerics_passed=all(r["numerics_passed"] for r in audits),
        formal_eligible=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "capture"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    import json

    result = audit_pair(
        (args.first, args.first_sha256),
        (args.second, args.second_sha256),
        args.capture,
        args.capture_sha256,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(int(not (result["exact_passed"] and result["numerics_passed"])))
