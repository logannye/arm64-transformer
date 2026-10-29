"""Negative controls for saved QKV evidence; no GPU or model execution."""

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from scripts import report_gpu_qkv as report
from transformer.gpu_policy import bundle, identity, reference, with_qkv


def paired(ratio=0.9, offset=0):
    result = {}
    for label, seconds in (("reference", 0.002), ("candidate", 0.002 * ratio)):
        result[label] = {
            "warmups": 20,
            "observations": 100,
            "blocks": 5,
            "seconds": [seconds] * 100,
            "block_median_seconds": [seconds] * 5,
            "median_seconds": seconds,
            "exact_block_replay": True,
            "exact_state_restoration": True,
            "guards": True,
        }
    result.update(
        paired_latency_ratios=[ratio] * 5,
        median_paired_latency_ratio=ratio,
        paired_improvement_fractions=[1 - ratio] * 5,
        median_paired_improvement_fraction=1 - ratio,
        block_order=[
            ["reference", "candidate"] if (i + offset) % 2 == 0 else ["candidate", "reference"]
            for i in range(5)
        ],
        boundary="train_step + finite scan + scalar loss readback",
    )
    return result


def condition_receipt():
    return {
        f"conditions_{when}{suffix}": value
        for when in ("before", "after")
        for suffix, value in (
            (
                "",
                {
                    "power": {"returncode": 0, "stdout": "AC Power"},
                    "thermal": {"stdout": "GPU_Speed_Limit = 100"},
                },
            ),
            ("_assessment", {"valid": True, "ac_power_confirmed": True}),
        )
    }


def memory_receipt():
    values = {
        "owned_buffer_count": 4,
        "workspace_buffer_count": 1,
        "input_buffer_count": 0,
        "batch_buffer_count": 3,
        "owned_buffer_bytes_including_guards": 4096,
    }
    return {
        "fresh_instances_without_qualification_allocations": True,
        "equal_initial_state": True,
        "memory_before_timing": {"reference": values, "candidate": values.copy()},
        "allocation_validation": {
            "equal_owned_counts_and_bytes": True,
            "compared_fields": list(values),
        },
    }


def numeric(name, shape, tolerance):
    return {
        "name": name,
        "passed": True,
        "failed_coordinates": 0,
        "actual_shape": list(shape),
        "expected_shape": list(shape),
        "coordinate_count": math.prod(shape),
        "failure_examples": [],
        "max_absolute_error": 0.0,
        "max_tolerance_ratio": 0.0,
        "rmse": 0.0,
        "tolerance": tolerance,
    }


def full_model(width=32):
    sizes = {
        "token": 128 * width,
        "position": 32 * width,
        "ln1_weight": width,
        "ln1_bias": width,
        "qkv_weight": 3 * width * width,
        "qkv_bias": 3 * width,
        "attention_weight": width * width,
        "attention_bias": width,
        "ln2_weight": width,
        "ln2_bias": width,
        "ff1_weight": 2 * width * width,
        "ff1_bias": 2 * width,
        "ff2_weight": 2 * width * width,
        "ff2_bias": width,
        "ln3_weight": width,
        "ln3_bias": width,
        "head_weight": 128 * width,
        "head_bias": 128,
    }
    records = []
    for batch in (1, 16):
        records += [
            numeric(f"B{batch}/logits", [batch, 32, 128], "forward"),
            numeric(f"B{batch}/loss", [], "forward"),
        ]
        records += [
            numeric(f"B{batch}/gradient/{name}", [count], "gradient")
            for name, count in sizes.items()
        ]
    records += [
        numeric(f"fixed_gradient_adam/{name}", [sum(sizes.values())], "optimizer")
        for name in ("m", "v", "parameters")
    ]
    controls = [
        f"B{b}/{n}"
        for b in (1, 16)
        for n in (
            "finite",
            "forward_repeat",
            "gradient_repeat",
            "guards",
            "loss_repeat",
            "training_repeat",
        )
    ]
    controls += [
        "B1/causal_future_intervention",
        "B16/sequence_isolation",
        "fixed_gradient_adam_repeat",
        "fixed_gradient_adam_step",
    ]
    return {
        "passed": True,
        "checks": records,
        "controls": dict.fromkeys(controls, True),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def group_fixture(tmp_path, ratios, *, monkeypatch):
    """Selection test isolates policy/timing evidence from numerical gate tests."""
    monkeypatch.setattr(report, "qualification", lambda *_: None)
    monkeypatch.setattr(report, "fresh_pair", lambda *_: None)
    current = with_qkv(reference())
    rows = []
    for index, candidate in enumerate(c for c in report.inventory() if c["family"] == "forward"):
        policy = report.apply_candidate(current, candidate)
        row = {
            **candidate,
            "mode": "training",
            "policy": policy,
            "policy_sha256": identity(policy),
            "qualification": {"passed": True},
            "status": "qualified_measured",
            "fresh_pair": {},
            "paired_timing": paired(ratios[index], index % 2),
            **condition_receipt(),
        }
        path = tmp_path / f"{candidate['id']}.json"
        write_json(path, row)
        row["evidence"] = {"path": path.name, "sha256": report.old.digest(path)}
        rows.append(row)
    eligible = [
        (ratio, row["id"], row["policy"])
        for row, ratio in zip(rows, ratios, strict=True)
        if ratio <= 0.97
    ]
    winner = min(eligible) if eligible else None
    group = {
        "width": 32,
        "mode": "training",
        "family": "forward",
        "before_policy": current,
        "after_policy": winner[2] if winner else current,
        "chosen_id": winner[1] if winner else None,
        "candidates": rows,
    }
    return group, current, dict.fromkeys([r["id"] for r in rows], True)


def select(group, current, supported, root):
    return report.select_group(
        group, "training", "forward", current, root, report.old.Receipts(), {}, supported, supported
    )


def test_raw_timings_define_p50_p95_and_paired_speed_ratio():
    result = report.paired_summary(paired(0.8), "training")
    assert result["speed_ratio_reference_over_selected"] == pytest.approx(1.25)
    assert result["candidate_p50_ms"] == pytest.approx(1.6)
    assert result["candidate_p95_ms"] == pytest.approx(1.6)
    assert result["candidate_training_targets_per_second"] == pytest.approx(320000)


@pytest.mark.parametrize("mutation", ["summary", "order", "warmup", "guard", "boundary", "budget"])
def test_timing_mutations_fail_closed(mutation):
    value = paired()
    if mutation == "summary":
        value["median_paired_latency_ratio"] = 0.5
    elif mutation == "order":
        value["block_order"][1] = value["block_order"][0]
    elif mutation == "warmup":
        value["candidate"]["exact_state_restoration"] = False
    elif mutation == "guard":
        value["reference"]["guards"] = False
    elif mutation == "boundary":
        value["boundary"] = "forward only"
    else:
        value["candidate"]["observations"] = 99
    with pytest.raises(ValueError):
        report.paired_summary(value, "training")


def test_null_retention_is_required_below_three_percent(tmp_path, monkeypatch):
    group, current, supported = group_fixture(tmp_path, [0.98] * 8, monkeypatch=monkeypatch)
    after, summary = select(group, current, supported, tmp_path)
    assert after == current and summary["chosen_id"] is None
    group["chosen_id"] = group["candidates"][0]["id"]
    group["after_policy"] = group["candidates"][0]["policy"]
    with pytest.raises(ValueError, match="3% threshold"):
        select(group, current, supported, tmp_path)


def test_best_qualified_ratio_and_fixed_identity_break_ties(tmp_path, monkeypatch):
    group, current, supported = group_fixture(
        tmp_path, [0.97, 0.9, 0.9, 1, 1, 1, 1, 1], monkeypatch=monkeypatch
    )
    _, summary = select(group, current, supported, tmp_path)
    assert summary["chosen_id"] == group["candidates"][1]["id"]
    group["chosen_id"] = group["candidates"][2]["id"]
    with pytest.raises(ValueError, match="ranking"):
        select(group, current, supported, tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "changed_file"])
def test_all_candidate_evidence_is_required_including_unselected(tmp_path, monkeypatch, mutation):
    group, current, supported = group_fixture(tmp_path, [0.9] + [1] * 7, monkeypatch=monkeypatch)
    if mutation == "missing":
        group["candidates"].pop()
    elif mutation == "duplicate":
        group["candidates"][-1] = group["candidates"][0]
    elif mutation == "reordered":
        group["candidates"].reverse()
    else:
        path = tmp_path / group["candidates"][-1]["evidence"]["path"]
        saved = json.loads(path.read_text())
        saved["status"] = "numerically_rejected_unranked"
        write_json(path, saved)
    with pytest.raises(ValueError):
        select(group, current, supported, tmp_path)


def test_resource_rejected_candidate_cannot_be_ranked(tmp_path, monkeypatch):
    group, current, supported = group_fixture(tmp_path, [0.8] * 8, monkeypatch=monkeypatch)
    supported[group["candidates"][0]["id"]] = False
    with pytest.raises(ValueError, match="Unsupported"):
        select(group, current, supported, tmp_path)


def test_allocation_symmetry_and_state_restoration_are_required():
    value = memory_receipt()
    report.old.validate_fresh_pair(value)
    value["memory_before_timing"]["candidate"]["input_buffer_count"] = 1
    with pytest.raises(ValueError):
        report.old.validate_fresh_pair(value)
    value = memory_receipt()
    value["equal_initial_state"] = False
    with pytest.raises(ValueError):
        report.old.validate_fresh_pair(value)


def test_full_model_requires_each_gradient_and_behavioral_control():
    value = full_model()
    report.full_model(value, 32)
    value["checks"].pop(10)
    with pytest.raises(ValueError, match="numerical checks"):
        report.full_model(value, 32)
    value = full_model()
    del value["controls"]["B16/sequence_isolation"]
    with pytest.raises(ValueError, match="behavioral control"):
        report.full_model(value, 32)


def test_one_ulp_disagreement_cannot_be_hidden_by_pass_flag():
    a = np.array([1.0], np.float32)
    b = np.nextafter(a, np.float32(2))
    left = {"dtype": a.dtype.str, "shape": [1], "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    right = {**left, "sha256": hashlib.sha256(b.tobytes()).hexdigest()}
    assert not report.old.exact_pass(
        {"passed": True, "different_coordinates": 0, "actual": left, "expected": right}
    )


def test_checkpoint_bytes_and_optimizer_budget_are_independently_checked(tmp_path):
    values = {"step": np.array(6, np.int64), "parameters": np.array([1], np.float32)}
    expected = {
        k: {
            "dtype": v.dtype.str,
            "shape": list(v.shape),
            "sha256": hashlib.sha256(v.tobytes()).hexdigest(),
        }
        for k, v in values.items()
    }
    np.savez(tmp_path / "first.npz", **values)
    report.retained_arrays(tmp_path, "first.npz", expected, report.old.Receipts())
    values["parameters"][0] = np.nextafter(np.float32(1), np.float32(2))
    np.savez(tmp_path / "first.npz", **values)
    with pytest.raises(ValueError, match="replay bytes"):
        report.retained_arrays(tmp_path, "first.npz", expected, report.old.Receipts())


def test_stale_native_or_policy_capability_is_not_qualification():
    policy = bundle(with_qkv(reference()))
    source = {
        "native/metal/backend.mm": "a" * 64,
        "native/metal/build/libtransformer_metal.dylib": "b" * 64,
    }
    cap = {
        "kernel_policy": policy,
        "kernel_policy_sha256": identity(policy),
        "host_source_sha256": "a" * 64,
        "compiled_host_source_sha256": "a" * 64,
        "binary_sha256": "b" * 64,
        "backend": "handwritten-metal-v3",
        "framework_numerical_calls": False,
        "storage_dtype": "float32",
        "width": 32,
        "parameter_count": 17952,
    }
    report.capability(cap, policy, source, 32)
    stale = copy.deepcopy(cap)
    stale["compiled_host_source_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Stale native"):
        report.capability(stale, policy, source, 32)
    stale = copy.deepcopy(cap)
    stale["kernel_policy"]["training"]["qkv"]["forward"] = {"m": 16, "n": 32, "k": 16}
    with pytest.raises(ValueError, match="policy capability"):
        report.capability(stale, policy, source, 32)


def test_thermal_evidence_cannot_be_overridden_by_valid_boolean():
    value = condition_receipt()
    value["conditions_after"]["thermal"]["stdout"] = "GPU_Speed_Limit = 75"
    with pytest.raises(ValueError, match="throttling"):
        report.conditions(value)


def test_failed_or_missing_repeatability_cannot_yield_final_report(tmp_path):
    write_json(
        tmp_path / "summary.json",
        {
            "scope": "development_only",
            "engines": ["metal"],
            "widths": list(report.WIDTHS),
            "stages": ["full_model", "training"],
            "formal_admission_opened": False,
            "complete": True,
            "exact_passed": True,
            "accuracy_passed": True,
            "workers": [],
            "runtime_identity_sha256": "a" * 64,
        },
    )
    with pytest.raises(ValueError, match="twelve fresh workers"):
        report.repeatability(
            tmp_path, [{"runtime_identity": {"inventory_sha256": "a" * 64}}], report.old.Receipts()
        )


def preflight_fixture(tmp_path, mutation=None):
    from transformer.gpu_optimize import pipeline_resource_checks

    candidate = report.inventory()[0]
    base = with_qkv(reference())
    policy = report.apply_candidate(base, candidate)
    source = {
        "native/metal/backend.mm": "a" * 64,
        "native/metal/build/libtransformer_metal.dylib": "b" * 64,
    }
    pipelines = (
        "v3_qkv_forward",
        "qkv_dx",
        "attention_dqkv",
        "linear_forward",
        "linear_dx",
        "linear_dw",
        "norm_forward",
        "norm_dx",
        "norm_dparam",
        "linear_db",
        "attention_scores",
        "attention_softmax",
        "attention_values",
        "attention_da",
        "attention_ds",
        "embedding_grad",
        "adam",
    )
    cap = {
        "kernel_policy": bundle(policy),
        "kernel_policy_sha256": identity(bundle(policy)),
        "host_source_sha256": "a" * 64,
        "compiled_host_source_sha256": "a" * 64,
        "binary_sha256": "b" * 64,
        "backend": "handwritten-metal-v3",
        "framework_numerical_calls": False,
        "storage_dtype": "float32",
        "families": ["Apple9"],
        "width": 32,
        "parameter_count": 17952,
        "max_threadgroup_memory_bytes": 32768,
        "pipelines": {
            prefix + name: {
                "max_threads_per_group": 1024,
                "thread_execution_width": 32,
                "static_threadgroup_memory_bytes": 2048,
            }
            for prefix in ("", "inference:")
            for name in pipelines
        },
    }
    entry = {
        "id": candidate["id"],
        "status": "supported",
        "policy": policy,
        "policy_sha256": identity(policy),
        "capabilities": cap,
        "resources": pipeline_resource_checks(policy, cap),
    }
    if mutation:
        mutation(entry)
    path = tmp_path / "preflight.json.entries" / (candidate["id"] + ".json")
    write_json(path, entry)
    preflight = {
        "source_sha256": source,
        "numeric_dispatches": 0,
        "timing_observations": 0,
        "eligibility_frozen_before_next_scoring_phase": True,
        "records": [
            {k: entry[k] for k in ("id", "status", "policy_sha256")}
            | {"evidence": path.name, "evidence_sha256": report.old.digest(path)}
        ],
    }
    write_json(tmp_path / "preflight.json", preflight)
    return candidate, base, source, entry, path


def test_preflight_summary_hash_cannot_hide_modified_support_evidence(tmp_path):
    candidate, base, source, entry, path = preflight_fixture(tmp_path)
    report.preflight(
        tmp_path,
        "preflight.json",
        [candidate],
        source,
        report.old.Receipts(),
        base,
        32,
    )
    entry["status"] = "resource_rejected_unranked"
    write_json(path, entry)
    with pytest.raises(ValueError, match="Changed preflight evidence"):
        report.preflight(
            tmp_path,
            "preflight.json",
            [candidate],
            source,
            report.old.Receipts(),
            base,
            32,
        )


@pytest.mark.parametrize("mutation", ["capabilities", "resources", "memory", "threads", "policy"])
def test_supported_preflight_requires_bound_real_resources_and_expected_policy(tmp_path, mutation):
    def change(entry):
        if mutation in ("capabilities", "resources"):
            entry.pop(mutation)
        elif mutation == "memory":
            entry["capabilities"]["max_threadgroup_memory_bytes"] = 1
        elif mutation == "threads":
            entry["capabilities"]["pipelines"]["v3_qkv_forward"]["max_threads_per_group"] = 32
        else:
            entry["policy"]["qkv"]["forward"]["k"] = 32
            entry["policy_sha256"] = identity(entry["policy"])

    candidate, base, source, _, _ = preflight_fixture(tmp_path, change)
    with pytest.raises(ValueError):
        report.preflight(
            tmp_path, "preflight.json", [candidate], source, report.old.Receipts(), base, 32
        )


@pytest.mark.parametrize("width", report.WIDTHS)
def test_actual_full_model_shapes_and_tolerances_all_widths(width):
    report.full_model(full_model(width), width)


@pytest.mark.parametrize("mutation", ["shape", "gradient_tolerance", "adam_tolerance"])
def test_full_model_pass_flags_cannot_hide_wrong_equations(mutation):
    value = full_model()
    if mutation == "shape":
        value["checks"][0].update(actual_shape=[1], expected_shape=[1], coordinate_count=1)
    elif mutation == "gradient_tolerance":
        value["checks"][2]["tolerance"] = "forward"
    else:
        value["checks"][-1]["tolerance"] = "gradient"
    with pytest.raises(ValueError):
        report.full_model(value, 32)


def fingerprint(shape):
    return {
        "dtype": "float32",
        "shape": list(shape),
        "bytes": 4 * math.prod(shape),
        "sha256": "a" * 64,
    }


def stress_fixture(width=32):
    cases = []
    for index, name in enumerate(report.stress_names(width, True)):
        case = {"name": name, "passed": True}
        if index < 22:
            rows = (
                int(name[11:])
                if name.startswith("random_rows")
                else 32
                if name == "qkv_common_bias"
                else 3
            )
            mapping = {
                "normalized": ([rows, width], "forward"),
                "inverse_std": ([rows], "math"),
                "ln": ([rows, width], "forward"),
                "qkv": ([rows, 3 * width], "forward"),
                "dx": ([rows, width], "gradient"),
                "dgamma": ([width], "gradient"),
                "dbeta": ([width], "gradient"),
                "dweight": ([3 * width, width], "gradient"),
                "dbias": ([3 * width], "gradient"),
                "dln": ([rows, width], "gradient"),
            }
            checks = [
                numeric(name + "/joint/" + key + "_fp64", *val) for key, val in mapping.items()
            ]
            checks += [
                numeric(
                    name + "/linear_same_operands/" + key + "_fp64",
                    *mapping["qkv" if key == "output" else key],
                )
                for key in ("output", "dln", "dweight", "dbias")
            ]
            checks += [
                numeric(name + "/norm_same_incoming/" + key + "_fp64", *mapping[key])
                for key in ("dx", "dgamma", "dbeta")
            ]
            shapes = {
                "norm": {"output": [rows, width], "norm": [rows, width], "inv": [rows]},
                "linear": {
                    "output": [rows, 3 * width],
                    "dx": [rows, width],
                    "dweight": [3 * width, width],
                    "dbias": [3 * width],
                },
                "backward": {
                    "output": [rows, width],
                    "norm": [rows, width],
                    "inv": [rows],
                    "dx": [rows, width],
                    "dgamma": [width],
                    "dbeta": [width],
                },
            }
            operands = {
                "x": [rows, width],
                "gamma": [width],
                "beta": [width],
                "weight": [3 * width, width],
                "bias": [3 * width],
                "dqkv": [rows, 3 * width],
            }
            case.update(
                width=width,
                rows=rows,
                epsilon=1e-5,
                checks=checks,
                failed_checks=[],
                attempted=True,
                guards_passed=True,
                inputs_unchanged=True,
                outputs_finite=True,
                repeat_exact=True,
                repeat_checks={f"{s}/{k}": True for s, outputs in shapes.items() for k in outputs},
                output_identity={
                    s: {k: fingerprint(shape) for k, shape in outputs.items()}
                    for s, outputs in shapes.items()
                },
                fixture={k: fingerprint(shape) for k, shape in operands.items()},
            )
        elif index < 45:
            tokens = int(name.split("_t")[1]) if name.startswith(("random_t", "prefix_t")) else 32
            checks = [
                numeric(name + "/" + key, shape, tol)
                for key, shape, tol in (
                    ("output", [tokens, width], "forward"),
                    ("probabilities", [4, tokens, tokens], "math"),
                    ("dqkv", [tokens, 3 * width], "gradient"),
                    ("same_stored_probability_gradient", [tokens, 3 * width], "gradient"),
                    ("mass", [4, tokens], "math"),
                )
            ]
            controls = ["exact_repeat", "guards", "mask_exact_zero"]
            if tokens > 1:
                controls += [
                    "causal_future_intervention",
                    "prefix_gradient_exact",
                    "future_gradient_exact_zero",
                ]
            case.update(checks=checks, controls=dict.fromkeys(controls, True))
        else:
            if index < 51:
                rows, n, k = map(int, name.split("_")[1:])
                values = (
                    ("output", [rows, n], "forward"),
                    ("dx", [rows, k], "gradient"),
                    ("dweight", [n, k], "gradient"),
                    ("dbias", [n], "gradient"),
                )
            else:
                values = (
                    ("token", [128 * width], "gradient"),
                    ("position", [32 * width], "gradient"),
                )
            case.update(
                checks=[numeric(name + "/" + key, shape, tol) for key, shape, tol in values],
                exact_repeat=True,
                guards=True,
            )
        cases.append(case)
    return cases


@pytest.mark.parametrize("width", report.WIDTHS)
def test_complete_stress_equation_shapes_all_widths(width):
    report.stress(stress_fixture(width), width, True)


@pytest.mark.parametrize("index", [0, 22, 23, 45, 51])
def test_removed_stress_equation_is_failure_even_with_pass_flags(index):
    cases = stress_fixture()
    cases[index]["checks"].pop()
    with pytest.raises(ValueError, match="numerical checks"):
        report.stress(cases, 32, True)


@pytest.mark.parametrize(
    "mutation", ["norm_repeat", "causal_control", "norm_dtype", "gradient_tolerance"]
)
def test_required_stress_controls_and_operands_cannot_be_erased(mutation):
    cases = stress_fixture()
    if mutation == "norm_repeat":
        del cases[0]["repeat_checks"]["linear/dx"]
    elif mutation == "causal_control":
        del cases[23]["controls"]["causal_future_intervention"]
    elif mutation == "norm_dtype":
        cases[0]["fixture"]["x"]["dtype"] = "float16"
    else:
        cases[45]["checks"][1]["tolerance"] = "forward"
    with pytest.raises(ValueError):
        report.stress(cases, 32, True)


def test_declared_source_inventory_cannot_drop_gate_or_arithmetic_sources():
    value = {
        "source_sha256": dict.fromkeys(report.SOURCE_FILES, "a" * 64),
        "input_sha256": {"initial.f32": "b" * 64, "train.i32": "c" * 64},
        "source_revision": "d" * 40,
    }
    del value["source_sha256"]["transformer/precision_audit.py"]
    with pytest.raises(ValueError, match="Incomplete QKV source binding"):
        report.verify_sources(value, report.ROOT)


def direct_fixture(width=32):
    inventory, cases = [], []
    for index, name in enumerate(report.DIRECT_NAMES):
        rows = report.DIRECT_ROWS[index] if index < 12 else 3
        kind = "random" if index < 12 else name
        operands = {
            "x": [rows, width],
            "weight": [3 * width, width],
            "bias": [3 * width],
            "dy": [rows, 3 * width],
        }
        outputs = {
            "output": [rows, 3 * width],
            "dx": [rows, width],
            "dweight": [3 * width, width],
            "dbias": [3 * width],
        }
        inventory.append(
            {
                "name": name,
                "rows": rows,
                "width": width,
                "kind": kind,
                **{key + "_shape": shape for key, shape in operands.items()},
            }
        )
        control_names = [
            "oracle_keys",
            "oracle_crosscheck",
            "reference_input_immutability",
            "candidate_input_immutability",
            "reference_keys",
            "candidate_keys",
            "reference_repeat_exact",
            "candidate_repeat_exact",
            "reference_guards",
            "candidate_guards",
        ]
        control_names += [
            f"{label}/{key}/dtype" for key in outputs for label in ("reference", "candidate")
        ]
        control_names += [key + "/reference_exact" for key in outputs]
        checks = [
            numeric(
                f"{name}/{label}/{key}/fp64", shape, "forward" if key == "output" else "gradient"
            )
            for key, shape in outputs.items()
            for label in ("reference", "candidate")
        ]
        locations = {
            "output": [(0, 0), (rows // 2, 3 * width // 2), (rows - 1, 3 * width - 1)],
            "dx": [(0, 0), (rows // 2, width // 2), (rows - 1, width - 1)],
            "dweight": [(0, 0), (3 * width // 2, width // 2), (3 * width - 1, width - 1)],
            "dbias": [(0,), (3 * width // 2,), (3 * width - 1,)],
        }
        oracle = [
            {
                "name": key,
                "coordinate": list(coordinate),
                "actual": 0.0,
                "expected": 0.0,
                "absolute_error": 0.0,
                "allowed": 1e-10,
                "passed": True,
            }
            for key, positions in locations.items()
            for coordinate in sorted(set(positions))
        ]
        cases.append(
            {
                "name": name,
                "rows": rows,
                "width": width,
                "kind": kind,
                "passed": True,
                "controls": dict.fromkeys(control_names, True),
                "checks": checks,
                "oracle_crosschecks": oracle,
                "operands": {key: fingerprint(shape) for key, shape in operands.items()},
                "output_identities": {
                    label: {key: fingerprint(shape) for key, shape in outputs.items()}
                    for label in ("reference", "candidate")
                },
            }
        )
    return {
        "contract": "gpu-v3-qkv-ordered-development-v1",
        "seed": 20261103,
        "scope": "development_only_no_admission_operands_or_formal_measurement",
        "family": 53,
        "width": width,
        "inventory_complete": True,
        "passed": True,
        "inventory": inventory,
        "cases": cases,
    }


@pytest.mark.parametrize("width", report.WIDTHS)
def test_complete_direct_qkv_shapes_controls_and_fp64_locations(width):
    report.direct_qkv(direct_fixture(width), width)


@pytest.mark.parametrize(
    "mutation",
    [
        "control",
        "duplicate_oracle",
        "wrong_oracle",
        "shape",
        "tolerance",
        "operand_dtype",
        "output_hash",
        "inventory",
    ],
)
def test_direct_qkv_false_pass_mutations(mutation):
    value = direct_fixture()
    case = value["cases"][0]
    if mutation == "control":
        del case["controls"]["reference_input_immutability"]
    elif mutation == "duplicate_oracle":
        case["oracle_crosschecks"][-1] = case["oracle_crosschecks"][0].copy()
    elif mutation == "wrong_oracle":
        case["oracle_crosschecks"][0]["coordinate"] = [99, 99]
    elif mutation == "shape":
        case["checks"][0].update(actual_shape=[1], expected_shape=[1], coordinate_count=1)
    elif mutation == "tolerance":
        case["checks"][2]["tolerance"] = "forward"
    elif mutation == "operand_dtype":
        case["operands"]["x"]["dtype"] = "float16"
    elif mutation == "output_hash":
        for label in ("reference", "candidate"):
            case["output_identities"][label]["dx"]["sha256"] = "invented"
    else:
        value["inventory"][0]["rows"] = 2
    with pytest.raises(ValueError):
        report.direct_qkv(value, 32)


def test_wrong_width_capabilities_cannot_qualify_timing(tmp_path):
    _, _, source, entry, _ = preflight_fixture(tmp_path)
    with pytest.raises(ValueError, match="model width"):
        report.capability(entry["capabilities"], entry["policy"], source, 64)
