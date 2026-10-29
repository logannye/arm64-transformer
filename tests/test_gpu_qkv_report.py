"""Negative controls for saved QKV evidence; no GPU or model execution."""

import copy
import hashlib
import json

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


def numeric(name):
    return {
        "name": name,
        "passed": True,
        "failed_coordinates": 0,
        "actual_shape": [1],
        "expected_shape": [1],
        "coordinate_count": 1,
        "failure_examples": [],
        "max_absolute_error": 0.0,
        "max_tolerance_ratio": 0.0,
        "rmse": 0.0,
        "tolerance": "forward",
    }


def full_model():
    names = [
        f"B{b}/{n}"
        for b in (1, 16)
        for n in ("logits", "loss", *(f"gradient/{p}" for p in report.PARAMETERS))
    ]
    names += [f"fixed_gradient_adam/{n}" for n in ("m", "v", "parameters")]
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
        "checks": [numeric(n) for n in names],
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
    report.full_model(value)
    value["checks"].pop(10)
    with pytest.raises(ValueError, match="numerical checks"):
        report.full_model(value)
    value = full_model()
    del value["controls"]["B16/sequence_isolation"]
    with pytest.raises(ValueError, match="behavioral control"):
        report.full_model(value)


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
    }
    report.capability(cap, policy, source)
    stale = copy.deepcopy(cap)
    stale["compiled_host_source_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Stale native"):
        report.capability(stale, policy, source)
    stale = copy.deepcopy(cap)
    stale["kernel_policy"]["training"]["qkv"]["forward"] = {"m": 16, "n": 32, "k": 16}
    with pytest.raises(ValueError, match="policy capability"):
        report.capability(stale, policy, source)


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
        },
    )
    with pytest.raises(ValueError, match="twelve fresh workers"):
        report.repeatability(tmp_path, [], report.old.Receipts())


def test_preflight_summary_hash_cannot_hide_modified_support_evidence(tmp_path):
    candidate = report.inventory()[0]
    policy = report.apply_candidate(with_qkv(reference()), candidate)
    entry = {
        "id": candidate["id"],
        "status": "supported",
        "policy": policy,
        "policy_sha256": identity(policy),
    }
    path = tmp_path / "preflight.json.entries" / (candidate["id"] + ".json")
    write_json(path, entry)
    preflight = {
        "source_sha256": {"source": "a" * 64},
        "numeric_dispatches": 0,
        "timing_observations": 0,
        "eligibility_frozen_before_next_scoring_phase": True,
        "records": [
            {k: entry[k] for k in ("id", "status", "policy_sha256")}
            | {"evidence": path.name, "evidence_sha256": report.old.digest(path)}
        ],
    }
    write_json(tmp_path / "preflight.json", preflight)
    report.preflight(
        tmp_path, "preflight.json", [candidate], preflight["source_sha256"], report.old.Receipts()
    )
    entry["status"] = "resource_rejected_unranked"
    write_json(path, entry)
    with pytest.raises(ValueError, match="Changed preflight evidence"):
        report.preflight(
            tmp_path,
            "preflight.json",
            [candidate],
            preflight["source_sha256"],
            report.old.Receipts(),
        )
