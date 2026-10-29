"""Validate and report saved QKV development evidence, without model execution.

The final direct paired comparison is the only source of reported speed ratios.
Candidate timings select a policy; their ratios are never multiplied. A report
without fresh-process repeatability evidence is explicitly an interim report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import report_gpu_v2_engineering as old  # noqa: E402
from transformer.gpu_policy import bundle, identity, validate, with_qkv  # noqa: E402

WIDTHS, MODES = old.WIDTHS, old.MODES
SCOPE = "qkv_training_partition_development_only"
require = old.require
PARAMETERS = (
    "token",
    "position",
    "ln1_weight",
    "ln1_bias",
    "qkv_weight",
    "qkv_bias",
    "attention_weight",
    "attention_bias",
    "ln2_weight",
    "ln2_bias",
    "ff1_weight",
    "ff1_bias",
    "ff2_weight",
    "ff2_bias",
    "ln3_weight",
    "ln3_bias",
    "head_weight",
    "head_bias",
)
TOLERANCES = {
    "forward": (2e-5, 2e-4),
    "gradient": (3e-5, 3e-3),
    "math": (2e-7, 2e-6),
    "oracle": (1e-10, 1e-10),
}
DIRECT_ROWS = (1, 3, 9, 16, 17, 31, 32, 33, 48, 96, 511, 512)
DIRECT_NAMES = [f"random_rows{n}" for n in DIRECT_ROWS] + [
    "forward_cancellation",
    "dx_cancellation",
    "common_bias",
    "mixed_exponents",
]
LEDGER_KEYS = (
    "schema",
    "scope",
    "width",
    "source_revision",
    "source_sha256",
    "runtime_identity",
    "input_sha256",
    "incumbent_policy",
    "incumbent_policy_file_sha256",
    "candidate_inventory",
    "retained_v2_policy",
    "family_order",
    "min_improvement",
    "warmups",
    "observations",
    "blocks",
    "environment",
)


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def hash_map(value, label):
    require(isinstance(value, dict) and value and all(sha(v) for v in value.values()), label)


def same_number(actual, expected, label):
    require(
        isinstance(actual, (int, float))
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15),
        label,
    )


def inventory():
    return [
        {
            "id": f"qkv-{direction}-{m}-{n}-{k}",
            "family": direction,
            "value": {"m": m, "n": n, "k": k},
        }
        for direction in ("forward", "dx")
        for m, n, k in itertools.product((16, 32), (32, 64), (16, 32))
    ]


def apply_candidate(policy, candidate):
    result = with_qkv(policy)
    result["qkv"][candidate["family"]] = candidate["value"]
    return validate(result)


def paired_summary(pair, mode, order_offset=0):
    summary = old.paired_summary(pair, mode=mode)
    expected = [
        ["reference", "candidate"] if (i + order_offset) % 2 == 0 else ["candidate", "reference"]
        for i in range(5)
    ]
    require(pair.get("block_order") == expected, "Nonalternating or reordered paired blocks")
    improvements = [1 - v for v in summary["paired_ratios"]]
    require(
        len(pair.get("paired_improvement_fractions", [])) == 5,
        "Missing paired improvement observations",
    )
    for actual, expected in zip(pair["paired_improvement_fractions"], improvements, strict=True):
        same_number(actual, expected, "Paired improvement disagrees with raw timings")
    same_number(
        pair.get("median_paired_improvement_fraction"),
        statistics.median(improvements),
        "Median improvement disagrees with raw timings",
    )
    expected_boundary = (
        "train_step + finite scan + scalar loss readback"
        if mode == "training"
        else "forward + GPU completion; finite/guard/readback checks outside timing"
    )
    require(pair.get("boundary") == expected_boundary, "Changed timing boundary")
    for label in ("reference", "candidate"):
        same_number(
            pair[label].get("median_seconds"),
            statistics.median(pair[label]["seconds"]),
            "Reported sample median disagrees with raw timings",
        )
    summary["speed_ratio_reference_over_selected"] = 1 / summary["candidate_over_reference_latency"]
    return summary


def comparison(check):
    require(
        check.get("passed") is True and check.get("failed_coordinates") == 0,
        "Failed or incomplete numerical comparison",
    )
    require(
        check.get("actual_shape") == check.get("expected_shape")
        and isinstance(check.get("actual_shape"), list),
        "Comparison shape mismatch",
    )
    count = math.prod(check["actual_shape"])
    require(check.get("coordinate_count") == count and count > 0, "Wrong comparison size")
    require(check.get("failure_examples") == [], "Retained numerical failures")
    for name in ("max_absolute_error", "max_tolerance_ratio", "rmse"):
        value = check.get(name)
        require(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
            "Nonfinite or missing numerical error",
        )
    require(check["max_tolerance_ratio"] <= 1, "Numerical tolerance exceeded")
    require(check.get("tolerance") in (*TOLERANCES, "optimizer"), "Unknown numerical tolerance")


def checks(values, names=None):
    require(isinstance(values, list) and values, "Missing numerical checks")
    actual = [v.get("name") for v in values]
    require(len(set(actual)) == len(actual), "Duplicate numerical checks")
    if names is not None:
        require(actual == names, "Missing, duplicate, or reordered numerical checks")
    for value in values:
        comparison(value)


def controls(value, names=None):
    require(
        isinstance(value, dict) and value and all(v is True for v in value.values()),
        "Missing or failed behavioral controls",
    )
    if names is not None:
        require(set(value) == set(names), "Missing behavioral control")


def full_model(value):
    require(value.get("passed") is True, "Full-model qualification failed")
    names = [
        f"B{b}/{n}"
        for b in (1, 16)
        for n in ("logits", "loss", *(f"gradient/{p}" for p in PARAMETERS))
    ]
    names += [f"fixed_gradient_adam/{n}" for n in ("m", "v", "parameters")]
    checks(value.get("checks"), names)
    expected = [
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
    expected += [
        "B1/causal_future_intervention",
        "B16/sequence_isolation",
        "fixed_gradient_adam_repeat",
        "fixed_gradient_adam_step",
    ]
    controls(value.get("controls"), expected)


def stress_names(width, complete):
    names = [f"random_rows{n}" for n in (1, 3, 32, 33, 511, 512)]
    names += [
        f"offset_{offset}_spread_{spread}"
        for offset in (0, 64, 4096)
        for spread in ("0", "0.000244141", "0.01", "1")
    ]
    names += [
        "affine_signed_scale",
        "affine_zero_channels",
        "qkv_common_bias",
        "qkv_reduction_cancellation",
    ]
    if complete:
        names += [f"random_t{n}" for n in (1, 3, 32)]
        names += [
            f"offset_{offset}_gap_{gap}"
            for offset in (0, 64, 512, 4096)
            for gap in ("0", "0.125", "0.75")
        ]
        names += [f"prefix_t{n}" for n in (2, 3, 5, 31)]
        names += ["negative_offset", "cancellation_16", "cancellation_64", "cancellation_256"]
        names += [
            "dense_1_1_1",
            "dense_5_19_17",
            "dense_31_65_33",
            f"dense_3_{width}_{width}",
            f"dense_32_{2 * width}_{width}",
            f"dense_33_{width}_{2 * width}",
            "all_same_33_rows",
            "alternating_63_rows",
        ]
    return names


def stress(value, width, complete):
    require(
        [v.get("name") for v in value] == stress_names(width, complete),
        "Incomplete or reordered stress inventory",
    )
    for index, case in enumerate(value):
        require(case.get("passed") is True, "Stress case failed")
        checks(case.get("checks"))
        if index < 22:
            require(
                all(
                    case.get(k) is True
                    for k in (
                        "attempted",
                        "guards_passed",
                        "inputs_unchanged",
                        "outputs_finite",
                        "repeat_exact",
                    )
                ),
                "Normalization/QKV control failed",
            )
            require(case.get("failed_checks") == [], "Failed normalization/QKV checks")
            controls(case.get("repeat_checks"))
        elif index < 45:
            controls(case.get("controls"))
        else:
            require(
                case.get("exact_repeat") is True and case.get("guards") is True,
                "Dense/embedding behavioral failure",
            )


def direct_qkv(value, width):
    require(
        value.get("contract") == "gpu-v3-qkv-ordered-development-v1"
        and value.get("seed") == 20261103
        and value.get("family") == 53
        and value.get("width") == width
        and value.get("passed") is True
        and value.get("inventory_complete") is True,
        "Wrong direct QKV contract",
    )
    cases = value.get("cases", [])
    require([v.get("name") for v in cases] == DIRECT_NAMES, "Incomplete direct QKV cases")
    require(
        [v.get("name") for v in value.get("inventory", [])] == DIRECT_NAMES,
        "Incomplete direct QKV inventory",
    )
    for case in cases:
        require(
            case.get("passed") is True and case.get("width") == width,
            "Direct QKV qualification failed",
        )
        controls(case.get("controls"))
        expected = [
            f"{case['name']}/{label}/{name}/fp64"
            for name in ("output", "dx", "dweight", "dbias")
            for label in ("reference", "candidate")
        ]
        checks(case.get("checks"), expected)
        require(
            set(case.get("operands", {})) == {"x", "weight", "bias", "dy"},
            "Missing direct QKV operands",
        )
        for operand in case["operands"].values():
            require(
                sha(operand.get("sha256")) and operand.get("dtype") == "float32",
                "Missing direct QKV operand identity",
            )
        output = case.get("output_identities", {})
        require(
            set(output) == {"reference", "candidate"}
            and set(output["reference"]) == {"output", "dx", "dweight", "dbias"}
            and output["reference"] == output["candidate"],
            "QKV reference bytes changed",
        )
        oracle = case.get("oracle_crosschecks", [])
        require(len(oracle) == 12, "Missing independent FP64 crosschecks")
        for check in oracle:
            expected_limit = 1e-10 + 1e-10 * abs(check["expected"])
            same_number(check.get("allowed"), expected_limit, "Changed oracle tolerance")
            error = abs(check["actual"] - check["expected"])
            same_number(check.get("absolute_error"), error, "Oracle error mismatch")
            require(
                check.get("passed") is True and error <= expected_limit,
                "Independent FP64 oracle failure",
            )


def qualification(value, width, complete):
    require(value.get("passed") is True, "Unqualified candidate")
    standard = value.get("qualification", {})
    require(standard.get("passed") is True, "Missing standard qualification")
    full_model(standard.get("full_model", {}))
    stress(standard.get("stress", []), width, complete)
    direct_qkv(value.get("direct_qkv", {}), width)


def byte_equivalence(value):
    require(value.get("passed") is True, "Retained v2 byte equivalence failed")
    controls(
        value.get("checks"),
        ("training/forward", "inference/forward", "loss_and_gradient", "adam_state"),
    )


def capability(cap, policy, source):
    require(
        cap.get("kernel_policy") == bundle(policy)
        and cap.get("kernel_policy_sha256") == identity(bundle(policy)),
        "Kernel policy capability mismatch",
    )
    require(
        cap.get("binary_sha256") == source.get("native/metal/build/libtransformer_metal.dylib")
        and cap.get("host_source_sha256") == source.get("native/metal/backend.mm")
        and cap.get("compiled_host_source_sha256") == source.get("native/metal/backend.mm"),
        "Stale native binary/source identity",
    )
    require(
        cap.get("backend") == "handwritten-metal-v3"
        and cap.get("framework_numerical_calls") is False
        and cap.get("storage_dtype") == "float32",
        "Changed backend or stored precision",
    )


def fresh_pair(value, before, after, source):
    summary = old.validate_fresh_pair(value)
    for label, policy in (("reference", before), ("candidate", after)):
        capability(value.get("capabilities", {}).get(label, {}), policy, source)
    return summary


def conditions(value):
    for when in ("before", "after"):
        state = value.get(f"conditions_{when}", {})
        assessment = value.get(f"conditions_{when}_assessment", {})
        require(
            assessment.get("valid") is True and assessment.get("ac_power_confirmed") is True,
            "Invalid execution conditions",
        )
        power, thermal = state.get("power", {}), state.get("thermal", {})
        require(
            power.get("returncode") == 0 and "AC Power" in power.get("stdout", ""),
            "AC power not demonstrated",
        )
        text = thermal.get("stdout", "")
        limits = re.findall(
            r"\b(?:CPU_Speed_Limit|CPU_Scheduler_Limit|GPU_Speed_Limit)\s*=\s*(\d+)", text
        )
        require(not any(int(v) < 100 for v in limits), "Known thermal throttling")
        require(
            not re.search(
                r"\bthermal\s+(?:pressure|state)\s*[:=]\s*"
                r"(?:serious|critical|heavy|throttled)\b",
                text,
                re.I,
            ),
            "Known thermal pressure",
        )


def verify_sources(value, repository):
    source = value.get("source_sha256")
    hash_map(source, "Missing source hashes")
    hash_map(value.get("input_sha256"), "Missing input hashes")
    require(set(value["input_sha256"]) == {"initial.f32", "train.i32"}, "Wrong data scope")
    revision = value.get("source_revision", "")
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Missing committed revision")
    required = {
        "native/metal/qkv.metal",
        "native/metal/backend.mm",
        "transformer/gpu_policy.py",
        "transformer/gpu_qkv_optimize.py",
        "scripts/check_metal_qkv_optimization.py",
        "native/metal/build/libtransformer_metal.dylib",
    }
    require(required <= source.keys(), "Incomplete QKV source binding")
    for relative, expected in source.items():
        path = old.safe_file(repository, relative)
        require(old.digest(path) == expected, f"Stale source or binary: {relative}")
        if not relative.endswith(".dylib"):
            data = subprocess.run(
                ["git", "show", f"{revision}:{relative}"],
                cwd=repository,
                capture_output=True,
                check=False,
            )
            require(
                data.returncode == 0 and hashlib.sha256(data.stdout).hexdigest() == expected,
                f"Source was not frozen in the declared commit: {relative}",
            )


def preflight(root, relative, candidates, source, receipts):
    value = receipts.read(root, relative, "search")
    require(
        value.get("source_sha256") == source
        and value.get("eligibility_frozen_before_next_scoring_phase") is True
        and value.get("numeric_dispatches") == 0
        and value.get("timing_observations") == 0,
        "Invalid compile/resource preflight",
    )
    records = value.get("records", [])
    require(
        [r.get("id") for r in records] == [c["id"] for c in candidates],
        "Incomplete preflight inventory",
    )
    for record in records:
        entry_relative = str(Path(str(relative) + ".entries") / record["evidence"])
        entry = receipts.read(root, entry_relative, "search")
        require(
            old.digest(old.safe_file(root, entry_relative)) == record.get("evidence_sha256"),
            "Changed preflight evidence",
        )
        require(
            all(entry.get(k) == record.get(k) for k in ("id", "status", "policy_sha256")),
            "Preflight summary mismatch",
        )
        require(
            identity(validate(entry["policy"])) == record["policy_sha256"],
            "Preflight policy hash mismatch",
        )
    return {r["id"]: r["status"] == "supported" for r in records}


def select_group(group, mode, family, current, root, receipts, source, supported, composed):
    require(
        group.get("mode") == mode
        and group.get("family") == family
        and group.get("before_policy") == current,
        "Broken candidate policy chain",
    )
    candidates = [c for c in inventory() if c["family"] == family]
    rows = group.get("candidates", [])
    require(
        [r.get("id") for r in rows] == [c["id"] for c in candidates],
        "Missing, duplicated, or reordered candidate outcomes",
    )
    measured, outcomes = [], Counter()
    for index, (row, candidate) in enumerate(zip(rows, candidates, strict=True)):
        require(all(row.get(k) == v for k, v in candidate.items()), "Changed candidate identity")
        policy = apply_candidate(current, candidate)
        require(
            row.get("mode") == mode
            and row.get("policy") == policy
            and row.get("policy_sha256") == identity(policy),
            "Changed candidate policy",
        )
        evidence = row.get("evidence", {})
        saved = receipts.read(root, evidence["path"], "search")
        require(
            old.digest(old.safe_file(root, evidence["path"])) == evidence.get("sha256")
            and saved == {k: v for k, v in row.items() if k != "evidence"},
            "Changed candidate evidence or removed failure",
        )
        status = row.get("status")
        outcomes[status] += 1
        if not supported[row["id"]] or not composed[row["id"]]:
            require(
                status == "compile_or_resource_rejected_unranked" and "paired_timing" not in row,
                "Unsupported candidate received a ranking",
            )
        elif status == "numerically_rejected_unranked":
            require(
                row.get("qualification", {}).get("passed") is False and "paired_timing" not in row,
                "Numerically failed candidate was ranked",
            )
        else:
            require(status == "qualified_measured", "Incomplete candidate outcome")
            qualification(row["qualification"], group["width"], False)
            conditions(row)
            fresh_pair(row.get("fresh_pair", {}), current, policy, source)
            summary = paired_summary(row.get("paired_timing", {}), mode, index % 2)
            measured.append((summary["candidate_over_reference_latency"], row["id"], policy))
    eligible = [item for item in measured if item[0] <= 0.97]
    winner = min(eligible, key=lambda x: (x[0], x[1])) if eligible else None
    expected_id, after = (winner[1], winner[2]) if winner else (None, current)
    require(
        group.get("chosen_id") == expected_id and group.get("after_policy") == after,
        "Selection violates 3% threshold, ranking, or null retention",
    )
    return after, {
        "mode": mode,
        "family": family,
        "chosen_id": expected_id,
        "outcomes": dict(outcomes),
        "eligible_candidates": len(eligible),
    }


def read_search(root, width, receipts, repository=ROOT):
    relative = f"width-{width}.json"
    result = receipts.read(root, relative, "search")
    require(
        result.get("schema") == 1
        and result.get("scope") == SCOPE
        and result.get("status") == "completed"
        and result.get("width") == width,
        "Incomplete or wrong-scope QKV search",
    )
    require(
        result.get("formal_fits") == 0 and result.get("admission_stream_opened") is False,
        "Formal data entered a development report",
    )
    require(
        all(
            result.get(k) is True
            for k in ("source_unchanged", "runtime_unchanged", "input_unchanged")
        ),
        "Experiment identity changed",
    )
    verify_sources(result, repository)
    ledger_name = relative + ".ledger.json"
    ledger = receipts.read(root, ledger_name, "search")
    require(
        old.digest(old.safe_file(root, ledger_name)) == result.get("ledger_sha256"),
        "Changed frozen ledger",
    )
    require(
        all(ledger.get(k) == result.get(k) for k in LEDGER_KEYS)
        and ledger.get("frozen_before_performance_scoring") is True
        and ledger.get("profiles") == result.get("incumbent_profiles"),
        "Result disagrees with frozen candidate ledger",
    )
    require(
        result.get("candidate_inventory") == inventory()
        and result.get("min_improvement") == 0.03
        and (result.get("warmups"), result.get("observations"), result.get("blocks"))
        == (20, 100, 5),
        "Changed candidate grid or measurement schedule",
    )
    retained_path = old.safe_file(repository, "experiments/gpu-v2/policies/selected.json")
    retained = json.loads(retained_path.read_text())["widths"][str(width)]
    require(
        result.get("retained_v2_policy") == retained
        and result.get("incumbent_policy_file_sha256") == old.digest(retained_path),
        "Substituted retained v2 incumbent",
    )
    original, selected = bundle(result["incumbent_policy"]), bundle(result["selected_policy"])
    require(
        original == {m: with_qkv(p) for m, p in retained.items()},
        "Null-QKV incumbent changed v2 policy",
    )
    byte_equivalence(result.get("incumbent_v2_byte_equivalence", {}))
    byte_equivalence(result.get("selected_v2_byte_equivalence", {}))
    conditions(result)
    before = {m: old.profile_summary(result["incumbent_profiles"][m]) for m in MODES}
    after = {m: old.profile_summary(result["selected_profiles"][m]) for m in MODES}
    groups = result.get("candidates", [])
    summaries, expected_group_order = [], []
    for mode in MODES:
        families = ["forward", "dx"] if mode == "training" else ["forward"]
        costs = {
            f: sum(
                v
                for k, v in before[mode]["stage_median_gpu_seconds"].items()
                if k.startswith("qkv_" + f)
            )
            for f in families
        }
        require(all(v > 0 for v in costs.values()), "Missing QKV profile stage")
        order = sorted(families, key=lambda f: (-costs[f], f))
        require(result["family_order"].get(mode) == order, "Changed measured family order")
        qualification(result["incumbent_qualification"][mode], width, True)
        qualification(result["final_qualification"][mode], width, True)
        active = [c for c in inventory() if mode == "training" or c["family"] == "forward"]
        supported = preflight(
            root,
            f"width-{width}.evidence/{mode}-isolated-preflight.json",
            active,
            result["source_sha256"],
            receipts,
        )
        current = original[mode]
        for family in order:
            index = len(expected_group_order)
            expected_group_order.append((mode, family))
            require(index < len(groups), "Missing candidate family")
            composed = preflight(
                root,
                f"width-{width}.evidence/{mode}-{family}-composed-preflight.json",
                [c for c in inventory() if c["family"] == family],
                result["source_sha256"],
                receipts,
            )
            current, summary = select_group(
                {**groups[index], "width": width},
                mode,
                family,
                current,
                root,
                receipts,
                result["source_sha256"],
                supported,
                composed,
            )
            summaries.append(summary)
        require(current == selected[mode], "Final policy differs from selected policy chain")
    require(len(groups) == len(expected_group_order), "Unexpected candidate family")
    full_model(result.get("selected_bundle_full_model", {}))
    memory = fresh_pair(
        result.get("final_pair_memory", {}), original, selected, result["source_sha256"]
    )
    timing = {m: paired_summary(result["selected_vs_incumbent"][m], m) for m in MODES}
    return {
        "width": width,
        "source_revision": result["source_revision"],
        "source_sha256": result["source_sha256"],
        "runtime_identity": result["runtime_identity"],
        "input_sha256": result["input_sha256"],
        "selected_policy": selected,
        "incumbent_policy": original,
        "selection": summaries,
        "timing": timing,
        "profiles": {"incumbent": before, "selected": after},
        "matched_allocations": memory,
        "capabilities": result["final_pair_memory"]["capabilities"]["candidate"],
        "development_numerical_checks_passed": True,
        "v2_full_model_byte_equivalence_passed": True,
    }


def repeat_accuracy(value, name):
    required = (
        {"logits", "loss", "gradient", "independent_forward_oracle"}
        if name.startswith("full_model/")
        else {"checkpoint_logits", "independent_forward_oracle"}
    )
    require(set(value) == required, "Missing repeatability accuracy checks")
    for key, check in value.items():
        tolerance = (
            "oracle"
            if key == "independent_forward_oracle"
            else ("gradient" if key == "gradient" else "forward")
        )
        require(
            check.get("tolerance") == tolerance
            and (check.get("atol"), check.get("rtol")) == TOLERANCES[tolerance]
            and check.get("passed") is True
            and check.get("failed_coordinates") == 0
            and check.get("nonfinite_coordinates") == 0,
            "Repeatability accuracy failure or changed tolerance",
        )


def retained_arrays(root, relative, expected, receipts):
    """Hash original saved arrays; this is not a model or oracle evaluation."""
    import numpy as np

    path = old.safe_file(root, relative)
    before = old.digest(path)
    with np.load(path, allow_pickle=False) as arrays:
        require(set(arrays.files) == set(expected), "Retained replay array inventory changed")
        for name in arrays.files:
            value = arrays[name]
            fingerprint = {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
            require(fingerprint == expected[name], "Retained replay bytes differ from receipt")
            if name == "step":
                require(
                    value.shape == () and value.dtype.str == "<i8" and int(value) == 6,
                    "Repeatability training budget changed",
                )
    require(old.digest(path) == before, "Replay arrays changed while reading")
    receipts.sources[f"repeatability/{Path(relative).as_posix()}"] = before


def repeat_worker(worker, search, root, relative, receipts):
    width = search["width"]
    expected_names = [
        f"{stage}/b{batch}-t{length}"
        for stage in ("full_model", "training")
        for batch in (1, 3, 16)
        for length in (3, 32)
    ]
    require(
        worker.get("contract") == "gpu-v2-standard-baseline-repeatability-v1"
        and worker.get("scope") == "development_only"
        and worker.get("engine") == "metal"
        and worker.get("width") == width
        and worker.get("stages") == ["full_model", "training"]
        and worker.get("repetitions") == 10,
        "Wrong selected-Metal repeatability contract",
    )
    require(
        all(
            worker.get(k) is True
            for k in (
                "complete",
                "inventory_complete",
                "exact_passed",
                "accuracy_passed",
                "source_unchanged",
            )
        )
        and worker.get("cleanup_errors") == [],
        "Incomplete or failed repeatability worker",
    )
    require(
        worker.get("source_sha256") == worker.get("source_sha256_after"), "Worker source changed"
    )
    hashes = worker.get("source_sha256", {})
    hash_map(hashes, "Missing worker source hashes")
    required = {
        "native/metal/qkv.metal",
        "native/metal/backend.mm",
        "native/metal/kernels.metal",
        "native/metal/optimized.metal",
        "transformer/gpu_metal.py",
        "transformer/gpu_policy.py",
    }
    require(required <= hashes.keys(), "Worker omits QKV source identity")
    require(
        all(
            search["source_sha256"][k] == v
            for k, v in hashes.items()
            if k in search["source_sha256"]
        ),
        "Worker and timing source identities differ",
    )
    cap = worker.get("capabilities", {})
    capability(cap, search["selected_policy"], search["source_sha256"])
    for key in (
        "binary_sha256",
        "shader_sha256",
        "training_shader_sha256",
        "inference_shader_sha256",
        "compiled_host_source_sha256",
    ):
        require(
            cap.get(key) == search["capabilities"].get(key) and sha(cap.get(key)),
            "Repeatability does not bind the measured binary/shaders",
        )
    cases = worker.get("cases", [])
    require(
        worker.get("expected_cases") == expected_names
        and len(cases) == len(expected_names)
        and {c.get("name") for c in cases} == set(expected_names),
        "Missing or duplicated repeatability cases",
    )
    fingerprints = {}
    for case in cases:
        name = case["name"]
        require(
            all(case.get(k) is True for k in ("complete", "exact_passed", "accuracy_passed")),
            "Incomplete repeatability case",
        )
        executions = case.get("executions", [])
        require(
            [e.get("execution") for e in executions] == list(range(10)),
            "Missing or reordered repeat execution",
        )
        names = (
            {"logits", "loss", "gradient"}
            if name.startswith("full_model/")
            else {"parameters", "m", "v", "step", "logits", "losses"}
        )
        first = {}
        for execution in executions:
            require(
                execution.get("finite") is True and "error" not in execution,
                "Repeatability runtime or finite-state failure",
            )
            exact = execution.get("exact", {})
            require(
                set(exact) == names and all(old.exact_pass(c) for c in exact.values()),
                "Exact repeatability failed; one ULP is still a failure",
            )
            actual = {k: v["actual"] for k, v in exact.items()}
            require(all(sha(v["sha256"]) for v in actual.values()), "Invalid repeat fingerprint")
            first = first or actual
            require(actual == first, "Execution differs from first repeat despite pass flag")
            repeat_accuracy(execution.get("accuracy", {}), name)
        retained_arrays(root, str(Path(relative).parent / case["first_arrays"]), first, receipts)
        fingerprints[name] = first
    return fingerprints


def repeatability(root, searches, receipts):
    summary = receipts.read(root, "summary.json", "repeatability")
    require(
        summary.get("scope") == "development_only"
        and summary.get("engines") == ["metal"]
        and summary.get("widths") == list(WIDTHS)
        and summary.get("stages") == ["full_model", "training"]
        and summary.get("formal_admission_opened") is False,
        "Wrong repeatability matrix",
    )
    require(
        all(summary.get(k) is True for k in ("complete", "exact_passed", "accuracy_passed")),
        "Repeatability summary failed",
    )
    workers = summary.get("workers", [])
    require(len(workers) == 12, "Expected twelve fresh workers")
    keys = [(w.get("engine"), w.get("width"), w.get("process")) for w in workers]
    expected = [("metal", w, p) for w in WIDTHS for p in (0, 1)]
    require(len(set(keys)) == 12 and set(keys) == set(expected), "Duplicate or missing worker")
    by_key = dict(zip(keys, workers, strict=True))
    for search in searches:
        width = search["width"]
        values = []
        for process in (0, 1):
            launch = by_key[("metal", width, process)]
            require(
                launch.get("returncode") == 0
                and all(
                    launch.get(k) is True for k in ("complete", "exact_passed", "accuracy_passed")
                ),
                "Worker failed",
            )
            worker = receipts.read(root, launch["result"], "repeatability")
            values.append(repeat_worker(worker, search, root, launch["result"], receipts))
        require(values[0] == values[1], "Fresh-process training/gradient bytes differ")
        cross = receipts.read(root, f"metal-w{width}-cross-process.json", "repeatability")
        require(
            all(cross.get(k) is True for k in ("passed", "source_equal", "inventory_equal")),
            "Fresh-process comparison failed",
        )
        cases = cross.get("cases", [])
        require(
            len(cases) == len(values[0]) and {c.get("name") for c in cases} == set(values[0]),
            "Incomplete cross-process checks",
        )
        for case in cases:
            exact = case.get("exact", {})
            require(
                case.get("passed") is True
                and case.get("inventory_equal") is True
                and set(exact) == set(values[0][case["name"]])
                and all(
                    old.exact_pass(v) and v["actual"] == values[0][case["name"]][k]
                    for k, v in exact.items()
                ),
                "Cross-process fingerprint mismatch",
            )
    return {
        "passed": True,
        "workers": 12,
        "processes_per_width": 2,
        "cases_per_worker": 12,
        "repetitions_per_case": 10,
        "scope": "selected Metal full-model and six-step training development checks",
    }


def assemble(searches_root, repeatability_root=None, repository=ROOT):
    receipts = old.Receipts()
    searches = [read_search(searches_root, w, receipts, repository) for w in WIDTHS]
    for key in ("source_revision", "source_sha256", "runtime_identity"):
        require(all(s[key] == searches[0][key] for s in searches), "Cross-width identity mismatch")
    repeats = repeatability(repeatability_root, searches, receipts) if repeatability_root else None
    feasibility = receipts.read(
        repository, "experiments/gpu-v3-qkv/baseline-feasibility.json", "source"
    )
    blocked = receipts.read(repository, "experiments/gpu-v2/blocked-study.json", "historical")
    require(
        feasibility.get("formal_campaign_allowed") is False
        and feasibility.get("reserved_admission_opened") is False
        and feasibility.get("scope")
        == "official_stable_source_inspection_only_no_runtime_qualification"
        and blocked.get("completed_fits") == 0
        and blocked.get("planned_fits") == 180,
        "Changed formal-study boundary",
    )
    return {
        "schema": 1,
        "scope": SCOPE,
        "status": "completed_development_report" if repeats else "interim_repeatability_pending",
        "formal_campaign_allowed": False,
        "formal_fits": 0,
        "framework_or_cpu_speed_comparisons_claimed": False,
        "numerically_qualified_development_result": repeats is not None,
        "widths": searches,
        "repeatability": repeats,
        "baseline_feasibility": feasibility,
        "formal_study": {
            k: blocked[k]
            for k in ("status", "planned_fits", "completed_fits", "blocked_fits", "reasons")
        },
        "evidence_sha256": receipts.sources,
        "profiling_scope": old.PROFILE_SCOPE,
        "hardware_counters": "Unavailable; no occupancy, bandwidth, or register-pressure claim",
        "uncertainty": (
            "Five paired block ratios describe timing variability, not a confidence interval"
        ),
    }


def write_csv(path, rows):
    require(bool(rows), "Empty CSV")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_figures(result, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "savefig.dpi": 170,
        }
    )
    colors = {"training": "#246A8D", "inference": "#A54A21"}
    rows = result["widths"]
    interim = (
        "\nInterim: fresh-process repeatability pending" if result["repeatability"] is None else ""
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.78, wspace=0.27)
    for ax, mode in zip(axes, MODES, strict=True):
        y = [r["timing"][mode]["speed_ratio_reference_over_selected"] for r in rows]
        low = [
            v - 1 / r["timing"][mode]["maximum_paired_ratio"] for v, r in zip(y, rows, strict=True)
        ]
        high = [
            1 / r["timing"][mode]["minimum_paired_ratio"] - v for v, r in zip(y, rows, strict=True)
        ]
        ax.bar(range(6), y, color=colors[mode], width=0.62, alpha=0.92)
        ax.errorbar(range(6), y, yerr=[low, high], fmt="none", color="#182A37", capsize=3)
        ax.axhline(1, color="#4B5563", linewidth=1, linestyle="--")
        ax.set_xticks(range(6), [str(w) for w in WIDTHS])
        ax.set_xlabel("Model width")
        ax.set_ylabel("Incumbent / selected latency (×)")
        ax.set_title(
            "B16/T32 complete training step" if mode == "training" else "B1/T32 resident forward"
        )
        ax.set_ylim(0, max([1.2, *(v + h for v, h in zip(y, high, strict=True))]) * 1.18)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        for index, (value, row, top) in enumerate(zip(y, rows, high, strict=True)):
            changed = row["selected_policy"][mode] != row["incumbent_policy"][mode]
            label = f"{value:.2f}×" + ("\nretained" if not changed else "")
            ax.text(
                index,
                value + top + ax.get_ylim()[1] * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(
        "Precision-protected QKV: complete-operation development timing" + interim, fontsize=13
    )
    fig.text(
        0.5,
        0.035,
        "Versus retained GPU v2 optimized policies, under the same new build.\n"
        "Whiskers: range of five paired blocks; no confidence interval.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.savefig(output / "qkv-development.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.78, wspace=0.27)
    for ax, mode in zip(axes, MODES, strict=True):
        for label, shift, color in (
            ("incumbent", -0.19, "#8A98A5"),
            ("selected", 0.19, colors[mode]),
        ):
            values = [
                r["profiles"][label][mode]["category_sum_of_stage_medians_seconds"].get("qkv", 0)
                * 1000
                for r in rows
            ]
            ax.bar(
                [x + shift for x in range(6)],
                values,
                width=0.36,
                color=color,
                label="Retained v2" if label == "incumbent" else "Selected QKV",
            )
        ax.set_xticks(range(6), [str(w) for w in WIDTHS])
        ax.set_yscale("log")
        ax.set_xlabel("Model width")
        ax.set_ylabel("QKV stage GPU time (ms; log scale)")
        ax.set_title(
            "Training: QKV forward + dX + dW" if mode == "training" else "Inference: QKV forward"
        )
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Where the protected contractions spend GPU time" + interim, fontsize=13)
    fig.text(
        0.5,
        0.035,
        "Intrusive stage profiles: separate synchronized command buffers, median of three runs.\n"
        "These are not production latency measurements.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.savefig(output / "qkv-stage-costs.png")
    plt.close(fig)


def write_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    timing, profile = [], []
    for row in result["widths"]:
        for mode in MODES:
            summary = row["timing"][mode]
            timing.append(
                {
                    "width": row["width"],
                    "mode": mode,
                    "policy_changed": row["incumbent_policy"][mode] != row["selected_policy"][mode],
                    **{k: v for k, v in summary.items() if not isinstance(v, list)},
                }
            )
            for label in ("incumbent", "selected"):
                for stage, seconds in row["profiles"][label][mode][
                    "stage_median_gpu_seconds"
                ].items():
                    profile.append(
                        {
                            "width": row["width"],
                            "mode": mode,
                            "policy": label,
                            "stage": stage,
                            "category": old.stage_category(stage),
                            "median_gpu_ms": 1000 * seconds,
                            "scope": old.PROFILE_SCOPE,
                        }
                    )
    write_csv(output / "timing.csv", timing)
    write_csv(output / "stage-profiles.csv", profile)
    render_figures(result, output)
    lines = [
        "# Precision-protected QKV development results",
        "",
        "Cooperative tiling reuses QKV operands while preserving each protected dot product's "
        "arithmetic order and final FP32 bytes. This report compares the selected QKV policy "
        "with the retained GPU v2 optimized policy under the same new build.",
        "",
        "**Status: " + result["status"].replace("_", " ") + ".** "
        "These are training-partition development measurements. The formal five-engine study "
        "remains blocked; zero of its 180 fits have run.",
        "",
        "Each mode uses 20 warmups and 100 synchronized observations in five alternating paired "
        "blocks, with exact state restoration and allocation-matched fresh backend instances. "
        "A candidate must pass numerical checks and reduce median complete-operation latency "
        "by at least 3% to replace its incumbent. Final ratios below come from direct selected "
        "versus original-incumbent measurements, not products of candidate ratios.",
        "",
        "![Complete-operation development timing](qkv-development.png)",
        "",
        "| Width | Training p50, v2 → selected (ms) | Training ratio "
        "| Forward p50, v2 → selected (µs) | Forward ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["widths"]:
        t, i = (row["timing"][m] for m in MODES)
        lines.append(
            f"| {row['width']} | {t['reference_p50_ms']:.3f} → {t['candidate_p50_ms']:.3f} "
            f"| {t['speed_ratio_reference_over_selected']:.3f}× "
            f"| {i['reference_p50_us']:.1f} → {i['candidate_p50_us']:.1f} "
            f"| {i['speed_ratio_reference_over_selected']:.3f}× |"
        )
    lines += [
        "",
        "Ratios are incumbent latency divided by selected latency. Repeated measurements "
        "of an unchanged policy can differ; a retained label means no candidate met the selection "
        "rule. All p50/p95 values and training throughput are in [timing.csv](timing.csv). "
        "No widths are pooled into a headline ratio.",
        "",
        "![Intrusive QKV stage profiles](qkv-stage-costs.png)",
        "",
        "Stage profiles use separate synchronized command buffers and are diagnostic only. "
        "They must not be added to or substituted for complete-operation latency. "
        "[Stage data](stage-profiles.csv) retain every measured stage. Hardware counters were "
        "unavailable, so occupancy, memory bandwidth and register pressure are not inferred.",
        "",
        "The reporter checks all candidate outcomes and frozen-ledger hashes, recomputes timing "
        "ratios and policy selections, and requires full-model, 53 original stress-family, "
        "16 direct-QKV, independent-FP64, guard, and byte-equivalence evidence. "
        + (
            "Fresh-process selected-Metal replay passed across all six widths: two processes, "
            "12 cases per process and ten executions per case."
            if result["repeatability"]
            else "**Fresh-process repeatability evidence has not been supplied. "
            "This interim report makes no completed development-qualification claim.**"
        ),
        "",
        "Standard PyTorch MPS and MLX are still outside the unchanged exact-repeatability gate. "
        "The official stable-source inspection in results.json is source evidence, not a new "
        "framework runtime qualification. Local upstream patches cannot substitute for standard "
        "released baselines. A new bound freeze, passing admission, fresh CPU/GPU fits and full "
        "checkpoint audits remain necessary for the complete matched study.",
        "",
        "[Machine-readable results and evidence SHA256 inventory](results.json)",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--searches", type=Path, required=True)
    parser.add_argument("--repeatability", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(args.searches, args.repeatability)
    write_report(result, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "formal_campaign_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
