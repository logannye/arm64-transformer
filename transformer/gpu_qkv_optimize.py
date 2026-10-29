"""Prespecified, precision-preserving QKV development search; no formal data."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from contextlib import contextmanager
from pathlib import Path

from transformer import gpu_optimize as op
from transformer.gpu_policy import identity, validate, with_qkv

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "qkv_training_partition_development_only"
INCUMBENT = ROOT / "experiments/gpu-v2/policies/selected.json"
INVENTORY = ROOT / "experiments/gpu-v3-qkv/candidates.json"


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


def apply(policy, candidate):
    if candidate not in inventory():
        raise ValueError("Candidate is outside the frozen inventory")
    value = with_qkv(policy)
    value["qkv"][candidate["family"]] = candidate["value"]
    return validate(value)


def family_order(profiles, mode):
    stages = profiles[mode]["median_gpu_seconds"]
    names = ("forward", "dx") if mode == "training" else ("forward",)
    costs = {
        name: sum(value for key, value in stages.items() if key.startswith("qkv_" + name))
        for name in names
    }
    if any(value <= 0 for value in costs.values()):
        raise ValueError("Missing protected QKV stage profile")
    return sorted(names, key=lambda name: (-costs[name], name))


@contextmanager
def backend(initial, policy):
    from transformer.gpu_metal import Backend

    value = Backend(initial, kernel_policy=policy)
    try:
        yield value
    finally:
        value.close()


def qualification(initial, policy, reference, cache, direct_cases, *, complete=False):
    from scripts.check_metal_qkv_optimization import qualify_qkv

    with backend(initial, policy) as candidate, backend(initial, reference) as retained:
        standard = op.qualify(candidate, cache, "all" if complete else "norm")
        direct = qualify_qkv(candidate, retained, cache.width, cached_cases=direct_cases)
    return {
        "qualification": standard,
        "direct_qkv": direct,
        "passed": standard["passed"] and direct["passed"],
    }


def checked_sources(expected):
    if op.source_identity() != expected:
        raise RuntimeError("Source or native binary changed during the experiment")


def load_incumbent(width, sources):
    """Read precisely the retained policy bytes included in the committed freeze."""
    raw = INCUMBENT.read_bytes()
    relative = str(INCUMBENT.relative_to(ROOT))
    if sources.get(relative) != hashlib.sha256(raw).hexdigest():
        raise RuntimeError("Retained v2 policy is missing from or differs from the source freeze")
    return json.loads(raw)["widths"][str(width)]


@contextmanager
def candidate_evidence(group, row, path, output_root):
    """Retain in-flight failures even if execution or evidence persistence fails."""
    group["candidates"].append(row)
    primary = None
    try:
        yield
    except BaseException as error:
        primary = error
        op.record_failure(row, error, "execution_rejected_unranked")
        raise
    finally:
        try:
            op.write_fresh(path, row)
            row["evidence"] = {
                "path": str(path.relative_to(output_root)),
                "sha256": op.digest(path),
            }
        except Exception as error:
            row.setdefault("evidence_errors", []).append(f"{type(error).__name__}: {error}")
            if primary is None:
                op.record_failure(row, error, "evidence_rejected_unranked")
                raise
            primary.add_note(f"Also failed to persist {path}: {type(error).__name__}: {error}")


def v2_equivalence(initial, policy, original, fixture):
    """Retained implementation is an exact control, separate from FP64 accuracy."""
    checks = {}
    with backend(initial, policy) as candidate, backend(initial, original) as retained:
        cb, rb = op.batches(candidate, fixture), op.batches(retained, fixture)
        for mode in ("training", "inference"):
            left = candidate.to_numpy(candidate.forward(cb[mode]))
            right = retained.to_numpy(retained.forward(rb[mode]))
            checks[mode + "/forward"] = left.tobytes() == right.tobytes()
        left = op.capture(
            candidate,
            dict(zip(("loss", "gradient"), candidate.loss_and_grad(cb["training"]), strict=True)),
        )
        right = op.capture(
            retained,
            dict(zip(("loss", "gradient"), retained.loss_and_grad(rb["training"]), strict=True)),
        )
        checks["loss_and_gradient"] = op.same_arrays(left, right)
        candidate.to_numpy(candidate.train_step(cb["training"]))
        retained.to_numpy(retained.train_step(rb["training"]))
        checks["adam_state"] = op.state_equal(candidate.export_state(), retained.export_state())
    return {"checks": checks, "passed": all(checks.values())}


def run(width, output):
    from scripts.check_metal_qkv_optimization import qkv_reference, qkv_stress_cases

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(output.suffix + ".ledger.json").exists():
        raise FileExistsError(output)
    evidence = output.with_suffix(".evidence")
    evidence.mkdir(exist_ok=False)
    candidates = inventory()
    if json.loads(INVENTORY.read_text()) != candidates:
        raise ValueError("Committed candidate ledger differs from the implementation")
    fixture = op.load_development(width)
    source = op.source_identity()
    revision = op.committed_revision(source)
    runtime = op.runtime_identity()
    retained_v2 = load_incumbent(width, source)
    original = {mode: with_qkv(policy) for mode, policy in retained_v2.items()}
    initial = fixture["initial"]
    result = {
        "schema": 1,
        "scope": SCOPE,
        "width": width,
        "status": "started",
        "source_revision": revision,
        "source_sha256": source,
        "runtime_identity": runtime,
        "input_sha256": fixture["input_sha256"],
        "incumbent_policy": original,
        "incumbent_policy_file_sha256": source[str(INCUMBENT.relative_to(ROOT))],
        "retained_v2_policy": retained_v2,
        "candidate_inventory": candidates,
        "min_improvement": op.MIN_IMPROVEMENT,
        "warmups": op.WARMUPS,
        "observations": op.OBSERVATIONS,
        "blocks": op.BLOCKS,
        "environment": op.environment_identity(),
        "candidates": [],
        "formal_fits": 0,
        "admission_stream_opened": False,
    }
    try:
        op.check_conditions(result, "conditions_before")
        cache = op.OracleCache(fixture, width)
        direct_cases = [(case, qkv_reference(case)) for case in qkv_stress_cases(width)]
        result["incumbent_v2_byte_equivalence"] = v2_equivalence(
            initial, original, retained_v2, fixture
        )
        if not result["incumbent_v2_byte_equivalence"]["passed"]:
            raise RuntimeError("Null-QKV schema upgrade changed retained v2 arithmetic")
        # These intrusive profiles inform family order, never performance ranking.
        with backend(initial, original) as retained:
            prepared = op.batches(retained, fixture)
            result["incumbent_profiles"] = {
                mode: op.stage_profile(retained, prepared[mode], mode)
                for mode in ("training", "inference")
            }
        result["family_order"] = {
            mode: family_order(result["incumbent_profiles"], mode)
            for mode in ("training", "inference")
        }
        ledger_fields = (
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
        ledger = {key: result[key] for key in ledger_fields}
        ledger.update(frozen_before_performance_scoring=True, profiles=result["incumbent_profiles"])
        ledger_path = output.with_suffix(output.suffix + ".ledger.json")
        op.write_fresh(ledger_path, ledger)
        result["ledger_sha256"] = op.digest(ledger_path)

        # Freeze compile/resource eligibility for every applicable isolated candidate
        # at this width before the first scored observation in either mode.
        support = {}
        for mode in ("training", "inference"):
            active = [c for c in candidates if mode == "training" or c["family"] == "forward"]
            preflight = op.preflight_policies(
                fixture,
                [{"id": c["id"], "policy": apply(original[mode], c)} for c in active],
                evidence / f"{mode}-isolated-preflight.json",
            )
            support[mode] = {r["id"]: r["status"] == "supported" for r in preflight["records"]}
        selected = {}
        result["incumbent_qualification"] = {}
        for mode in ("training", "inference"):
            current = original[mode]
            result["incumbent_qualification"][mode] = qualification(
                initial, current, current, cache, direct_cases, complete=True
            )
            if not result["incumbent_qualification"][mode]["passed"]:
                raise RuntimeError(f"Retained {mode} incumbent failed unchanged qualification")
            for family in result["family_order"][mode]:
                family_rows = [c for c in candidates if c["family"] == family]
                group = {
                    "mode": mode,
                    "family": family,
                    "before_policy": current,
                    "candidates": [],
                    "chosen_id": None,
                }
                result["candidates"].append(group)
                composed = op.preflight_policies(
                    fixture,
                    [{"id": c["id"], "policy": apply(current, c)} for c in family_rows],
                    evidence / f"{mode}-{family}-composed-preflight.json",
                )
                composed_support = {
                    r["id"]: r["status"] == "supported" for r in composed["records"]
                }
                for index, candidate in enumerate(family_rows):
                    checked_sources(source)
                    policy = apply(current, candidate)
                    row = {
                        **candidate,
                        "mode": mode,
                        "policy": policy,
                        "policy_sha256": identity(policy),
                        "status": "started",
                    }
                    path = evidence / f"{mode}-{candidate['id']}.json"
                    with candidate_evidence(group, row, path, output.parent):
                        if (
                            not support[mode][candidate["id"]]
                            or not composed_support[candidate["id"]]
                        ):
                            row["status"] = "compile_or_resource_rejected_unranked"
                        else:
                            row["qualification"] = qualification(
                                initial, policy, original[mode], cache, direct_cases
                            )
                            if not row["qualification"]["passed"]:
                                row["status"] = "numerically_rejected_unranked"
                            else:
                                op.check_conditions(row, "conditions_before")
                                row["fresh_pair"] = {}
                                with op.fresh_timing_pair(
                                    fixture, current, policy, row["fresh_pair"]
                                ) as pair:
                                    left, right, left_batches, right_batches = pair
                                    row["paired_timing"] = op.measure_paired(
                                        left,
                                        right,
                                        left_batches[mode],
                                        right_batches[mode],
                                        mode,
                                        order_offset=index % 2,
                                    )
                                op.check_conditions(row, "conditions_after")
                                row["status"] = "qualified_measured"
                        checked_sources(source)
                    print(
                        json.dumps(
                            {
                                "width": width,
                                "mode": mode,
                                "candidate": candidate["id"],
                                "status": row["status"],
                                "ratio": row.get("paired_timing", {}).get(
                                    "median_paired_latency_ratio"
                                ),
                            }
                        ),
                        flush=True,
                    )
                winner = op.select_winner(group["candidates"])
                if winner is not None:
                    current = winner["policy"]
                    group["chosen_id"] = winner["id"]
                group["after_policy"] = current
            selected[mode] = current
        result["selected_policy"] = selected
        result["selected_v2_byte_equivalence"] = v2_equivalence(
            initial, selected, retained_v2, fixture
        )
        if not result["selected_v2_byte_equivalence"]["passed"]:
            raise RuntimeError("Selected QKV policy changed full-model reference bytes")
        result["final_qualification"] = {
            mode: qualification(initial, policy, original[mode], cache, direct_cases, complete=True)
            for mode, policy in selected.items()
        }
        if not all(row["passed"] for row in result["final_qualification"].values()):
            raise RuntimeError("Final selected policy failed complete qualification")
        with backend(initial, selected) as final:
            result["selected_bundle_full_model"] = op.full_checks(final, cache)
        if not result["selected_bundle_full_model"]["passed"]:
            raise RuntimeError("Combined training/inference bundle failed full-model checks")
        result["final_pair_memory"] = {}
        with op.fresh_timing_pair(fixture, original, selected, result["final_pair_memory"]) as pair:
            left, right, lb, rb = pair
            result["selected_vs_incumbent"] = {
                mode: op.measure_paired(left, right, lb[mode], rb[mode], mode)
                for mode in ("training", "inference")
            }
        with backend(initial, selected) as final:
            prepared = op.batches(final, fixture)
            result["selected_profiles"] = {
                mode: op.stage_profile(final, prepared[mode], mode)
                for mode in ("training", "inference")
            }
        checked_sources(source)
        result["source_unchanged"] = True
        result["runtime_unchanged"] = op.runtime_identity() == runtime
        result["input_unchanged"] = (
            op.load_development(width)["input_sha256"] == fixture["input_sha256"]
        )
        if not result["runtime_unchanged"] or not result["input_unchanged"]:
            raise RuntimeError("Runtime or fixture changed")
        op.check_conditions(result, "conditions_after")
        result["status"] = "completed"
    except BaseException as error:
        op.record_failure(result, error, "failed")
        raise
    finally:
        op.write_fresh(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(32, 64, 128, 256, 512, 1024), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ARM64_TRANSFORMER_WIDTH", str(args.width)) != str(args.width):
        raise ValueError("Width conflicts with environment")
    os.environ["ARM64_TRANSFORMER_WIDTH"] = str(args.width)
    print(run(args.width, args.output)["status"], flush=True)


if __name__ == "__main__":
    main()
