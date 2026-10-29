"""Collect compact numerical receipts from preserved GPU development evidence.

This reads saved JSON only. It never executes a GPU model or changes a result.
The original paths/hashes and contemporaneous implementation identities remain
attached to every compact record. Unexecuted inventory cells stay explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
WIDTHS = (32, 64, 128, 256, 512, 1024)
PHASES = ("reference-v0", "p1", "p1-complete", "p2", "p3")
FAMILIES = ("attention", "ln-qkv", "smoke")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def controls(case, family):
    names = (
        (
            "exact_repeat",
            "causal_future_intervention",
            "future_gradient_exact_zero",
            "prefix_gradient_exact",
            "mask_exact_zero",
            "input_unchanged",
            "finite",
            "guards",
        )
        if family == "attention"
        else ("inputs_unchanged", "repeat_exact", "guards_passed", "outputs_finite")
    )
    values = {name: {"passed": case[name], "executed": True} for name in names}
    if family == "attention" and case["tokens"] == 1:
        for name in (
            "causal_future_intervention",
            "future_gradient_exact_zero",
            "prefix_gradient_exact",
        ):
            values[name] = {"passed": None, "executed": False, "reason": "Requires T > 1"}
    return values


def compact(path, phase, family, source_root):
    full = read(path)
    capability = full.get("capabilities", {})
    identity = {
        key: capability[key]
        for key in (
            "binary_sha256",
            "shader_sha256",
            "host_source_sha256",
            "compiled_host_source_sha256",
            "host_compiler",
            "compiler_math_mode",
            "floating_point_functions",
            "device",
            "shader_language",
        )
        if key in capability
    }
    record = {
        "phase": phase,
        "family": family,
        "width": full["width"],
        "source_path": str(path.relative_to(source_root)),
        "source_sha256": digest(path),
        "passed": full["passed"],
        "implementation_identity": identity,
        "driver_sha256": full.get("script_sha256", full.get("driver_sha256")),
        "source_sha256_before": full.get("source_sha256_before"),
        "source_sha256_after": full.get("source_sha256_after"),
        "source_unchanged": full.get("source_unchanged"),
    }
    if family == "smoke":
        record["checks"] = full["checks"]
        record["controls"] = {
            name: full[name]
            for name in ("exact_state_repeat", "exact_warmup_restore", "finite", "guards")
        }
        record["lifecycle_controls"] = full.get("integrity", {})
    else:
        record["cases"] = []
        for case in full["cases"]:
            item = {
                "name": case["name"],
                "passed": case["passed"],
                "checks": case["checks"],
                "controls": controls(case, family),
            }
            for name in ("tokens", "rows", "kind", "fixture", "fixture_sha256", "repeat_checks"):
                if name in case:
                    item[name] = case[name]
            record["cases"].append(item)
    return record


def checks(record):
    return (
        record["checks"]
        if "checks" in record
        else [check for case in record["cases"] for check in case["checks"]]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "work/gpu-v1/metal-development")
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--post-p3", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = ("receipts.json", "summary.json", "baseline-context.json", "provenance.json")
    if args.integration:
        outputs += ("integration.json",)
    if args.post_p3:
        outputs += ("post-p3.json",)
    if not args.overwrite and any((DEST / name).exists() for name in outputs):
        raise FileExistsError("Compact evidence exists; use --overwrite only for regeneration")

    records = []
    inventory = []
    for phase in PHASES:
        for family in FAMILIES:
            for width in WIDTHS:
                path = args.source_root / phase / f"{family}-width-{width}.json"
                cell = {"phase": phase, "family": family, "width": width, "status": "not_run"}
                if path.exists():
                    record = compact(path, phase, family, args.source_root)
                    records.append(record)
                    numerical = checks(record)
                    cases = record.get("cases", [])
                    cell.update(
                        status="passed" if record["passed"] else "failed",
                        passed_cases=sum(case["passed"] for case in cases) if cases else None,
                        total_cases=len(cases) if cases else None,
                        numeric_comparisons=len(numerical),
                        failed_numeric_comparisons=sum(not value["passed"] for value in numerical),
                        failed_coordinates=sum(
                            value["failed_coordinates"] or 0 for value in numerical
                        ),
                    )
                inventory.append(cell)

    final = [record for record in records if record["phase"] == "p3"]
    assert len(final) == 18 and all(record["passed"] for record in final)
    counts = {
        family: sum(len(checks(r)) for r in final if r["family"] == family) for family in FAMILIES
    }
    assert counts == {"attention": 690, "ln-qkv": 2244, "smoke": 252}
    summary = {
        "status": "development_pass_formal_gate_closed",
        "widths": list(WIDTHS),
        "scope": "Synthetic development operands and initial-state integration checks only",
        "precision_fixture_seed": 20261103,
        "smoke_operand_seed": 20261105,
        "initial_model_seed": 42,
        "reserved_admission_seed_opened": False,
        "selected_candidate": "P3",
        "inventory": inventory,
        "final_numeric_comparisons": counts,
        "final_numeric_comparisons_total": sum(counts.values()),
        "numeric_denominator_note": (
            "Array comparisons, not coordinates, controls, or independent observations"
        ),
        "final_case_counts": {"attention": 138, "ln-qkv": 132, "smoke_reports": 6},
        "controls_scope": {
            "attention": (
                "Repeat, mask, input immutability, finite/shape and guards; "
                "future/prefix interventions only for T > 1"
            ),
            "ln-qkv": "Exact primitive repeats, input immutability, finite outputs and guards",
            "smoke": (
                "Exact state repeat/warmup restore, finite states, guards, resident gather "
                "and lifecycle checks; nonfinite fault injection only at width32"
            ),
        },
        "final_initial_model_errors": {
            str(record["width"]): {
                check["name"]: check["max_absolute_error"]
                for check in record["checks"]
                if check["name"] in ("full/logits", "full/loss", "full/grad", "adam/parameters")
            }
            for record in final
            if record["family"] == "smoke"
        },
    }
    receipts = {
        "scope": (
            "Compact lossless numerical/check fields from original development JSON; "
            "pipeline/memory/setup-time fields omitted"
        ),
        "source_root": "work/gpu-v1/metal-development",
        "reports": records,
    }
    baseline = {
        "scope": (
            "Separate training-only width32 repeat isolation; "
            "not the custom precision test denominator"
        ),
        "provenance_limit": (
            "Early baseline diagnostics did not retain contemporaneous executed-source hashes; "
            "hashes authenticate retained JSON bytes only"
        ),
        "receipts": {},
    }
    contextual_sources = {}
    for name in ("mps-repeat-diagnosis-32.json", "mlx-repeat-diagnosis-32.json"):
        path = ROOT / "work/gpu-v1" / name
        baseline["receipts"][name] = read(path)
        contextual_sources[str(path.relative_to(ROOT))] = digest(path)
    if args.integration:
        path = ROOT / "work/gpu-v1/integration-v2-repeat/repeatability.json"
        baseline["six_update_integration_repeats"] = read(path)
        contextual_sources[str(path.relative_to(ROOT))] = digest(path)
        summary["tiny_training_repeat_scope"] = (
            "Widths32/1024, three engines, six updates, two fresh processes; training-only rows"
        )
        summary["tiny_training_repeat_exact"] = [
            {key: record[key] for key in ("width", "engine", "exact")}
            for record in baseline["six_update_integration_repeats"]
        ]
        integration = {
            "scope": (
                "Training-only development: widths32/1024, three engines, two fresh processes, "
                "eight training rows, three development validation rows and three development "
                "test rows; all drawn from the original training partition. B3, two epochs, "
                "six updates per run. No corpus validation/test rows or formal timings."
            ),
            "repeatability": baseline.pop("six_update_integration_repeats"),
            "fp64_audits": [],
            "runs": [],
        }
        for repeat in ("integration-v2", "integration-v2-repeat"):
            for width in (32, 1024):
                path = ROOT / "work/gpu-v1" / repeat / f"fp64-width-{width}.json"
                audit = read(path)
                integration["fp64_audits"].append({"repeat": repeat, **audit})
                contextual_sources[str(path.relative_to(ROOT))] = digest(path)
                for engine in ("mps", "mlx", "metal"):
                    path = ROOT / "work/gpu-v1" / repeat / f"{engine}-{width}/runtime.json"
                    runtime = read(path)
                    fields = (
                        "status",
                        "scope",
                        "engine",
                        "contract",
                        "model",
                        "config",
                        "environment",
                        "host_thread_environment_set_before_numpy_import",
                        "python",
                        "platform",
                        "input_sha256",
                        "source_sha256",
                        "git_revision",
                        "source_dirty",
                        "selected_validation_replay_exact",
                        "inference_replay_exact",
                        "best_epoch",
                        "steps",
                        "inputs_unchanged",
                        "sources_unchanged",
                        "artifact_sha256",
                    )
                    integration["runs"].append(
                        {"repeat": repeat, "width": width, **{key: runtime[key] for key in fields}}
                    )
                    contextual_sources[str(path.relative_to(ROOT))] = digest(path)
        count = sum(len(audit["checks"]) for audit in integration["fp64_audits"])
        assert count == 36 and all(audit["passed"] for audit in integration["fp64_audits"])
        summary["tiny_training_integration_fp64_comparisons"] = count
        (DEST / "integration.json").write_text(
            json.dumps(integration, separators=(",", ":"), allow_nan=False) + "\n"
        )
    if args.post_p3:
        shape_path = ROOT / "work/gpu-v1/training-shape/summary.json"
        shape = read(shape_path)
        assert shape["passed"] and shape["test_count"] == 36
        contextual_sources[str(shape_path.relative_to(ROOT))] = digest(shape_path)
        for run in shape["runs"]:
            assert digest(ROOT / run["log"]) == run["sha256"]
        repair = []
        for width in WIDTHS:
            path = args.source_root / "lifecycle-repaired" / f"smoke-width-{width}.json"
            record = compact(path, "lifecycle-repaired", "smoke", args.source_root)
            assert record["passed"]
            repair.append(record)
            contextual_sources[str(path.relative_to(ROOT))] = digest(path)
        assert {r["implementation_identity"]["shader_sha256"] for r in repair} == {
            r["implementation_identity"]["shader_sha256"] for r in final
        }
        probe_path = ROOT / "work/gpu-v1/metal-lifecycle-fix-fresh-checkout/cleanup-probe.json"
        cleanup = read(probe_path)
        assert cleanup["passed"]
        contextual_sources[str(probe_path.relative_to(ROOT))] = digest(probe_path)
        # Keep numerical/lifecycle evidence without machine-specific build paths.
        for control in cleanup["controls"].values():
            control.pop("compile_command", None)
        post = {
            "scope": (
                "Additional development controls after P3; separate denominators and source "
                "snapshots. No shader arithmetic change, formal admission, or timing result."
            ),
            "training_shape": shape,
            "lifecycle_repaired_smoke": repair,
            "native_cleanup_mocked_context_control": cleanup,
        }
        trace_path = ROOT / "work/gpu-v1/metal-trace-components/receipt.json"
        trace = read(trace_path)
        assert trace["passed"] and not trace["gpu_executed"]
        post["trace_components_cpu_only"] = trace
        contextual_sources[str(trace_path.relative_to(ROOT))] = digest(trace_path)
        hardware_path = ROOT / "work/gpu-v1/metal-trace-components/hardware/receipt.json"
        hardware = read(hardware_path)
        assert hardware["passed"] and {run["width"] for run in hardware["runs"]} == {32, 1024}
        assert len(hardware["runs"]) == 2
        for run in hardware["runs"]:
            assert run["trace_smoke_passed"]
            assert run["development_fit_status"] == "completed"
            assert run["all_saved_numerical_artifacts_exact_against_integration_v2"]
            assert run["source_sha256"]["native/metal/kernels.metal"] in {
                record["implementation_identity"]["shader_sha256"] for record in final
            }
        post["trace_components_hardware"] = hardware
        contextual_sources[str(hardware_path.relative_to(ROOT))] = digest(hardware_path)
        summary["post_p3_counts"] = {
            "b16_t32_test_functions": 36,
            "repaired_smoke_reports": len(repair),
            "repaired_smoke_numeric_comparisons": sum(len(record["checks"]) for record in repair),
            "original_p3_count_unchanged": 3186,
            "final_trace_hardware_widths": [32, 1024],
            "final_tiny_fit_artifacts_exact_against_integration_v2": True,
        }
        (DEST / "post-p3.json").write_text(
            json.dumps(post, separators=(",", ":"), allow_nan=False) + "\n"
        )
    for name, data in (
        ("receipts.json", receipts),
        ("summary.json", summary),
        ("baseline-context.json", baseline),
    ):
        (DEST / name).write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    provenance = {
        "scope": (
            "Publication extraction only; no model execution, timing, "
            "tolerance changes or new assessment"
        ),
        "collector_sha256": digest(Path(__file__)),
        "source_receipts": {record["source_path"]: record["source_sha256"] for record in records},
        "contextual_source_receipts": contextual_sources,
        "derived_payloads": {
            name: digest(DEST / name) for name in outputs if name != "provenance.json"
        },
        "adapter_formatting_receipt": read(args.source_root / "p3/adapter-formatting-receipt.json"),
        "source_snapshot_manifests": {
            str(path.relative_to(args.source_root)): read(path)
            for path in sorted(args.source_root.rglob("manifest.json"))
        },
        "original_source_and_binary_payloads": (
            "Retained in work/gpu-v1/metal-development; "
            "binary/state payloads are intentionally absent from Git"
        ),
    }
    (DEST / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                "reports": len(records),
                "final_comparisons": sum(counts.values()),
                "bytes": sum((DEST / name).stat().st_size for name in outputs),
            }
        )
    )


if __name__ == "__main__":
    main()
