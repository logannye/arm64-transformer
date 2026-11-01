"""Bounded, evidence-linked performance summaries after complete postrun auditing."""

import argparse
import importlib.util
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("library_result_audit", HERE / "audit_campaign.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
C, F, AD = AUDIT.C, AUDIT.F, AUDIT.AD


def summarize(reports):
    groups, ratios = [], []
    engines = AUDIT.R.ENGINES
    for width in C.WIDTHS:
        for engine in engines:
            selected = [
                r for r in reports if r["cell"]["width"] == width and r["cell"]["engine"] == engine
            ]
            C.require(
                len(selected) == 6
                and {(r["cell"]["seed"], r["cell"]["repeat"]) for r in selected}
                == {(s, t) for s in (42, 43, 44) for t in (1, 2)},
                "Summary cell denominator differs",
            )
            row = dict(
                width=width,
                engine=engine,
                fits=6,
                median_training_seconds=statistics.median(r["training_seconds"] for r in selected),
                median_training_targets_per_second=statistics.median(
                    524288 / r["training_seconds"] for r in selected
                ),
                median_latency_p50_seconds=statistics.median(r["latency_p50"] for r in selected),
                median_latency_p95_seconds=statistics.median(r["latency_p95"] for r in selected),
                median_test_nll=statistics.median(r["quality"]["nll"] for r in selected),
                median_test_accuracy=statistics.median(r["quality"]["accuracy"] for r in selected),
            )
            groups.append(row)
        for engine in engines:
            pairs = []
            for seed in (42, 43, 44):

                def arm(name):
                    return [
                        r
                        for r in reports
                        if r["cell"]["width"] == width
                        and r["cell"]["seed"] == seed
                        and r["cell"]["engine"] == name
                    ]

                baseline, candidate = arm("cpu_torch"), arm(engine)
                pairs.append(
                    dict(
                        seed=seed,
                        training_speedup=statistics.median(r["training_seconds"] for r in baseline)
                        / statistics.median(r["training_seconds"] for r in candidate),
                        resident_b1_speedup=statistics.median(r["latency_p50"] for r in baseline)
                        / statistics.median(r["latency_p50"] for r in candidate),
                    )
                )
            ratios.append(
                dict(
                    width=width,
                    engine=engine,
                    baseline="cpu_torch",
                    paired_seeds=pairs,
                    median_paired_training_speedup=statistics.median(
                        r["training_speedup"] for r in pairs
                    ),
                    median_paired_resident_b1_speedup=statistics.median(
                        r["resident_b1_speedup"] for r in pairs
                    ),
                )
            )
    return dict(groups=groups, cpu_torch_relative=ratios)


def execute(args):
    C.require_execution_support()
    manifest_binding = C.binding(args.manifest)
    C.require(manifest_binding["sha256"] == args.manifest_sha256, "Manifest trust anchor differs")
    manifest = F.read(args.manifest, args.manifest_sha256)
    fits = len(manifest["comparison"]["cells"])
    test_predictions = manifest["comparison"]["planned_test_token_predictions"]
    binding = C.binding(args.audit)
    C.require(binding["sha256"] == args.audit_sha256, "Campaign audit trust anchor differs")
    audit = F.PLAN.load(binding)
    AD.verify(manifest, manifest_binding, audit["admission"])
    campaign = AUDIT.ledger(audit["campaign"], manifest, manifest_binding, audit["admission"])
    expected = [
        (r["cell"], checkpoint)
        for r in campaign["cells"]
        for checkpoint in ("best.f32", "last.f32")
    ]
    C.require(
        audit["schema"] == "frozen-library-campaign-audit-v1"
        and audit["manifest"] == manifest_binding
        and audit["complete"] is True
        and audit["audited_checkpoints"] == 2 * fits
        and audit["completed_fits"] == fits
        and audit["audited_test_token_predictions"] == test_predictions
        and [(r["cell"], r["checkpoint"]) for r in audit["reports"]] == expected
        and len(audit["repeat_checks"]) == fits // 2,
        "Incomplete campaign audit",
    )
    checked = []
    for row in audit["reports"]:
        value = F.PLAN.load(row["report"])
        C.require(
            value["schema"] == "frozen-library-checkpoint-audit-v1"
            and value["cell"] == row["cell"]
            and value["checkpoint"] == row["checkpoint"]
            and value["manifest"] == manifest_binding
            and value["evidence_valid"] is True
            and value["passed"] is row["passed"],
            "Postrun report identity differs",
        )
        checked.append(value)
    accepted = all(r["passed"] for r in checked) and all(
        r["passed"] for r in audit["repeat_checks"]
    )
    C.require(
        audit["scientific_acceptance"] is accepted,
        "Scientific acceptance differs from complete evidence",
    )
    result = dict(
        schema="frozen-library-comparison-report-v1",
        manifest=manifest_binding,
        audit=binding,
        protocol=C.PROTOCOL,
        scientific_acceptance=accepted,
        formal_fits=fits,
        checkpoints=2 * fits,
        test_token_predictions=test_predictions,
        scope_amendment=manifest["comparison"]["scope_amendment"],
        hardware=manifest["bindings"]["hardware"],
        scope=(
            "Project-maintained five-engine FP32-storage transformer; original ASCII, "
            "T32, B16 workload; resident B1 forward latency; phase 1 widths 32–256 only. "
            "Widths 512/1024 and 60 fits are deferred, not accepted. Original study incomplete."
        ),
        framework_selection=manifest["comparison"]["framework_selection"],
        model_selection=manifest["comparison"]["model_selection"],
        timing=manifest["comparison"]["timing"],
        workload=manifest["comparison"]["workload"],
    )
    if accepted:
        result.update(summarize([r for r in checked if r["checkpoint"] == "best.f32"]))
    output = Path(args.output).absolute()
    C.require(
        not output.exists() and not output.is_symlink() and output.resolve() == output,
        "Fresh report output required",
    )
    capture = F.PLAN.load(manifest["bindings"]["capture"])
    inventory = F.PLAN.load(capture["runtime_inventory"])
    request = F.PLAN.load(F.PLAN.load(manifest["bindings"]["raw_prerequisites"])["request"])
    AUDIT.R.W.protected_output(
        dict(output=str(output), raw_store=request["raw_store"]),
        type(
            "MetadataRuntime", (), dict(roles=manifest["bindings"]["roles"], inventory=inventory)
        )(),
    )
    output.mkdir()
    C.write(output / "report.json", result)
    lines = [
        "# Maintained-library five-engine transformer comparison",
        "",
        result["scope"],
        "",
        f"First-phase scientific acceptance: **{accepted}**. Audited: {fits} fits, "
        f"{2 * fits} checkpoints, {fits // 2} repeat pairs "
        f"and {test_predictions:,} test predictions.",
        "",
        "All model state uses FP32 storage. The selected CPU and GPU precision operators "
        "use the internal arithmetic declared in the frozen model selection. Framework builds "
        "are project-maintained. Timing applies to this hardware and workload.",
        "",
    ]
    if accepted:
        lines.extend(
            [
                "Each row is the median of six fits across three paired seeds and two repeats. "
                "Training throughput includes the frozen training boundary; latency measures "
                "resident B1 forward and completion, with 20 warmups and 100 samples.",
                "",
                "| Width | Engine | Training targets/s | B1 p50 ms | B1 p95 ms | Test NLL |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in result["groups"]:
            lines.append(
                f"| {row['width']} | {row['engine']} | "
                f"{row['median_training_targets_per_second']:.2f} | "
                f"{1000 * row['median_latency_p50_seconds']:.3f} | "
                f"{1000 * row['median_latency_p95_seconds']:.3f} | "
                f"{row['median_test_nll']:.6f} |"
            )
    else:
        lines.append(
            "One or more postrun gates failed. The complete negative evidence is retained "
            "in the linked audit; this report does not promote a qualified performance comparison."
        )
    with (output / "report.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    F.verify(manifest)
    C.bound(binding)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "manifest-sha256", "audit", "audit-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    execute(parser.parse_args())
