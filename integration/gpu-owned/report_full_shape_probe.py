"""Report only a closed, hash-bound full-shape development execution and array audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import audit_full_shape_probe as audited
import prepare_full_shape_evidence as evidence

LABELS = {"cpu_torch": "Owned PyTorch CPU", "mps": "Owned PyTorch MPS", "mlx": "Owned compiled MLX"}
COLORS = {"cpu_torch": "#526f93", "mps": "#bc6948", "mlx": "#138876"}


def reduce_rows(audit):
    evidence.archive.require(
        [row["cell"] for row in audit["rows"]] == evidence.cells(),
        "Missing, duplicate or reordered report cells",
    )
    result = []
    for width in evidence.WIDTHS:
        for batch, length in evidence.SHAPES:
            for engine in evidence.ENGINES:
                selected = [
                    row
                    for row in audit["rows"]
                    if (row["cell"]["width"], row["cell"]["batch"], row["cell"]["engine"])
                    == (width, batch, engine)
                ]
                evidence.archive.require(
                    [row["cell"]["process_index"] for row in selected] == [0, 1],
                    "Missing or reordered process pair",
                )
                expected, _ = audited.shapes(width, batch, length)
                for row in selected:
                    evidence.archive.require(
                        set(row["accuracy"]) == set(expected),
                        "Missing or substituted accuracy tensor",
                    )
                    for item in row["accuracy"].values():
                        error = item["max_absolute_error"]
                        evidence.archive.require(
                            item["passed"] is True
                            and type(item["failed_coordinates"]) is int
                            and item["failed_coordinates"] == 0
                            and type(item["nonfinite_coordinates"]) is int
                            and item["nonfinite_coordinates"] == 0
                            and type(error) in (int, float)
                            and math.isfinite(error)
                            and error >= 0,
                            "Report contradicts a finite passing coordinate-wise audit",
                        )
                gradients = [
                    (key, item["max_absolute_error"])
                    for row in selected
                    for key, item in row["accuracy"].items()
                    if key.startswith("gradient/")
                ]
                worst_name, worst_error = max(gradients, key=lambda item: item[1])
                pair_keys = ("fingerprints", "trajectory_hashes", "checkpoint_predictions")
                evidence.archive.require(
                    all(selected[0][key] == selected[1][key] for key in pair_keys),
                    "Report contains a fresh-process replay difference",
                )
                result.append(
                    {
                        "width": width,
                        "batch": batch,
                        "length": length,
                        "engine": engine,
                        "fresh_processes": 2,
                        "initial_replays_per_process": 10,
                        "training_repeats_per_process": 2,
                        "updates_per_training_repeat": 6,
                        "retained_checkpoint_steps": "3;6",
                        "saved_initial_tensor_comparisons": 2 * len(expected),
                        "max_absolute_logit_error": max(
                            row["accuracy"]["logits"]["max_absolute_error"] for row in selected
                        ),
                        "max_absolute_loss_error": max(
                            row["accuracy"]["loss"]["max_absolute_error"] for row in selected
                        ),
                        "max_absolute_parameter_gradient_error": worst_error,
                        "worst_gradient_tensor": worst_name,
                        "initial_failed_coordinates": 0,
                        "exact_fresh_process_replay": True,
                        "exact_trajectory_hashes": True,
                        "exact_checkpoint_prediction_hashes": True,
                        "formal_eligible": False,
                    }
                )
    return result


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "owned-full-shape-development-v1",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    figure.patch.set_facecolor("#f7f9fc")
    for row_index, (batch, length) in enumerate(evidence.SHAPES):
        for column_index, (column, label) in enumerate(
            (
                ("max_absolute_logit_error", "Initial logits"),
                ("max_absolute_parameter_gradient_error", "All initial parameter gradients"),
            )
        ):
            axis = axes[row_index, column_index]
            axis.set_facecolor("#f7f9fc")
            for engine, marker in zip(evidence.ENGINES, ("o", "s", "^"), strict=True):
                selected = [r for r in rows if r["engine"] == engine and r["batch"] == batch]
                axis.plot(
                    [r["width"] for r in selected],
                    [r[column] for r in selected],
                    color=COLORS[engine],
                    marker=marker,
                    label=LABELS[engine],
                    linewidth=2,
                )
            axis.set_xscale("log", base=2)
            values = [r[column] for r in rows if r["batch"] == batch]
            if min(values) > 0:
                axis.set_yscale("log")
            else:
                axis.set_yscale("symlog", linthresh=1e-12)
                axis.set_ylim(bottom=0)
            axis.set_xticks(evidence.WIDTHS, [str(w) for w in evidence.WIDTHS])
            axis.set_xlabel("Model width")
            axis.set_ylabel("Maximum absolute error vs saved FP64")
            axis.set_title(
                f"B{batch}/T{length} · {label}", loc="left", weight="bold", color="#172d43"
            )
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Full-context development checks for owned frameworks",
        x=0.025,
        y=0.975,
        ha="left",
        weight="bold",
        fontsize=18,
        color="#172d43",
    )
    figure.text(
        0.025,
        0.925,
        "Six widths · both study batch shapes · two fresh processes per framework path",
        color="#42566b",
        fontsize=11,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=3, frameon=False
    )
    figure.text(
        0.025,
        0.10,
        "1,440 saved initial tensor comparisons pass coordinate-wise tolerances; "
        "720 initial replay records agree exactly.\n"
        "144 six-step trajectories and step-3/6 checkpoint reloads pass exact replay; "
        "trained-state FP64 accuracy is untested.",
        color="#42566b",
        fontsize=10,
        linespacing=1.6,
    )
    figure.text(
        0.025,
        0.025,
        "504 raw archives audited · 36 fresh-process pairs · no timing or formal admission · "
        "native CPU and custom Metal excluded",
        color="#42566b",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.085, right=0.985, top=0.825, bottom=0.21, hspace=0.4, wspace=0.27)
    figure.savefig(output / "initial-fp64-errors.png", dpi=180)
    figure.savefig(output / "initial-fp64-errors.svg", metadata={"Date": None})
    plt.close(figure)
    return matplotlib.__version__


def copy_receipts(workspace, output, summary, bound_anchors):
    wanted = dict(bound_anchors)
    for item in summary["workers"]:
        name = evidence.MODEL + "/" + item["cell"]["id"] + "/result.json"
        evidence.archive.require(name not in wanted, "Duplicate report receipt")
        wanted[name] = item["receipt"]["sha256"]
    catalog = []
    for name, expected in wanted.items():
        source = evidence.archive.source_path(workspace, name)
        before = evidence.archive.stable_hash(source)
        evidence.archive.require(before["sha256"] == expected, "Report source receipt changed")
        copied = output / "receipts" / Path(name).relative_to("work/gpu-v4-owned")
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
        shutil.copymode(source, copied)
        evidence.archive.require(
            evidence.archive.stable_hash(copied)["sha256"]
            == expected
            == evidence.archive.stable_hash(source)["sha256"],
            "Report copy changed bytes",
        )
        catalog.append(
            {
                "original_workspace_path": name,
                "report_copy": copied.relative_to(output).as_posix(),
                **before,
            }
        )
    verify_report_copies(workspace, output, catalog)
    return catalog


def verify_report_copies(workspace, output, catalog):
    # Catch a late change to any receipt, including a source copied at the start.
    for row in catalog:
        for path in (workspace / row["original_workspace_path"], output / row["report_copy"]):
            evidence.archive.require(
                evidence.archive.stable_hash(path)
                == {key: row[key] for key in ("sha256", "bytes", "mode")},
                "Report receipt changed after copying",
            )


def readme(rows):
    lines = [
        "# Full-context owned-framework development evidence",
        "",
        "The existing project-maintained PyTorch CPU/MPS build and compiled MLX build pass "
        "this synthetic development check at B1/T32 and B16/T32 across widths 32–1024. "
        "These are the study’s inference and training tensor shapes, exercised for six updates.",
        "",
        "![Initial FP64 errors](initial-fp64-errors.png)",
        "",
        "The independent saved-array audit checked 504 NPZ files from 72 workers and "
        "36 fresh-process pairs. All 1,440 saved initial tensor comparisons passed the frozen "
        "coordinate-wise FP64 tolerances. Each worker recorded ten identical initial replays, "
        "restored its original state after two warmup updates, and repeated a six-step "
        "training trajectory exactly. Raw states and predictions at steps 3 and 6 "
        "bind checkpoint replay.",
        "",
        "| Shape | Engine | Max logit error | Max parameter-gradient error |",
        "|---|---|---:|---:|",
    ]
    for batch, length in evidence.SHAPES:
        for engine in evidence.ENGINES:
            selected = [r for r in rows if r["batch"] == batch and r["engine"] == engine]
            lines.append(
                f"| B{batch}/T{length} | {LABELS[engine]} | "
                f"{max(r['max_absolute_logit_error'] for r in selected):.3g} | "
                f"{max(r['max_absolute_parameter_gradient_error'] for r in selected):.3g} |"
            )
    lines += [
        "",
        "Table maxima span the six widths; this is numerical error, not a speed comparison. "
        "[The complete CSV](accuracy.csv) retains each width, engine and shape.",
        "",
        "This audit rechecks saved initial PyTorch/NumPy FP64 outputs and the retained "
        "checkpoint bytes; it does not rerun model equations or runtime authentication. "
        "Intermediate trajectory steps and other initial replays have authenticated execution-time "
        "hash records rather than separately retained raw arrays.",
        "",
        "No timing, corpus fit, trained-state FP64 audit, native CPU or custom Metal comparison "
        "is included. Broader framework qualification and the complete stress/repeatability "
        "inventory remain required. Formal fits remain 0/180 and admission remains unopened. "
        "The original failed attempt remains in the "
        "[earlier evidence report](../model-report-v2/README.md); "
        "no failure is waived or overwritten.",
        "",
        "[Receipt catalog](receipt-catalog.json), [report provenance](provenance.json), and "
        "[archive instructions](../../../integration/gpu-owned/FULL-SHAPE-PUBLICATION.md) "
        "bind the original workspace paths, copied receipts, raw evidence and producing tools.",
        "",
    ]
    return "\n".join(lines)


def build(workspace, output, bound_anchors):
    producer = evidence.archive.stable_hash(Path(__file__))["sha256"]
    helper = evidence.helper_identity()
    summary, _, audit = evidence.load_closed(workspace, bound_anchors)
    evidence.archive.reject_nested_output(
        workspace,
        output,
        {"groups": [{"roots": [evidence.MODEL, evidence.MANIFEST, evidence.AUDIT]}]},
    )
    rows = reduce_rows(audit)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    catalog = copy_receipts(workspace, output, summary, bound_anchors)
    (output / "receipt-catalog.json").write_text(
        json.dumps(
            {
                "schema": "gpu-owned-full-shape-report-catalog-v1",
                "files": catalog,
                "original_paths_inside_receipts_preserved_verbatim": True,
                "raw_npz_copied": False,
                "raw_evidence_archive_plan": (
                    "integration/gpu-owned/full-shape-evidence-plan-v1.json"
                ),
                "formal_eligible": False,
            },
            indent=2,
        )
        + "\n"
    )
    version = plot(rows, output)
    (output / "README.md").write_text(readme(rows))
    evidence.load_closed(workspace, bound_anchors)
    verify_report_copies(workspace, output, catalog)
    evidence.archive.require(
        evidence.archive.stable_hash(Path(__file__))["sha256"] == producer
        and evidence.helper_identity() == helper,
        "Report producer changed during generation",
    )
    generated = [path for path in output.iterdir() if path.is_file()]
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "schema": "gpu-owned-full-shape-report-provenance-v1",
                "formal_eligible": False,
                "formal_fits": 0,
                "admission_stream_opened": False,
                "anchors": bound_anchors,
                "producer_sha256": producer,
                "evidence_helper_sha256": helper,
                "matplotlib": version,
                "rows": len(rows),
                "workers": 72,
                "raw_archives_audited": 504,
                "saved_initial_tensor_comparisons": 1440,
                "recorded_initial_replays": 720,
                "recorded_six_step_trajectories": 144,
                "generated": [
                    {"path": path.name, **evidence.archive.stable_hash(path)}
                    for path in sorted(generated)
                ],
                "limits": audit["limits"],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "copied_receipts": len(catalog),
        "receipt_bytes": sum(row["bytes"] for row in catalog),
        "rows": len(rows),
        "formal_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.workspace.resolve(),
                args.output.resolve(),
                evidence.anchors(args.summary_sha256, args.audit_sha256),
            )
        )
    )


if __name__ == "__main__":
    main()
