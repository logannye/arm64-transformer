"""Create a compact report from the hash-bound saved-array model audit."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import prepare_model_evidence as evidence

LABELS = {"cpu_torch": "Owned PyTorch CPU", "mps": "Owned PyTorch MPS", "mlx": "Owned compiled MLX"}
COLORS = {"cpu_torch": "#526f93", "mps": "#bc6948", "mlx": "#138876"}
FAILED_SUMMARY_SHA = "0a3a5c50acb3f8265f9b14944c4248319aaf1dbe2572ddf540d8fdad8caa272b"


def reduce_rows(audit):
    result = []
    for width in evidence.WIDTHS:
        for engine in evidence.ENGINES:
            selected = [
                row
                for row in audit["rows"]
                if row["cell"]["width"] == width and row["cell"]["engine"] == engine
            ]
            evidence.archive.require(
                [row["cell"]["process_index"] for row in selected] == [0, 1],
                "Missing or reordered report cells",
            )
            gradients = [
                (key, item["max_absolute_error"])
                for row in selected
                for key, item in row["accuracy"].items()
                if key.startswith("gradient/")
            ]
            worst_name, worst_error = max(gradients, key=lambda item: item[1])
            result.append(
                {
                    "width": width,
                    "engine": engine,
                    "fresh_processes": 2,
                    "initial_replays_per_process": 10,
                    "training_repeats_per_process": 2,
                    "updates_per_training_repeat": 6,
                    "max_absolute_logit_error": max(
                        row["accuracy"]["logits"]["max_absolute_error"] for row in selected
                    ),
                    "max_absolute_loss_error": max(
                        row["accuracy"]["loss"]["max_absolute_error"] for row in selected
                    ),
                    "max_absolute_parameter_gradient_error": worst_error,
                    "worst_gradient_tensor": worst_name,
                    "initial_failed_coordinates": sum(
                        item["failed_coordinates"]
                        for row in selected
                        for item in row["accuracy"].values()
                    ),
                    "exact_fresh_process_replay": selected[0]["fingerprints"]
                    == selected[1]["fingerprints"],
                    "exact_trajectory_hashes": selected[0]["trajectory_hashes"]
                    == selected[1]["trajectory_hashes"],
                    "exact_checkpoint_prediction_hashes": selected[0]["checkpoint_predictions"]
                    == selected[1]["checkpoint_predictions"],
                    "formal_eligible": False,
                }
            )
    evidence.archive.require(
        all(
            row["initial_failed_coordinates"] == 0
            and row["exact_fresh_process_replay"]
            and row["exact_trajectory_hashes"]
            and row["exact_checkpoint_prediction_hashes"]
            for row in result
        ),
        "Report contradicts passing audit",
    )
    return result


def plot(rows, output):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "owned-model-development-v2",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    figure.patch.set_facecolor("#f7f9fc")
    for axis, column, label in zip(
        axes,
        ("max_absolute_logit_error", "max_absolute_parameter_gradient_error"),
        ("Initial logits", "All initial parameter gradients"),
        strict=True,
    ):
        axis.set_facecolor("#f7f9fc")
        for engine, marker in zip(evidence.ENGINES, ("o", "s", "^"), strict=True):
            selected = [row for row in rows if row["engine"] == engine]
            axis.plot(
                [row["width"] for row in selected],
                [row[column] for row in selected],
                color=COLORS[engine],
                marker=marker,
                label=LABELS[engine],
                linewidth=2,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(evidence.WIDTHS, [str(width) for width in evidence.WIDTHS])
        axis.set_xlabel("Model width")
        axis.set_ylabel("Maximum absolute error vs saved FP64")
        axis.set_title(label, loc="left", weight="bold", color="#172d43")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Owned frameworks pass the small-model development check",
        x=0.025,
        y=0.975,
        ha="left",
        weight="bold",
        fontsize=18,
        color="#172d43",
    )
    figure.text(
        0.025,
        0.9,
        "B3/T3 synthetic inputs · six widths · two fresh processes per framework path",
        color="#42566b",
        fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.85), ncol=3, frameon=False
    )
    figure.text(
        0.025,
        0.11,
        "All 720 saved initial tensor comparisons pass the frozen coordinate-wise tolerances.\n"
        "Six-step trajectories and checkpoint reloads pass exact replay; "
        "post-training FP64 accuracy is not measured here.",
        color="#42566b",
        fontsize=10,
        linespacing=1.6,
    )
    figure.text(
        0.025,
        0.025,
        "216 raw archives audited · 18 fresh-process pairs · no timing or formal admission · "
        "native CPU and custom Metal excluded",
        color="#42566b",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.085, right=0.985, top=0.72, bottom=0.265, wspace=0.27)
    figure.savefig(output / "initial-fp64-errors.png", dpi=180)
    figure.savefig(output / "initial-fp64-errors.svg", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    summary, _, audit = evidence.load_closed(workspace)
    rows = reduce_rows(audit)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "accuracy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = list(evidence.ANCHORS)
    for item in summary["workers"]:
        name = evidence.MODEL + "/" + item["cell"]["id"] + "/result.json"
        evidence.archive.require(
            evidence.archive.stable_hash(workspace / name)["sha256"] == item["receipt"]["sha256"],
            "Worker receipt changed before report",
        )
        paths.append(name)
    failed = workspace / "work/gpu-v4-owned/model-compatibility-v1/summary.json"
    evidence.archive.require(
        evidence.archive.stable_hash(failed)["sha256"] == FAILED_SUMMARY_SHA,
        "Retained failed model summary changed",
    )
    paths.append(failed.relative_to(workspace).as_posix())
    snapshots = workspace / "work/gpu-v4-owned/failed-model-probe-v1"
    snapshot = json.loads((snapshots / "snapshot.json").read_text())
    for item in snapshot["files"]:
        source = snapshots / item["copy"]
        evidence.archive.require(
            evidence.archive.stable_hash(source)["sha256"] == item["sha256"],
            "Failed model helper snapshot changed",
        )
    paths.extend(
        path.relative_to(workspace).as_posix()
        for path in sorted(snapshots.iterdir())
        if path.is_file()
    )
    catalog = []
    for name in paths:
        source = workspace / name
        before = evidence.archive.stable_hash(source)
        copied = output / "receipts" / Path(name).relative_to("work/gpu-v4-owned")
        if source.suffix == ".py":
            copied = copied.with_suffix(".py.txt")
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
        evidence.archive.require(
            evidence.archive.stable_hash(copied)["sha256"]
            == before["sha256"]
            == evidence.archive.stable_hash(source)["sha256"],
            "Receipt copy changed bytes",
        )
        catalog.append(
            {
                "original_workspace_path": name,
                "report_copy": copied.relative_to(output).as_posix(),
                **before,
            }
        )
    (output / "receipt-catalog.json").write_text(
        json.dumps(
            {
                "schema": "gpu-owned-model-report-catalog-v1",
                "files": catalog,
                "original_paths_inside_receipts_preserved_verbatim": True,
                "raw_npz_copied": False,
                "raw_evidence_archive_plan": "integration/gpu-owned/model-evidence-plan-v1.json",
                "scope": "Byte-preserved execution receipts; "
                "no framework qualification or formal admission",
            },
            indent=2,
        )
        + "\n"
    )
    plot(rows, output)
    generated = [path for path in output.iterdir() if path.is_file()]
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "schema": "gpu-owned-model-report-provenance-v1",
                "formal_eligible": False,
                "anchors": evidence.ANCHORS,
                "producer_sha256": evidence.archive.stable_hash(Path(__file__))["sha256"],
                "matplotlib": matplotlib.__version__,
                "rows": len(rows),
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
    print(
        json.dumps(
            {
                "copied_receipts": len(catalog),
                "receipt_bytes": sum(row["bytes"] for row in catalog),
                "rows": len(rows),
                "formal_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
