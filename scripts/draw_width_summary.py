"""Render a readable publication layout from the immutable audited sweep summary.

The original source-rendered speedup.png is retained. This changes presentation
only; it never recomputes or edits the numerical summary or audit records.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = args.report
    summary_path = report / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "completed" or summary["protocol_version"] != "v3":
        raise ValueError("A completed audited width-sweep summary is required")
    output = report / "speedup-publication.png"
    metadata = report / "figure-publication.json"
    source = report / "figure-publication.py"
    if any(path.exists() for path in (output, metadata, source)):
        raise FileExistsError("Publication figure outputs already exist")
    widths = summary["widths"]
    train = [
        summary["by_width"][str(w)]["aggregate"]["geometric_mean_training_ratio_python_over_arm64"]
        for w in widths
    ]
    forward = [
        summary["by_width"][str(w)]["forward_p50_mean_ratio_python_over_arm64"] for w in widths
    ]
    fig, axis = plt.subplots(figsize=(10, 5.8))
    positions = np.arange(len(widths))
    for values, offset, color, label in (
        (train, -0.18, "#3568ad", "Training: geometric mean of paired ratios"),
        (forward, 0.18, "#d66b34", "Forward p50: ratio of implementation means"),
    ):
        bars = axis.barh(positions + offset, values, height=0.30, color=color, label=label)
        labels = axis.bar_label(
            bars, labels=[f"{value:.3f}×" for value in values], padding=6, fontsize=10
        )
        for label in labels:
            label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 1})
    axis.axvline(1, color="#333333", linestyle="--", linewidth=1)
    axis.set_yticks(positions, [f"Width {width}" for width in widths])
    axis.invert_yaxis()
    axis.set_xlim(0, max(train + forward) * 1.20)
    axis.set_xlabel("PyTorch time / ARM64 time", labelpad=10)
    axis.set_title(
        "Where the native implementation wins, and where it loses", pad=16, weight="bold"
    )
    axis.grid(axis="x", alpha=0.18)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(axis="y", length=0)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.065), frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.025,
        "Above 1 favors assembly; below 1 favors PyTorch. "
        "48 fits · 3 seeds · 2 repeats · 1 CPU thread",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    source.write_bytes(Path(__file__).read_bytes())
    metadata.write_text(
        json.dumps(
            {
                "kind": "publication_only_figure_layout",
                "reason": "Separate bar rows avoid overlapping labels in the original line chart.",
                "measured_revision": summary["revision"],
                "summary_sha256": digest(summary_path),
                "original_figure": "speedup.png",
                "original_figure_sha256": digest(report / "speedup.png"),
                "publication_figure": output.name,
                "publication_figure_sha256": digest(output),
                "renderer": source.name,
                "renderer_sha256": digest(source),
                "numerical_summaries_audits_and_original_figures_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
