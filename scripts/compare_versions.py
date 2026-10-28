"""Plot two audited campaigns without treating their timings as a paired study.

Example:
    python scripts/compare_versions.py --v1 experiments/matched-v1/summary.json \
        --v2 experiments/matched-v2/summary.json --output experiments/v1-v2.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

FIELDS = ("training_seconds", "forward_p50_ms")
IMPLEMENTATIONS = ("python", "arm64")
COLORS = {"python": "#3568ad", "arm64": "#d66b34"}
LABELS = {"python": "PyTorch", "arm64": "ARM64 assembly"}


def load_summary(path):
    raw = path.read_bytes()
    summary = json.loads(raw)
    if summary.get("status") != "completed":
        raise ValueError(f"{path}: requires a completed, audited result summary")
    measurements = summary["aggregate"]["by_implementation"]
    for impl in IMPLEMENTATIONS:
        for field in FIELDS:
            stats = measurements[impl][field]
            if stats["n_seeds"] != 3:
                raise ValueError(f"{path}: expected three independent seed means")
            if not all(
                math.isfinite(stats[key]) and stats[key] >= 0 for key in ("mean", "sample_sd")
            ):
                raise ValueError(f"{path}: nonfinite or negative timing statistic")
    return summary, hashlib.sha256(raw).hexdigest()


def create_plot(v1, v2, output):
    output = Path(output)
    metadata_path = output.with_suffix(".json")
    if output.suffix.lower() != ".png":
        raise ValueError("Choose a .png output path")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Output or companion metadata exists; choose a fresh path")
    first, first_hash = load_summary(Path(v1))
    second, second_hash = load_summary(Path(v2))
    summaries = (first, second)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180}
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for axis, field, title, ylabel, digits in zip(
        axes,
        FIELDS,
        ("Training + validation", "Full 32-character forward pass, p50"),
        ("Seconds per complete fit", "Milliseconds per context"),
        (2, 4),
        strict=True,
    ):
        plotted = []
        for campaign, summary in enumerate(summaries):
            for impl_index, impl in enumerate(IMPLEMENTATIONS):
                stats = summary["aggregate"]["by_implementation"][impl][field]
                location = campaign * 2.1 + impl_index * 0.75
                axis.bar(
                    location,
                    stats["mean"],
                    yerr=stats["sample_sd"],
                    width=0.6,
                    color=COLORS[impl],
                    edgecolor="white",
                    capsize=4,
                    hatch="//" if campaign == 0 else None,
                    alpha=0.9,
                )
                plotted.append((location, stats["mean"], stats["sample_sd"]))
        upper = max(mean + sd for _, mean, sd in plotted)
        padding = max(upper * 0.035, 1e-9)
        for location, mean, sd in plotted:
            axis.text(
                location,
                mean + sd + padding,
                f"{mean:.{digits}f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        axis.set_ylim(0, max(upper + padding * 6, 1e-8))
        axis.set_xticks([0.375, 2.475], ["v1: original kernels", "v2: optimized kernels"])
        axis.set_title(title, pad=12)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.17)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Transformer implementation timings across two campaigns",
        y=0.98,
        fontsize=15,
        fontweight="bold",
    )
    figure.legend(
        handles=[Patch(facecolor=COLORS[impl], label=LABELS[impl]) for impl in IMPLEMENTATIONS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        0.08,
        "Historical cross-campaign comparison; machine conditions may differ.\n"
        "Primary controlled conclusions use each campaign's freshly paired PyTorch/ARM64 runs.",
        ha="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.015,
        "Bars: three-seed mean; error bars: sample SD; repeats averaged within each seed.\n"
        f"Source revisions: v1 {first['revision'][:12]}  ·  v2 {second['revision'][:12]}",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.17, 1, 0.84))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    metadata = {
        "comparison_scope": "historical across separate campaigns; not a new paired experiment",
        "forward_scope": "batch one, full context of 32 characters; not cached token generation",
        "summary_sha256": {"v1": first_hash, "v2": second_hash},
        "source_revisions": {"v1": first["revision"], "v2": second["revision"]},
        "statistics": {
            f"v{index + 1}": {
                impl: {
                    field: summary["aggregate"]["by_implementation"][impl][field]
                    for field in FIELDS
                }
                for impl in IMPLEMENTATIONS
            }
            for index, summary in enumerate(summaries)
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_plot(args.v1, args.v2, args.output)
    print(json.dumps({"plot": str(args.output), "metadata": str(args.output.with_suffix(".json"))}))


if __name__ == "__main__":
    main()
