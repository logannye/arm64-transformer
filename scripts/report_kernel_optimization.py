"""Render separately scoped synthetic v4 profiling evidence, without timing code."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.profile_native import parse_sample  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], {}
    for width in (32, 64, 128, 256):
        kernel = args.candidate / f"width-{width}-kernels.json"
        old_sample = args.baseline / f"width-{width}-sample.txt"
        new_sample = args.candidate / f"width-{width}-runtime/sample.txt"
        records = json.loads(kernel.read_text())
        for path in (kernel, old_sample, new_sample):
            sources[str(path)] = digest(path)
        row = {"width": width, "libraries": records["libraries"], "kernels": {}}
        for case in ("full_forward", "full_backward_including_cross_entropy"):
            old = records["cases"]["baseline/" + case]["median_us"]
            new = records["cases"]["candidate/" + case]["median_us"]
            row["kernels"][case] = {
                "v3_us": old,
                "v4_us": new,
                "v3_over_v4": old / new,
            }
        row["runtime_samples"] = {
            "v3": parse_sample(old_sample.read_text()),
            "v4": parse_sample(new_sample.read_text()),
        }
        rows.append(row)
    summary = {
        "scope": (
            "Seed-777 synthetic diagnostics, separate from the 48-fit matched study. "
            "Kernel times are fresh balanced v3/v4 comparisons including dispatch, FFI "
            "and packing. Native stack samples are separate approximate process windows; "
            "percentages describe shares, not absolute time saved. Final v4 sampling "
            "deliberately stops each owned process and retains its partial outputs."
        ),
        "rows": rows,
        "source_sha256": sources,
        "renderer_sha256": digest(Path(__file__)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    x = np.arange(4)
    colors = {"v3": "#667085", "v4": "#087F8C"}
    for axis, case, title in (
        (axes[0], "full_forward", "Complete native forward"),
        (axes[1], "full_backward_including_cross_entropy", "Complete native backward + loss"),
    ):
        for version in ("v3", "v4"):
            y = [r["kernels"][case][version + "_us"] for r in rows]
            axis.plot(x, y, marker="o", linewidth=2.5, color=colors[version], label=version)
        axis.set_yscale("log")
        axis.set_ylabel("Median call time (µs, log scale)")
        axis.set_title(title, fontsize=12, weight="bold", pad=14)
        axis.grid(axis="y", alpha=0.18, which="both")
    axis = axes[2]
    for version, offset in (("v3", -0.19), ("v4", 0.19)):
        y = [
            r["runtime_samples"][version]["dense_linear_percent_of_post_loader_samples"]
            for r in rows
        ]
        bars = axis.bar(x + offset, y, width=0.34, color=colors[version])
        axis.bar_label(bars, labels=[f"{v:.0f}%" for v in y], padding=3, fontsize=9)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Dense matrix share of runtime samples (%)")
    axis.set_title("Measured bottleneck, including packing", fontsize=12, weight="bold", pad=14)
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)
    for axis in axes:
        axis.set_xticks(x, [str(r["width"]) for r in rows])
        axis.set_xlabel("Model width")
    fig.suptitle(
        "Restructuring matrix work changes the scaling curve", fontsize=17, weight="bold", y=0.98
    )
    fig.legend(
        axes[0].lines,
        ["v3: NEON tiles", "v4: revised NEON / SME matrix tiles"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "Synthetic diagnostics • one Apple M4 Max CPU thread • FP32 • packing included"
        " • sample shares are not elapsed times",
        ha="center",
        fontsize=10,
        color="#475467",
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    path = args.output / "kernel-diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    (args.output / "figure.json").write_text(
        json.dumps(
            {"sha256": digest(path), "summary_sha256": digest(args.output / "summary.json")},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
