"""Render the retained precision-development inventory; no model or GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("reference-v0", "p1", "p1-complete", "p2", "p3")
LABELS = (
    "C0: reference",
    "P1: first implementation",
    "P1: completed",
    "P2: add normalization",
    "P3: add QKV precision",
)
COLORS = {
    "ink": "#14283E",
    "muted": "#53677B",
    "passed": "#E7F4EF",
    "failed": "#FAE7DD",
    "not_run": "#F0F3F7",
    "line": "#D4DEE8",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(directory):
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    for name in ("summary.json", "receipts.json"):
        if digest(directory / name) != provenance["derived_payloads"][name]:
            raise ValueError(f"Publication payload changed: {name}")
    if summary["status"] != "development_pass_formal_gate_closed":
        raise ValueError("This renderer is scoped to the frozen development result")
    if summary["final_numeric_comparisons_total"] != 3186:
        raise ValueError("Unexpected declared comparison denominator")
    widths = summary["widths"]
    cells = {(cell["phase"], cell["family"], cell["width"]): cell for cell in summary["inventory"]}

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "savefig.facecolor": "white"}
    )
    fig = plt.figure(figsize=(13, 10), facecolor="white")
    fig.text(
        0.06,
        0.96,
        "Preserve small differences before rounding",
        color=COLORS["ink"],
        fontsize=20,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.06,
        0.919,
        "Metal precision development  |  Same equations and FP32 state  |  Unchanged limits",
        color=COLORS["muted"],
        fontsize=11.5,
    )

    boxes = (
        (0.15, "FP32 stored inputs"),
        (0.445, "Temporary two-float\naccumulation"),
        (0.74, "FP32 stored outputs"),
    )
    for left, text in boxes:
        fig.add_artist(
            FancyBboxPatch(
                (left, 0.831),
                0.22,
                0.062,
                boxstyle="round,pad=0.008,rounding_size=0.008",
                transform=fig.transFigure,
                facecolor="#F5F8FC",
                edgecolor=COLORS["line"],
            )
        )
        fig.text(
            left + 0.11, 0.862, text, ha="center", va="center", fontsize=11, color=COLORS["ink"]
        )
    for left in (0.376, 0.671):
        fig.add_artist(
            FancyArrowPatch(
                (left, 0.862),
                (left + 0.056, 0.862),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color=COLORS["muted"],
            )
        )

    for family, bottom, title, denominator in (
        ("attention", 0.465, "Attention: passed cases / 23", 23),
        ("ln-qkv", 0.135, "LayerNorm + QKV: passed cases / 22", 22),
    ):
        axis = fig.add_axes((0.245, bottom, 0.71, 0.255))
        axis.set_xlim(0, len(widths))
        axis.set_ylim(5, 0)
        axis.set_xticks([i + 0.5 for i in range(len(widths))], [str(w) for w in widths])
        axis.set_yticks([i + 0.5 for i in range(5)], LABELS)
        axis.xaxis.tick_top()
        axis.tick_params(axis="both", length=0, pad=9, labelcolor=COLORS["ink"])
        axis.set_title(title, loc="left", pad=34, color=COLORS["ink"], weight="bold", fontsize=13)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for row, phase in enumerate(PHASES):
            for column, width in enumerate(widths):
                cell = cells[phase, family, width]
                status = cell["status"]
                axis.add_patch(
                    Rectangle(
                        (column + 0.025, row + 0.04),
                        0.95,
                        0.92,
                        facecolor=COLORS[status],
                        edgecolor="white",
                        linewidth=1,
                    )
                )
                if status == "not_run":
                    text, color, size = "not run", COLORS["muted"], 10
                else:
                    if cell["total_cases"] != denominator:
                        raise ValueError("Incomplete stress inventory")
                    text = f"{cell['passed_cases']}/{denominator}"
                    color = "#205B48" if status == "passed" else "#954927"
                    size = 12
                axis.text(
                    column + 0.5,
                    row + 0.5,
                    text,
                    color=color,
                    ha="center",
                    va="center",
                    fontsize=size,
                    fontweight="bold" if status != "not_run" else "normal",
                )

    fig.legend(
        handles=[
            Patch(facecolor=COLORS[key], label=label)
            for key, label in (
                ("passed", "All cases pass"),
                ("failed", "At least one failure"),
                ("not_run", "Not executed"),
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.6, 0.074),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.06,
        0.054,
        "P3: 3,186 numerical comparisons pass; controls are recorded separately.",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.06,
        0.027,
        "Complete named stress inventories only. Development evidence; "
        "formal admission and measurement remain closed.",
        fontsize=9.7,
        color=COLORS["muted"],
    )
    output = directory / "precision-development.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    metadata = {
        "scope": "Static development result presentation; no new model computations",
        "summary_sha256": digest(summary_path),
        "receipts_sha256": digest(directory / "receipts.json"),
        "renderer_sha256": digest(Path(__file__)),
        "output_sha256": digest(output),
        "matplotlib": matplotlib.__version__,
    }
    (directory / "figure-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "experiments/gpu-development-v1")
    render(parser.parse_args().directory)
