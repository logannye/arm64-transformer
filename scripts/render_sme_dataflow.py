"""Render the SME matrix dataflow after the measured sweep has completed.

Example:
    python scripts/render_sme_dataflow.py --output docs/sme-dataflow.png

This script only draws an implementation diagram; it does not read results,
load a native library, or make performance claims. Use a PNG or SVG output path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLORS = {
    "ink": "#14283E",
    "muted": "#53677B",
    "arrow": "#687E92",
    "line": "#D4DEE8",
    "panel": "#F5F8FC",
    "input_fill": "#FFFFFF",
    "input_edge": "#A6BACD",
    "pack_fill": "#E9F1FD",
    "pack_edge": "#5C88BF",
    "matrix_fill": "#E7F4EF",
    "matrix_edge": "#459D80",
    "output_fill": "#F0EDFB",
    "output_edge": "#9381BE",
}


def render(output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    figure, axis = plt.subplots(figsize=(16.8, 9.4), facecolor="white")
    figure.subplots_adjust(left=0.018, right=0.982, bottom=0.018, top=0.982)
    axis.set_xlim(0, 160)
    axis.set_ylim(0, 92)
    axis.axis("off")

    def text(x, y, value, *, size=11, weight="normal", color="ink", **kwargs):
        return axis.text(
            x,
            y,
            value,
            fontsize=size,
            fontweight=weight,
            color=COLORS.get(color, color),
            va="center",
            **kwargs,
        )

    def panel(y, height):
        axis.add_patch(
            FancyBboxPatch(
                (1, y),
                158,
                height,
                boxstyle="round,pad=0,rounding_size=1.5",
                linewidth=0.8,
                edgecolor=COLORS["line"],
                facecolor=COLORS["panel"],
                zorder=0,
            )
        )

    def box(x, y, width, height, label, *, kind="input", size=11, weight="normal"):
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=0.8",
                linewidth=1.1,
                edgecolor=COLORS[f"{kind}_edge"],
                facecolor=COLORS[f"{kind}_fill"],
                zorder=2,
            )
        )
        text(
            x + width / 2,
            y + height / 2,
            label,
            size=size,
            weight=weight,
            ha="center",
            linespacing=1.55,
            zorder=3,
        )

    def arrow(x0, x1, y):
        axis.add_patch(
            FancyArrowPatch(
                (x0, y),
                (x1, y),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color=COLORS["arrow"],
                shrinkA=3,
                shrinkB=3,
                zorder=1,
            )
        )

    text(
        2,
        89,
        "How the transformer uses SME matrix instructions",
        size=22,
        weight="bold",
    )
    text(
        2,
        85.3,
        "32 token rows · canonical row-major buffers · 64-byte streaming vectors (16 FP32 lanes)",
        size=11.6,
        color="muted",
    )

    # Forward: one full 32-row input, one 32-output weight panel, four tiles.
    panel(47, 35)
    text(6, 78.8, "FORWARD", size=13.2, weight="bold")
    text(31, 78.8, "Y = XWᵀ + bias", size=13.2, color="muted")
    box(6, 55.5, 27, 17, "X  [32, I]\nW  [O, I]", size=12.4)
    box(
        41,
        55.5,
        34,
        17,
        "Pack X → [I, 32]\nPack W panel → [I, 32]",
        kind="pack",
        size=11.5,
    )
    text(58, 53, "X once; W one panel at a time", size=9.7, color="muted", ha="center")

    text(101.5, 73.7, "32 × 32 accumulator", size=11.4, weight="bold", ha="center")
    for tile, x, y in ((0, 88, 63), (1, 102.5, 63), (2, 88, 52.8), (3, 102.5, 52.8)):
        box(x, y, 12.5, 8.7, f"ZA{tile}\n16 × 16", kind="matrix", size=10.7)
    text(
        101.5,
        50.5,
        "Four FP32 outer products per reduction step",
        size=9.3,
        color="muted",
        ha="center",
    )
    box(
        128,
        55.5,
        26,
        17,
        "+ bias panel\nStore Y[:, o:o+32]\n32 × 32 output panel",
        kind="output",
        size=10.4,
    )
    arrow(33, 41, 64)
    arrow(75, 85, 64)
    arrow(118, 128, 64)
    text(6, 49.5, "Repeat over output panels", size=9.6, color="muted")

    # Backward uses the same four-tile accumulator with different operand flows.
    panel(8.3, 35.7)
    text(6, 40.7, "BACKWARD", size=13.2, weight="bold")
    text(31, 40.7, "Separate input-gradient and weight-gradient paths", size=12.2, color="muted")

    box(6, 27, 27, 9.8, "dY  [32, O]", size=12)
    box(41, 27, 34, 9.8, "Pack dY → [O, 32]\nOnce per backward call", kind="pack", size=10.8)
    box(84, 27, 35, 9.8, "4 ZA tiles: dY · W\nRead W rows directly", kind="matrix", size=10.8)
    box(128, 27, 26, 9.8, "Overwrite dX\nCanonical [32, I]", kind="output", size=10.8)
    arrow(33, 41, 31.9)
    arrow(75, 84, 31.9)
    arrow(119, 128, 31.9)

    box(6, 14, 27, 9.8, "dY rows + X rows", size=11.1)
    box(41, 14, 34, 9.8, "Read canonical rows\nNo operand packing", kind="input", size=10.8)
    box(
        84,
        14,
        35,
        9.8,
        "Load existing dW tile\nAdd dYᵀ · X in 4 ZA tiles",
        kind="matrix",
        size=10.5,
    )
    box(128, 14, 26, 9.8, "Accumulate dW\nCanonical [O, I]", kind="output", size=10.8)
    arrow(33, 41, 18.9)
    arrow(75, 84, 18.9)
    arrow(119, 128, 18.9)
    text(6, 10.7, "Bias gradient: db += sum of dY rows, in token order", size=10.2, color="muted")
    text(154, 10.7, "dW and db accumulate across sequences", size=10.2, color="muted", ha="right")

    text(
        80,
        4.8,
        "FP32 throughout. Packing, mode transitions, and stack-page touches "
        "are inside each measured call.",
        size=10.7,
        color="muted",
        ha="center",
    )
    text(
        80,
        1.9,
        "Forward scratch ≤128 KiB · backward scratch ≤96 KiB · no persistent packed-weight cache",
        size=10.7,
        weight="bold",
        color="muted",
        ha="center",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination PNG or SVG")
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".png", ".svg"}:
        parser.error("--output must end in .png or .svg")
    if args.output.exists():
        parser.error("Output exists; choose a fresh path")
    render(args.output)


if __name__ == "__main__":
    main()
