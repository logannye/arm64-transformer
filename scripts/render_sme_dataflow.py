"""Render the SME matrix dataflow outside measured benchmark execution.

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
    figure, axis = plt.subplots(figsize=(16.8, 11.6), facecolor="white")
    figure.subplots_adjust(left=0.018, right=0.982, bottom=0.018, top=0.982)
    axis.set_xlim(0, 160)
    axis.set_ylim(0, 112)
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
        108.5,
        "How the transformer uses SME matrix instructions",
        size=22,
        weight="bold",
    )
    text(
        2,
        104.5,
        "32 token rows · canonical row-major buffers · 64-byte streaming vectors (16 FP32 lanes)",
        size=11.6,
        color="muted",
    )

    # Wide forward: the same32-row packs, four independent reduction chains
    # for one16x16 subtile at a time. Repeat four times per32x32 output panel.
    panel(53, 48)
    text(6, 97.5, "FORWARD", size=13.2, weight="bold")
    text(31, 97.5, "Y = XWᵀ + bias", size=13.2, color="muted")
    text(154, 97.5, "Compiled widths 512 / 1024", size=11.8, weight="bold", ha="right")
    box(6, 73.5, 25, 17, "X  [32, I]\nW  [O, I]", size=12.4)
    box(
        38,
        73.5,
        34,
        17,
        "Pack X → [I, 32]\nPack W panel → [I, 32]",
        kind="pack",
        size=11.5,
    )
    text(55, 70.8, "X once; W one panel at a time", size=9.7, color="muted", ha="center")

    text(101, 93.1, "Four chains for one 16 × 16 subtile", size=11.1, weight="bold", ha="center")
    for tile, x, y in ((0, 81, 82), (1, 102, 82), (2, 81, 70.5), (3, 102, 70.5)):
        box(
            x,
            y,
            19,
            9.7,
            f"ZA{tile}  ·  16 × 16\nk = {tile}, {tile + 4}, {tile + 8}, …",
            kind="matrix",
            size=10.2,
        )
    text(
        101,
        66.8,
        "(ZA0 + ZA1) + (ZA2 + ZA3) + bias",
        size=11.1,
        weight="bold",
        ha="center",
    )
    text(
        101,
        63.4,
        "k is the input-feature index; all sums are FP32",
        size=9.8,
        color="muted",
        ha="center",
    )
    box(
        130,
        73.5,
        24,
        17,
        "Store 16 × 16 subtile\nRepeat for 4 subtiles\n→ 32 × 32 output panel",
        kind="output",
        size=10.1,
    )
    arrow(31, 38, 82)
    arrow(72, 80, 82)
    arrow(122, 130, 82)
    text(
        6,
        65.5,
        "Shorter accumulation chains;\nadditional operand loads",
        size=10.6,
        color="muted",
        linespacing=1.5,
    )
    text(
        80,
        57,
        "SME calls in compiled widths ≤256 retain 4 spatial ZA tiles per 32 × 32 result.\n"
        "The width-32 model uses NEON.",
        size=10.5,
        color="muted",
        ha="center",
        linespacing=1.5,
    )

    # Backward uses the same four-tile accumulator with different operand flows.
    panel(14, 35.7)
    text(6, 46.4, "BACKWARD", size=13.2, weight="bold")
    text(
        31,
        46.4,
        "Unchanged: separate input-gradient and weight-gradient paths",
        size=12.2,
        color="muted",
    )

    box(6, 32.7, 27, 9.8, "dY  [32, O]", size=12)
    box(41, 32.7, 34, 9.8, "Pack dY → [O, 32]\nOnce per backward call", kind="pack", size=10.8)
    box(84, 32.7, 35, 9.8, "4 ZA tiles: dY · W\nRead W rows directly", kind="matrix", size=10.8)
    box(128, 32.7, 26, 9.8, "Overwrite dX\nCanonical [32, I]", kind="output", size=10.8)
    arrow(33, 41, 37.6)
    arrow(75, 84, 37.6)
    arrow(119, 128, 37.6)

    box(6, 19.7, 27, 9.8, "dY rows + X rows", size=11.1)
    box(41, 19.7, 34, 9.8, "Read canonical rows\nNo operand packing", kind="input", size=10.8)
    box(
        84,
        19.7,
        35,
        9.8,
        "Load existing dW tile\nAdd dYᵀ · X in 4 ZA tiles",
        kind="matrix",
        size=10.5,
    )
    box(128, 19.7, 26, 9.8, "Accumulate dW\nCanonical [O, I]", kind="output", size=10.8)
    arrow(33, 41, 24.6)
    arrow(75, 84, 24.6)
    arrow(119, 128, 24.6)
    text(6, 16.4, "Bias gradient: db += sum of dY rows, in token order", size=10.2, color="muted")
    text(154, 16.4, "dW and db accumulate across sequences", size=10.2, color="muted", ha="right")

    text(
        80,
        10.3,
        "Packing, mode transitions, and stack-page touches are inside each call. "
        "No persistent packed-weight cache.",
        size=10.7,
        color="muted",
        ha="center",
    )
    text(
        80,
        6.8,
        "Maximum stack scratch: forward 512 KiB + 160 B frame · backward 384 KiB + 192 B frame",
        size=10.7,
        weight="bold",
        color="muted",
        ha="center",
    )
    text(
        80,
        3.3,
        "Caller stack must accommodate scratch + saved frame + its own active frames; "
        "the largest forward call exceeds 512 KiB.",
        size=10.5,
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
