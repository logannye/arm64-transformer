"""Render README figures from retained reports; never execute experiment workers.

Run with the existing report environment:
    MPLBACKEND=Agg MPLCONFIGDIR=work/readme-mpl python docs/render_readme_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/gpu-v5-library-phase1/results"
FIGURES = ROOT / "docs/figures"
INK = "#172B46"
MUTED = "#586A80"
RULE = "#DDE5ED"
FAINT = "#F4F7FB"
ENGINES = (
    ("cpu_arm64", "ARM64", "#245BC5", "o"),
    ("cpu_torch", "PyTorch CPU", "#687C94", "s"),
    ("metal", "Metal", "#11826D", "^"),
    ("mps", "MPS", "#99648E", "D"),
    ("mlx", "MLX", "#B56A16", "v"),
)


def configure():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 13,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": RULE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "svg.hashsalt": "arm64-transformer-readme-v1",
        }
    )


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180, facecolor="white")
    svg = path.with_suffix(".svg")
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def heading(fig, category, title, subtitle):
    fig.text(0.045, 0.952, category, fontsize=10.5, color=MUTED, weight="bold")
    fig.text(0.045, 0.893, title, fontsize=23, weight="bold")
    fig.text(0.045, 0.846, subtitle, fontsize=12.5, color=MUTED)


def performance():
    data = json.loads((RESULTS / "summary.json").read_text())
    assert data["scientific_acceptance"] and data["formal_fits"] == 120
    widths = [32, 64, 128, 256]
    fig = plt.figure(figsize=(14, 7.8))
    heading(
        fig,
        "ACCEPTED FORMAL RESULTS",
        "Training and inference on Apple M4 Max",
        "120 fits · widths 32–256 · three seeds × two repeats · lower is better",
    )
    panels = (
        ("median_training_seconds", 1, "Training", "Full fit · batch 16", "Seconds", "s"),
        (
            "median_latency_p50_seconds",
            1000,
            "Inference",
            "Resident forward · batch 1",
            "Milliseconds",
            "ms",
        ),
    )
    for index, (field, scale, title, subtitle, unit, suffix) in enumerate(panels):
        left = 0.075 + index * 0.48
        ax = fig.add_axes((left, 0.245, 0.40, 0.48))
        fig.text(left, 0.777, title, fontsize=17, weight="bold")
        fig.text(left, 0.741, subtitle, fontsize=11.5, color=MUTED)
        for engine, label, color, marker in ENGINES:
            rows = sorted(
                (row for row in data["groups"] if row["engine"] == engine),
                key=lambda row: row["width"],
            )
            assert [row["width"] for row in rows] == widths
            assert all(row["fits"] == 6 for row in rows)
            values = [row[field] * scale for row in rows]
            ax.plot(
                widths,
                values,
                color=color,
                marker=marker,
                linewidth=2.3,
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
            label_y = values[-1]
            if index == 0 and engine == "cpu_arm64":
                label_y = 5.0
            elif index == 0 and engine == "metal":
                label_y = 2.7
            value = f"{values[-1]:.3g}"
            label_text = f"{label}  {value} {suffix}"
            if engine == "cpu_torch":
                label_text = f"{label}\n{value} {suffix}"
            ax.annotate(
                label_text,
                xy=(256, values[-1]),
                xytext=(305, label_y),
                fontsize=11.2,
                color=color,
                weight="bold",
                va="center",
                annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(27, 720)
        ax.set_xticks(widths, labels=[str(width) for width in widths])
        ax.tick_params(axis="both", length=0, pad=8, labelsize=12)
        ax.set_xlabel("Model width", fontsize=12, labelpad=11)
        ax.set_ylabel(f"{unit} · log scale", fontsize=12, labelpad=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", color=RULE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_bounds(32, 256)
        if index == 0:
            ax.set_ylim(0.5, 600)
            ax.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
        else:
            ax.set_ylim(0.01, 40)
            ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
    fig.text(
        0.045,
        0.122,
        "Points show six-fit medians. End labels give width-256 values. "
        "Inference uses each fit’s p50 latency.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.079,
        "PyTorch / MPS / MLX: maintained builds with selected precision. "
        "Metal: reference kernel policy.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.036,
        "FP32 model storage · timing boundaries in the report · widths 512 / 1024 remain deferred",
        fontsize=11,
        color=MUTED,
    )
    save(fig, RESULTS / "performance")


def box(ax, x, y, width, height, *, edge=RULE, fill=FAINT):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.0,
    )
    ax.add_patch(patch)


def overview():
    fig = plt.figure(figsize=(14, 6.1))
    heading(
        fig,
        "MODEL & EXECUTION",
        "One transformer, five complete implementations",
        "Context 32 · four causal attention heads · feed-forward width 2W · ASCII vocabulary 128",
    )
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    stages = (
        (0.055, "Embed", "Token + position"),
        (0.295, "Attention", "LayerNorm → attention\nProjection + residual"),
        (0.535, "Feed-forward", "LayerNorm → FF1 → ReLU\nFF2 + residual · hidden 2W"),
        (0.775, "Readout", "LayerNorm → linear\n128 next-character logits"),
    )
    for x, title, detail in stages:
        box(ax, x, 0.555, 0.17, 0.19)
        ax.text(x + 0.085, 0.687, title, ha="center", fontsize=15, weight="bold")
        ax.text(x + 0.085, 0.611, detail, ha="center", va="center", fontsize=11.5, color=MUTED)
        if x < 0.775:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.184, 0.649),
                    (x + 0.226, 0.649),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    color="#8C9BB0",
                    linewidth=1.1,
                )
            )
    ax.plot([0.295, 0.295, 0.705, 0.705], [0.765, 0.783, 0.783, 0.765], color=RULE, lw=1)
    ax.text(0.50, 0.796, "ONE CAUSAL DECODER BLOCK", ha="center", fontsize=9.5, color=MUTED)
    ax.text(
        0.5,
        0.474,
        "Every engine executes forward propagation, backward propagation and Adam",
        ha="center",
        fontsize=13,
        weight="bold",
    )
    ax.text(0.055, 0.374, "CPU", fontsize=11, color=MUTED, weight="bold")
    ax.text(0.465, 0.374, "GPU", fontsize=11, color=MUTED, weight="bold")
    ax.plot([0.10, 0.41], [0.381, 0.381], color=RULE, lw=1)
    ax.plot([0.51, 0.945], [0.381, 0.381], color=RULE, lw=1)
    descriptions = (
        (0.055, "SME / NEON assembly"),
        (0.245, "Maintained PyTorch"),
        (0.465, "Custom reference kernels"),
        (0.635, "Maintained PyTorch"),
        (0.805, "Maintained, compiled MLX"),
    )
    for (_, label, color, _), (x, detail) in zip(ENGINES, descriptions, strict=True):
        width = 0.16 if x < 0.45 else 0.14
        # Keep the five columns aligned without enclosing them in decorative cards.
        ax.plot([x, x + width], [0.329, 0.329], color=color, lw=3)
        ax.text(x, 0.267, label, fontsize=15, weight="bold", color=color)
        detail = detail.replace("Maintained, compiled MLX", "Maintained MLX\nCompiled graph")
        detail = detail.replace("Custom reference kernels", "Custom kernels\nReference policy")
        ax.text(x, 0.207, detail, fontsize=11, va="top", color=MUTED, linespacing=1.6)
    ax.text(
        0.055,
        0.067,
        "Fixed equations, fixtures and schedule · FP32 stored state · "
        "internal precision declared per engine",
        fontsize=11.5,
        color=MUTED,
    )
    save(fig, FIGURES / "model-and-engines")


def precision():
    data = json.loads(
        (ROOT / "experiments/trained-precision-v1/mlx-branch-localization.json").read_text()
    )
    assert data["complete"] and not data["formal_eligible"] and data["gate_disagreements"] == 1
    example = data["gate_examples"][0]
    reference, observed = example["fp64"] * 1e7, example["mlx"] * 1e7
    failed = data["original_comparison"]["failed_coordinates"]
    fig = plt.figure(figsize=(14, 5.0))
    heading(
        fig,
        "DEVELOPMENT DIAGNOSTIC · WIDTH 1024",
        "A tiny sign change can switch a gradient on or off",
        "Observed FF1 preactivation values, shown against the ReLU backward gate",
    )
    ax = fig.add_axes((0.09, 0.285, 0.52, 0.40))
    ax.plot([-2.5, 0], [0, 0], color="#B5C0CF", lw=3)
    ax.plot([0, 5.1], [1, 1], color="#B5C0CF", lw=3)
    ax.plot([0, 0], [0, 1], color=RULE, lw=1.2, linestyle=(0, (3, 4)))
    ax.scatter([reference, observed], [0, 1], color=[INK, ENGINES[4][2]], s=90, zorder=4)
    ax.annotate(
        f"FP64 reference\n{reference:.2f} × 10⁻⁷",
        xy=(reference, 0),
        xytext=(-2.25, 0.40),
        fontsize=12,
        color=INK,
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
    )
    ax.annotate(
        f"MLX diagnostic\n+{observed:.2f} × 10⁻⁷",
        xy=(observed, 1),
        xytext=(2.3, 0.30),
        fontsize=12,
        color=ENGINES[4][2],
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": ENGINES[4][2], "lw": 0.8},
    )
    ax.set(xlim=(-2.5, 5.1), ylim=(-0.12, 1.25))
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_yticks([0, 1], labels=["0 · block", "1 · pass"])
    ax.set_xlabel("FF1 preactivation (× 10⁻⁷)", labelpad=10, fontsize=12)
    ax.set_ylabel("Gradient multiplier", labelpad=12, fontsize=12)
    ax.tick_params(length=0, pad=8, labelsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(0.685, 0.63, "Forward checks passed.", fontsize=16, weight="bold")
    fig.text(
        0.685,
        0.47,
        f"{failed} FF1 weight-gradient coordinates\nexceeded the frozen tolerance.",
        fontsize=14,
        color=MUTED,
        linespacing=1.6,
    )
    fig.text(
        0.685,
        0.29,
        "Gate substitution isolated\nthe mismatch to this branch.",
        fontsize=12,
        color=MUTED,
        linespacing=1.6,
    )
    fig.text(
        0.045,
        0.065,
        "Historical, unselected MLX candidate. "
        "This diagnostic does not qualify width 1024 for formal results.",
        fontsize=11,
        color=MUTED,
    )
    save(fig, FIGURES / "precision-gate")


if __name__ == "__main__":
    configure()
    overview()
    performance()
    precision()
