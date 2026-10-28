"""Draw the README's static architecture diagram, without interactive controls."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

root = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(13, 3.2), facecolor="white")
ax.set_xlim(0, 13)
ax.set_ylim(0, 3.2)
ax.axis("off")
ax.text(0.1, 2.93, "ONE MODEL · TWO IMPLEMENTATIONS", fontsize=16, weight="bold", color="#17324D")
ax.text(
    0.1,
    2.58,
    "PyTorch and ARM64 follow the same forward and learning computation",
    fontsize=12,
    color="#506477",
)
labels = [
    "32 input\ncharacters",
    "Token + position\nembeddings",
    "LayerNorm\nCausal attention\nResidual addition",
    "LayerNorm\nFeed-forward\nResidual addition",
    "Final LayerNorm\nVocabulary head",
    "128 next-character\nlogits per position",
]
for index, label in enumerate(labels):
    left = 0.08 + index * 2.17
    ax.add_patch(
        FancyBboxPatch(
            (left, 1.18),
            1.86,
            1.0,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="#EAF1F9",
            edgecolor="#6484A5",
            linewidth=1.3,
        )
    )
    ax.text(
        left + 0.93,
        1.68,
        label,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#17324D",
        linespacing=1.35,
    )
    if index < len(labels) - 1:
        ax.add_patch(
            FancyArrowPatch(
                (left + 1.91, 1.68),
                (left + 2.10, 1.68),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color="#426584",
            )
        )
ax.add_patch(
    FancyArrowPatch(
        (12.0, 1.09),
        (12.0, 0.71),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.4,
        color="#A65528",
    )
)
ax.add_patch(
    FancyBboxPatch(
        (0.08, 0.12),
        12.7,
        0.53,
        boxstyle="round,pad=0.03,rounding_size=0.07",
        facecolor="#FFF2E9",
        edgecolor="#C9875D",
        linewidth=1.2,
    )
)
ax.text(
    6.43,
    0.38,
    "Learning: next-character cross-entropy → backpropagation through every layer "
    "→ Adam parameter updates",
    ha="center",
    va="center",
    fontsize=11,
    color="#774322",
)
(root / "docs").mkdir(exist_ok=True)
fig.savefig(root / "docs/architecture.png", dpi=180, bbox_inches="tight", pad_inches=0.12)
plt.close(fig)
