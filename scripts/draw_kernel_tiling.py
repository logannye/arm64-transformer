"""Draw the actual four-token/four-output NEON register tile as a static PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.5)
ax.axis("off")
ax.text(6, 5.25, "One step inside the tiled matrix kernel", ha="center", fontsize=20, weight="bold")
ax.text(
    6, 4.82, "Compute four token rows × four output channels together", ha="center", fontsize=12
)


def vectors(x, color, title, row_label):
    ax.text(x + 1.1, 4.25, title, ha="center", fontsize=13, weight="bold")
    for row in range(4):
        y = 3.55 - row * 0.52
        ax.text(x - 0.1, y + 0.2, f"{row_label} {row}", ha="right", va="center", fontsize=10)
        for lane in range(4):
            ax.add_patch(Rectangle((x + lane * 0.5, y), 0.47, 0.4, facecolor=color, alpha=0.85))
    ax.text(
        x + 0.95,
        1.48,
        "4 loaded vectors\n4 float32 lanes each",
        ha="center",
        fontsize=11,
        linespacing=1.5,
    )


vectors(0.9, "#3568ad", "Input values", "token")
vectors(4.1, "#d66b34", "Weight values", "output")
ax.text(3.4, 2.95, "&", fontsize=24, ha="center", va="center")
ax.add_patch(
    FancyArrowPatch(
        (6.3, 2.95), (7.1, 2.95), arrowstyle="-|>", mutation_scale=18, color="#475569", linewidth=2
    )
)
ax.text(9.15, 4.25, "16 independent accumulators", ha="center", fontsize=13, weight="bold")
for row in range(4):
    for col in range(4):
        x, y = 7.55 + col * 0.82, 3.55 - row * 0.52
        ax.add_patch(Rectangle((x, y), 0.74, 0.4, facecolor="#388777", alpha=0.92))
        ax.text(
            x + 0.37,
            y + 0.2,
            f"v{row * 4 + col}",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
        )
ax.text(
    9.15,
    1.48,
    "Each cell is a 4-lane NEON register\nPartial sums stay in registers",
    ha="center",
    fontsize=11,
    linespacing=1.5,
)
ax.text(
    6,
    0.79,
    "Each loaded vector contributes to four outputs in the tile.",
    ha="center",
    fontsize=12,
    weight="bold",
)
ax.text(
    6,
    0.37,
    "Repeat over the input width in groups of four, then sum each register's lanes and add bias.",
    ha="center",
    fontsize=11,
    color="#475569",
)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
fig.savefig(ROOT / "docs/kernel-tiling.png", dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
