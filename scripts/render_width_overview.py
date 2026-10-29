"""Plot separate campaigns; a failed current audit requires explicit descriptive labeling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-failed-current", action="store_true")
    args = parser.parse_args()
    record = args.output.with_suffix(".json")
    if args.output.exists() or record.exists():
        raise FileExistsError("Overview image or provenance already exists")
    sources, rows = {}, []
    for directory, version, widths in (
        (args.historical, "v4", [32, 64, 128, 256]),
        (args.current, "v5", [512, 1024]),
    ):
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        failed = summary["status"] == "audit_failed"
        admitted_status = summary["status"] == "completed" or (
            version == "v5"
            and args.allow_failed_current
            and failed
            and summary.get("evidence_integrity_passed") is True
            and summary.get("post_run_failed_check_count", 0) > 0
            and summary.get("fit_count") == 24
        )
        if not (
            admitted_status
            and summary["protocol_version"] == version
            and summary["widths"] == widths
            and summary["training_target_tokens_per_fit"] == 524288
        ):
            raise ValueError("Expected the declared fixed-budget report and explicit audit status")
        sources[version] = {
            "path": str(path),
            "sha256": digest(path),
            "measured_revision": summary["revision"],
            "fits": len(summary["runs"]),
            "numerical_audit_status": summary["status"],
        }
        for width in widths:
            values = summary["descriptive_by_width" if failed else "by_width"][str(width)]
            rows.append(
                {
                    "width": width,
                    "campaign": version,
                    "audit_failed": failed,
                    "training_ratio": values["aggregate"][
                        "geometric_mean_training_ratio_python_over_arm64"
                    ],
                    "forward_ratio": values[
                        "forward_mean_p50_ratio_python_over_arm64"
                        if failed
                        else "forward_p50_mean_ratio_python_over_arm64"
                    ],
                }
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axis = plt.subplots(figsize=(11, 6.4))
    positions = np.arange(len(rows))
    for key, offset, color, label in (
        ("training_ratio", -0.18, "#137E8D", "Training: geometric mean of paired ratios"),
        ("forward_ratio", 0.18, "#CC6A30", "Forward: ratio of mean p50 latencies"),
    ):
        values = [r[key] for r in rows]
        bars = axis.barh(positions + offset, values, height=0.30, color=color, label=label)
        for bar, row in zip(bars, rows, strict=True):
            bar.set_alpha(0.48 if row["campaign"] == "v4" else 1)
            if row["audit_failed"]:
                bar.set_hatch("///")
                bar.set_edgecolor("#FFFFFF")
        labels = axis.bar_label(bars, labels=[f"{v:.2f}×" for v in values], padding=6, fontsize=10)
        for label in labels:
            label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 1})
    axis.axvline(1, color="#333333", linestyle="--", linewidth=1)
    axis.axhline(3.5, color="#98A2B3", linewidth=0.8)
    axis.set_yticks(
        positions,
        [f"{r['width']}  ·  {r['campaign']}" + ("*" if r["audit_failed"] else "") for r in rows],
    )
    axis.invert_yaxis()
    axis.set_xlim(0, max(r[k] for r in rows for k in ("training_ratio", "forward_ratio")) * 1.18)
    axis.set_ylabel("Model width · campaign", labelpad=12)
    axis.set_xlabel("PyTorch time / ARM64 time", labelpad=10)
    axis.set_title("Hardware gains depend on model size", pad=18, weight="bold")
    axis.grid(axis="x", alpha=0.16)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(axis="y", length=0)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.075), frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.035,
        "Above 1 favors ARM64. Each width: 3 seeds × 2 repeats, one M4 Max CPU thread.",
        ha="center",
        fontsize=9,
        color="#475467",
    )
    fig.text(
        0.5,
        0.009,
        (
            "* V5 numerical audit FAILED: hatched ratios are descriptive. "
            "V4 is historical; no pooled estimate."
            if any(row["audit_failed"] for row in rows)
            else "Separate v4/v5 campaigns; no pooled estimate. Lighter bars show historical v4."
        ),
        ha="center",
        fontsize=9,
        color="#475467",
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    record.write_text(
        json.dumps(
            {
                "scope": __doc__,
                "sources": sources,
                "rows": rows,
                "image_sha256": digest(args.output),
                "renderer_sha256": digest(Path(__file__)),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
