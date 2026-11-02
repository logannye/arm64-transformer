"""Publish closed development diagnostics as static figures and byte-preserving receipts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import audit_layout_diagnostic as auditor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

RUNTIME_SUMMARIES = {
    1: "a0df7a678c6749d7deb5d7e636c541a3537bdadd3812c5ba8b29567fde213dbb",
    2: "28006a5e3b52b6aaee075f180956a83e74514f757e687dcae4792b3eb26537dd",
    3: "3484b72f1951247c68d78b5ad167ea0d0d4355a902b20d6e60d81ad41a71e3b0",
    4: "f747bad17d83bf0af85fa6a8bdf73643e9c0abb989e2ff56cb7cd4525bdf9f4b",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def receipt_sources(workspace):
    root = workspace / "work/gpu-v4-owned"
    sources = [root / "joint-runtime-v3-retry.json"]
    for version, expected in RUNTIME_SUMMARIES.items():
        folder = root / f"joint-runtime-v{version}"
        auditor.frozen_file(folder / "summary.json", expected)
        summary = auditor.read(folder / "summary.json")
        auditor.require(
            summary["formal_eligible"] is False
            and summary["formal_fits"] == 0
            and summary["admission_stream_opened"] is False,
            "Runtime scope changed",
        )
        auditor.require(summary["passed"] is (version == 4), "Runtime result changed")
        for worker in summary["workers"]:
            path = folder / f"{worker['engine']}-p{worker['process_index']}" / "result.json"
            auditor.frozen_file(path, worker["sha256"])
        sources.extend(path for path in folder.rglob("*") if path.is_file())
    for name in ("layout-diagnostic-v1", "failed-runtime-probe-v1", "failed-runtime-probe-v2"):
        sources.extend(
            path
            for path in (root / name).rglob("*")
            if path.is_file() and path.suffix in (".json", ".log", ".py")
        )
    for version in (1, 2):
        folder = root / f"failed-runtime-probe-v{version}"
        snapshot = auditor.read(folder / "snapshot.json")
        entries = (
            snapshot.items()
            if version == 1
            else ((Path(v["copy"]).name, v) for v in snapshot["files"])
        )
        for name, record in entries:
            auditor.frozen_file(folder / name, record["sha256"], record["bytes"])
    return sorted(sources)


def draw(audit, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    values = np.array(
        [
            [
                worker[key]
                for key in (
                    "ordinary_matrix_failures",
                    "control_matrix_failures",
                    "guard_failures",
                    "contiguous_convolution_fp64_failures",
                    "channels_last_convolution_fp64_failures",
                )
            ]
            for worker in audit["workers"]
        ]
    )
    denominators = (76, 76, 76, 6, 6)
    fig, axis = plt.subplots(figsize=(12, 5.4))
    fig.patch.set_facecolor("#f7f9fc")
    axis.imshow(
        values > 0, cmap=ListedColormap(["#d9eee8", "#f3c5b8"]), vmin=0, vmax=1, aspect="auto"
    )
    for row in range(4):
        for column in range(5):
            axis.text(
                column,
                row,
                f"{values[row, column]}/{denominators[column]}",
                ha="center",
                va="center",
                color="#173a38" if values[row, column] == 0 else "#752718",
                weight="bold",
                fontsize=17,
            )
    axis.set_xticks(
        range(5),
        [
            "Strided matrix\noutput",
            "Matrix: GPU temporary\nthen strided copy",
            "Matrix output\nguard failures",
            "Conv3d: contiguous\nNCDHW",
            "Conv3d: channels-last\nNDHWC",
        ],
    )
    axis.xaxis.tick_top()
    axis.tick_params(axis="both", length=0, pad=10)
    axis.set_yticks(
        range(4),
        [
            "Ordinary · process 1",
            "Ordinary · process 2",
            "Metal 3.2 diagnostic · process 1",
            "Metal 3.2 diagnostic · process 2",
        ],
    )
    axis.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    axis.grid(which="minor", color="#f7f9fc", linewidth=4)
    axis.tick_params(which="minor", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.suptitle(
        "Development controls separate two layout failures",
        x=0.025,
        y=0.98,
        ha="left",
        weight="bold",
        fontsize=19,
        color="#172d43",
    )
    fig.text(
        0.025,
        0.9,
        "Failed cases / checked cases · M4 Max, macOS 26.2 · retained embedding-patched PyTorch",
        color="#42566b",
        fontsize=11,
    )
    fig.text(
        0.025,
        0.12,
        "Matrix: same-dtype PyTorch CPU reference. Conv3d: saved PyTorch CPU FP64 reference.\n"
        "All four workers used identical operands. Diagnostic mode disables MPP; "
        "it does not validate the new patch.",
        color="#42566b",
        fontsize=10,
        linespacing=1.6,
    )
    fig.text(
        0.025,
        0.025,
        "328 raw archives audited  •  680 comparisons recomputed  •  "
        "82 common operand/reference sets  •  no timing or admission",
        color="#42566b",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.27, right=0.985, top=0.71, bottom=0.25)
    fig.savefig(output / "layout-diagnostic.png", dpi=180)
    fig.savefig(output / "layout-diagnostic.svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    audit = auditor.audit(workspace / "work/gpu-v4-owned/layout-diagnostic-v1")
    sources = receipt_sources(workspace)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "array-audit.json", audit)
    with (output / "layout-cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit["case_rows"][0]))
        writer.writeheader()
        writer.writerows(audit["case_rows"])
    with (output / "layout-counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit["workers"][0]))
        writer.writeheader()
        writer.writerows(audit["workers"])
    catalog = []
    for source in sources:
        relative = source.relative_to(workspace / "work/gpu-v4-owned")
        copied = output / "receipts" / relative
        if source.suffix == ".py":
            copied = copied.with_suffix(".py.txt")
        copied.parent.mkdir(parents=True, exist_ok=True)
        before = auditor.digest(source)
        shutil.copyfile(source, copied)
        auditor.require(
            before == auditor.digest(copied) == auditor.digest(source), "Copy changed bytes"
        )
        catalog.append(
            {
                "original_workspace_path": source.relative_to(workspace).as_posix(),
                "report_copy": copied.relative_to(output).as_posix(),
                "bytes": copied.stat().st_size,
                "sha256": before,
            }
        )
    write_json(
        output / "receipt-catalog.json",
        {
            "schema": "gpu-owned-development-receipt-catalog-v1",
            "files": catalog,
            "original_receipt_paths_retained_verbatim": True,
            "raw_npz_copied": False,
            "raw_npz_location": "work/gpu-v4-owned/layout-diagnostic-v1; "
            "see array-audit.json bindings",
            "scope": "Original runtime attempts, failures and layout case receipts; "
            "not formal admission",
        },
    )
    draw(audit, output)
    source_paths = [
        Path(__file__),
        Path(auditor.__file__),
        Path(__file__).with_name("test_audit_layout_diagnostic.py"),
    ]
    outputs = [path for path in output.iterdir() if path.is_file()]
    write_json(
        output / "provenance.json",
        {
            "schema": "gpu-owned-development-report-provenance-v1",
            "formal_eligible": False,
            "formal_fits": 0,
            "admission_stream_opened": False,
            "raw_layout_summary_sha256": auditor.SUMMARY_SHA256,
            "runtime_summary_sha256": RUNTIME_SUMMARIES,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "sources": [
                {"path": path.relative_to(workspace).as_posix(), "sha256": auditor.digest(path)}
                for path in source_paths
            ],
            "outputs": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": auditor.digest(path)}
                for path in sorted(outputs)
            ],
            "cpu_only_negative_controls": "15 unittest cases; separate test execution is required",
            "qualification": "Evidence audit passed; "
            "candidate binary and complete study remain unqualified",
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "copied_receipts": len(catalog),
                "receipt_bytes": sum(item["bytes"] for item in catalog),
                "audit_passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
