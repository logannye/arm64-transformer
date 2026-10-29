"""Summarize completed baseline diagnostics; inspect saved arrays, run no model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_gpu_v2_engineering import (  # noqa: E402
    Receipts,
    baseline_rows,
    digest,
    require,
    safe_file,
)


def parameter_layout(width):
    """Flat layout from the frozen v2 model contract; checked against array size."""
    sizes = (
        ("token", 128 * width),
        ("position", 32 * width),
        ("ln1_weight", width),
        ("ln1_bias", width),
        ("qkv_weight", 3 * width * width),
        ("qkv_bias", 3 * width),
        ("attention_weight", width * width),
        ("attention_bias", width),
        ("ln2_weight", width),
        ("ln2_bias", width),
        ("ff1_weight", 2 * width * width),
        ("ff1_bias", 2 * width),
        ("ff2_weight", 2 * width * width),
        ("ff2_bias", width),
        ("ln3_weight", width),
        ("ln3_bias", width),
        ("head_weight", 128 * width),
        ("head_bias", 128),
    )
    result, cursor = {}, 0
    for name, count in sizes:
        result[name] = (cursor, count)
        cursor += count
    return result, cursor


def fingerprint(array):
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def localize(root, worker, receipts):
    import numpy as np

    layout, count = parameter_layout(worker["width"])
    for case in worker["cases"]:
        if not case["name"].startswith("full_model/"):
            continue
        for execution in case["executions"]:
            comparison = execution.get("exact", {}).get("gradient")
            if comparison is None or comparison.get("passed") is True:
                continue
            relative = f"{case['name']}/first-divergence-{execution['execution']:02}.npz"
            paths = [safe_file(root, name) for name in (case["first_arrays"], relative)]
            with (
                np.load(paths[0], allow_pickle=False) as first,
                np.load(paths[1], allow_pickle=False) as second,
            ):
                expected, actual = first["gradient"], second["gradient"]
                require(
                    expected.shape == actual.shape == (count,), "Unexpected flat gradient layout"
                )
                require(
                    fingerprint(actual) == comparison["actual"]
                    and fingerprint(expected) == comparison["expected"],
                    "Retained gradient fingerprint mismatch",
                )
                changed = actual.view(np.uint32) != expected.view(np.uint32)
                indices = np.flatnonzero(changed)
                require(
                    int(indices.size) == comparison["different_coordinates"],
                    "Mismatch count changed",
                )
                counts = {
                    name: int(changed[start : start + size].sum())
                    for name, (start, size) in layout.items()
                }
                counts = {name: value for name, value in counts.items() if value}
                first_index = int(indices[0])
                parameter = next(
                    name
                    for name, (start, size) in layout.items()
                    if start <= first_index < start + size
                )
                result = {
                    "case": case["name"],
                    "execution": execution["execution"],
                    "different_coordinates": int(indices.size),
                    "changed_parameter_counts": counts,
                    "first_flat_index": first_index,
                    "first_parameter": parameter,
                    "first_parameter_flat_offset": first_index - layout[parameter][0],
                    "max_absolute_difference": float(
                        np.max(np.abs(actual.astype(np.float64) - expected))
                    ),
                    "scope": (
                        "First divergent full-gradient case in process 0; "
                        "not an exhaustive localization of every shape"
                    ),
                }
            for path in paths:
                receipts.sources[
                    "baseline/" + str(path.relative_to(Path(root).resolve().parent))
                ] = digest(path)
            return result
    return None


def summarize(root, output):
    root, output = Path(root), Path(output)
    receipts = Receipts()
    rows = baseline_rows(root, receipts)
    for row in rows:
        stages = {}
        for process in range(2):
            relative = f"{row['engine']}-w{row['width']}-process{process}/worker.json"
            worker = receipts.read(root, relative, "baseline")
            require(
                worker["source_sha256"]["transformer/spec.py"]
                == digest(ROOT / "transformer/spec.py"),
                "Saved worker layout source differs",
            )
            if process == 0:
                row["first_full_gradient_divergence"] = localize(
                    root / Path(relative).parent, worker, receipts
                )
            for case in worker["cases"]:
                stage = case["name"].split("/")[0]
                data = stages.setdefault(
                    stage,
                    {
                        "cases": 0,
                        "complete": 0,
                        "exact_failed_cases": 0,
                        "runtime_failed_cases": 0,
                        "accuracy_failed_cases": 0,
                        "divergent_output_executions": Counter(),
                    },
                )
                data["cases"] += 1
                data["complete"] += int(case.get("complete") is True)
                data["exact_failed_cases"] += int(
                    case.get("complete") is True and case.get("exact_passed") is False
                )
                data["accuracy_failed_cases"] += int(
                    case.get("complete") is True and case.get("accuracy_passed") is False
                )
                data["runtime_failed_cases"] += int(
                    any("error" in item for item in case["executions"])
                )
                for execution in case["executions"]:
                    data["divergent_output_executions"].update(
                        key
                        for key, value in execution.get("exact", {}).items()
                        if value.get("passed") is False
                    )
        row["stages"] = stages
    result = {
        "schema": 1,
        "scope": "saved_baseline_diagnostics_only_no_model_execution",
        "standard_baselines_qualified": False,
        "formal_admission_opened": False,
        "workers": 24,
        "rows": rows,
        "source_artifacts": receipts.sources,
        "summarizer_sha256": digest(__file__),
        "layout_source_sha256": digest(ROOT / "transformer/spec.py"),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "| Engine | Width | Completed cases | Exact-failed complete cases | "
        "Runtime errors | FP64 accuracy | First gradient changes |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        location = row["first_full_gradient_divergence"]
        changed = (
            ", ".join(
                f"{name}: {count}" for name, count in location["changed_parameter_counts"].items()
            )
            if location
            else "unavailable"
        )
        lines.append(
            f"| {row['engine']} | {row['width']} | {row['completed_cases']}/{row['cases']} | "
            f"{len(row['exact_failed_cases'])} | {len(row['runtime_errors'])} | "
            f"{row['accuracy_status']} | {changed} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.baseline, args.output)
    print(args.output / "summary.md")


if __name__ == "__main__":
    main()
